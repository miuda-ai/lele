//! Fusions of QDQ (QuantizeLinear / DequantizeLinear) graphs.

use crate::compiler::ir::{DType, Graph, Op};
use crate::model::onnx_proto::TensorProto;

/// Collapses `QuantizeLinear` → `DequantizeLinear` round trips into a single
/// `FakeQuantizeLinear` node.
///
/// QDQ graphs wrap every quantized activation in such a pair. Because both
/// halves share a scale and zero point, the pair does not change the tensor
/// type — it only snaps values onto the quantization grid — so the two
/// passes can be done in one, halving the memory traffic and removing the
/// intermediate buffer. Constant-foldable pairs (quantized weights) are
/// already gone by the time this runs.
///
/// A pair is only fused when the intermediate tensor has exactly one
/// consumer, that consumer is the `DequantizeLinear`, and both nodes agree
/// on scale, zero point, `axis` and `block_size`.
pub(crate) fn fuse_qdq(g: &mut Graph) {
    // (QuantizeLinear, DequantizeLinear)
    let mut pairs = Vec::new();
    for (dq_id, dq) in g.nodes() {
        let Op::DequantizeLinear(dq_attrs) = &dq.op else {
            continue;
        };
        let Some(q_out) = dq.input(0) else {
            continue;
        };
        let Some(q_id) = g.value(q_out).producer() else {
            continue;
        };
        let q = g.node(q_id);
        let Op::QuantizeLinear(q_attrs) = &q.op else {
            continue;
        };
        if g.uses(q_out) != 1
            || q.input(1) != dq.input(1)
            || q.input(2) != dq.input(2)
            || q_attrs.axis != dq_attrs.axis
            || q_attrs.block_size != dq_attrs.block_size
        {
            continue;
        }
        pairs.push((q_id, dq_id));
    }

    // Collapse onto the QuantizeLinear: it already has both operands in
    // scope and keeps the `output_dtype` attribute that fixes the
    // saturation bounds. Only the output comes from the consumer.
    for (q_id, dq_id) in pairs {
        let out = g.node(dq_id).output(0);
        g.remove_node(dq_id);
        g.set_output(q_id, 0, out);
        let q = g.node_mut(q_id);
        if let Op::QuantizeLinear(attrs) = &q.op {
            q.op = Op::FakeQuantizeLinear(attrs.clone());
        }
    }
}

/// Rewrites `MatMul(dequantized activation, DequantizeLinear(int8 weight))`
/// into a single `QLinearMatMulI8` node that keeps the weight in int8.
///
/// This has to run *before* constant folding. Folding would turn the
/// weight's `DequantizeLinear` into an f32 constant, which is exactly the
/// information an integer kernel needs and cannot recover. Dropping the
/// node here means folding never sees it and the int8 initializer plus its
/// scale reach code generation intact.
///
/// The pattern is only taken when the arithmetic actually reduces to an
/// integer GEMM with a single correction term:
///
/// * the weight is a 2-D int8 initializer, quantized symmetrically (zero
///   point absent or all zero) with a per-output-channel or per-tensor
///   scale, i.e. `axis` selects the N dimension;
/// * the activation comes from a `DequantizeLinear` with a scalar scale and
///   a uint8 scalar zero point, so `a_zp * colsum(w)` is the whole
///   correction;
/// * the dequantized weight feeds nothing but this one `MatMul`.
///
/// The activation's own `DequantizeLinear` is left alone: it commonly feeds
/// several matmuls (one per attention projection), and [`fuse_qdq`] still
/// collapses it later.
pub(crate) fn fuse_qdq_matmul(g: &mut Graph) {
    // Every byte of an initializer is zero, so the tensor is too. Zero
    // points live in `raw_data` or `int32_data` depending on the exporter.
    let all_zero = |t: &TensorProto| {
        t.raw_data.iter().all(|&b| b == 0)
            && t.int32_data.iter().all(|&v| v == 0)
            && t.float_data.iter().all(|&v| v == 0.0)
    };
    let numel = |t: &TensorProto| t.dims.iter().product::<i64>().max(1);

    // (MatMul, weight DequantizeLinear, new MatMul inputs)
    let mut rewrites = Vec::new();
    for (mm_id, mm) in g.nodes() {
        if !matches!(mm.op, Op::MatMul) {
            continue;
        }
        let (Some(a), Some(wd)) = (mm.input(0), mm.input(1)) else {
            continue;
        };
        let Some(dq_w_id) = g.value(wd).producer() else {
            continue;
        };
        let dq_w = g.node(dq_w_id);
        let Op::DequantizeLinear(dq_w_attrs) = &dq_w.op else {
            continue;
        };
        if g.uses(wd) != 1 {
            continue;
        }
        let (Some(w), Some(w_scale)) = (dq_w.input(0), dq_w.input(1)) else {
            continue;
        };
        let Some(w_data) = g.data(w) else {
            continue;
        };
        // u8 weights would need a second correction term.
        if DType::from_onnx(w_data.data_type) != DType::I8 || w_data.dims.len() != 2 {
            continue;
        }
        let Some(w_scale_data) = g.data(w_scale) else {
            continue;
        };
        let scale_len = numel(w_scale_data);
        if scale_len != 1 && scale_len != w_data.dims[1] {
            continue;
        }
        // Per-channel scale must select the output dimension; a scale along
        // K cannot be pulled out of the accumulation.
        if scale_len != 1 && dq_w_attrs.axis != 1 {
            continue;
        }
        if let Some(zp) = dq_w.input(2)
            && !g.data(zp).is_some_and(all_zero)
        {
            continue;
        }

        let Some(dq_a) = g.producer(a) else {
            continue;
        };
        if !matches!(dq_a.op, Op::DequantizeLinear(_)) {
            continue;
        }
        let (Some(a_scale), Some(a_zp)) = (dq_a.input(1), dq_a.input(2)) else {
            continue;
        };
        if !g.data(a_scale).is_some_and(|t| numel(t) == 1) {
            continue;
        }
        // UINT8 is the operand type `vpdpbusd` takes for A.
        if !g
            .data(a_zp)
            .is_some_and(|t| DType::from_onnx(t.data_type) == DType::U8 && numel(t) == 1)
        {
            continue;
        }

        let inputs = vec![Some(a), Some(a_scale), Some(a_zp), Some(w), Some(w_scale)];
        rewrites.push((mm_id, dq_w_id, inputs));
    }

    for (mm_id, dq_w_id, inputs) in rewrites {
        g.set_inputs(mm_id, inputs);
        g.node_mut(mm_id).op = Op::QLinearMatMulI8;
        g.remove_node(dq_w_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::onnx_proto::{AttributeProto, GraphProto, NodeProto, ValueInfoProto};

    fn node(op: &str, inputs: &[&str], output: &str, attrs: &[(&str, i64)]) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            attribute: attrs
                .iter()
                .map(|(name, i)| AttributeProto {
                    name: name.to_string(),
                    i: *i,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn output(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Runs `pass` on `g` through the IR.
    fn run(g: GraphProto, pass: fn(&mut Graph)) -> GraphProto {
        let mut ir = Graph::from_onnx(g);
        pass(&mut ir);
        ir.into_onnx()
    }

    fn op_types(g: &GraphProto) -> Vec<&str> {
        g.node.iter().map(|n| n.op_type.as_str()).collect()
    }

    mod fake_quantize {
        use super::*;

        /// Graph of `nodes` whose last node's output is the only graph output.
        fn graph(nodes: Vec<NodeProto>) -> GraphProto {
            let last = nodes.last().unwrap().output[0].clone();
            GraphProto {
                node: nodes,
                output: vec![output(&last)],
                ..Default::default()
            }
        }

        #[test]
        fn fuses_matching_quantize_dequantize_pair() {
            let g = run(
                graph(vec![
                    node(
                        "QuantizeLinear",
                        &["x", "s", "z"],
                        "q",
                        &[("output_dtype", 2)],
                    ),
                    node("DequantizeLinear", &["q", "s", "z"], "y", &[]),
                ]),
                fuse_qdq,
            );
            assert_eq!(op_types(&g), ["FakeQuantizeLinear"]);
            // The fused node keeps the producer's operands and attributes but
            // takes the consumer's output name.
            assert_eq!(g.node[0].input, ["x", "s", "z"]);
            assert_eq!(g.node[0].output, ["y"]);
            assert_eq!(g.node[0].attribute.len(), 1);
            assert_eq!(g.node[0].attribute[0].name, "output_dtype");
        }

        #[test]
        fn fuses_pair_that_is_not_adjacent() {
            // Hoisting the DequantizeLinear onto the QuantizeLinear is safe
            // because they share their scale and zero-point operands.
            let g = run(
                graph(vec![
                    node("QuantizeLinear", &["x", "s", "z"], "q", &[]),
                    node("Relu", &["other"], "unrelated", &[]),
                    node("DequantizeLinear", &["q", "s", "z"], "y", &[]),
                    node("Relu", &["y"], "out", &[]),
                ]),
                fuse_qdq,
            );
            assert_eq!(op_types(&g), ["FakeQuantizeLinear", "Relu", "Relu"]);
            assert_eq!(g.node[0].output, ["y"]);
        }

        #[test]
        fn keeps_pair_when_quantized_tensor_has_another_consumer() {
            // `q` also feeds MatMulInteger, so the integer codes are still
            // needed.
            let g = run(
                graph(vec![
                    node("QuantizeLinear", &["x", "s", "z"], "q", &[]),
                    node("MatMulInteger", &["q", "w"], "m", &[]),
                    node("DequantizeLinear", &["q", "s", "z"], "y", &[]),
                    node("Add", &["y", "m"], "out", &[]),
                ]),
                fuse_qdq,
            );
            assert_eq!(
                op_types(&g),
                ["QuantizeLinear", "MatMulInteger", "DequantizeLinear", "Add"]
            );
        }

        #[test]
        fn keeps_pair_when_quantized_tensor_is_read_by_a_subgraph() {
            let then_branch = GraphProto {
                output: vec![output("q")],
                ..Default::default()
            };
            let mut if_node = node("If", &["cond"], "r", &[]);
            if_node.attribute = ["then_branch", "else_branch"]
                .map(|name| AttributeProto {
                    name: name.to_string(),
                    g: Some(then_branch.clone()),
                    ..Default::default()
                })
                .to_vec();
            let mut g = graph(vec![
                node("QuantizeLinear", &["x", "s", "z"], "q", &[]),
                if_node,
                node("DequantizeLinear", &["q", "s", "z"], "y", &[]),
            ]);
            g.output.push(output("r"));
            let g = run(g, fuse_qdq);
            assert_eq!(op_types(&g), ["QuantizeLinear", "If", "DequantizeLinear"]);
        }

        #[test]
        fn keeps_pair_when_quantized_tensor_is_a_graph_output() {
            let mut g = graph(vec![
                node("QuantizeLinear", &["x", "s", "z"], "q", &[]),
                node("DequantizeLinear", &["q", "s", "z"], "y", &[]),
            ]);
            g.output.push(output("q"));
            let g = run(g, fuse_qdq);
            assert_eq!(op_types(&g), ["QuantizeLinear", "DequantizeLinear"]);
        }

        #[test]
        fn keeps_pair_with_mismatched_quantization_params() {
            // Different scale: not a round trip, the values really do change.
            let g = run(
                graph(vec![
                    node("QuantizeLinear", &["x", "s1", "z"], "q", &[]),
                    node("DequantizeLinear", &["q", "s2", "z"], "y", &[]),
                ]),
                fuse_qdq,
            );
            assert_eq!(op_types(&g), ["QuantizeLinear", "DequantizeLinear"]);

            // Different axis: the parameters broadcast over different slices.
            let g = run(
                graph(vec![
                    node("QuantizeLinear", &["x", "s", "z"], "q", &[("axis", 0)]),
                    node("DequantizeLinear", &["q", "s", "z"], "y", &[("axis", 1)]),
                ]),
                fuse_qdq,
            );
            assert_eq!(op_types(&g), ["QuantizeLinear", "DequantizeLinear"]);

            // Zero point present on only one side.
            let g = run(
                graph(vec![
                    node("QuantizeLinear", &["x", "s", "z"], "q", &[]),
                    node("DequantizeLinear", &["q", "s"], "y", &[]),
                ]),
                fuse_qdq,
            );
            assert_eq!(op_types(&g), ["QuantizeLinear", "DequantizeLinear"]);
        }
    }

    mod qlinear_matmul {
        use super::*;

        fn init(name: &str, data_type: i32, dims: &[i64], raw: Vec<u8>) -> TensorProto {
            TensorProto {
                name: name.to_string(),
                data_type,
                dims: dims.to_vec(),
                raw_data: raw,
                ..Default::default()
            }
        }

        /// int8 weight, per-channel f32 scale, uint8 activation scale/zero
        /// point: the shapes `fuse_qdq_matmul` is meant to accept.
        fn initializers() -> Vec<TensorProto> {
            vec![
                init("w", 3, &[4, 2], vec![1u8; 8]),
                init("w_scale", 1, &[2], vec![0u8; 8]),
                init("w_zp", 3, &[2], vec![0u8; 2]),
                init("a_scale", 1, &[], vec![0u8; 4]),
                init("a_zp", 2, &[], vec![7u8]),
            ]
        }

        fn graph(nodes: Vec<NodeProto>) -> GraphProto {
            let last = nodes.last().unwrap().output[0].clone();
            GraphProto {
                node: nodes,
                initializer: initializers(),
                output: vec![output(&last)],
                ..Default::default()
            }
        }

        /// Activation dequantize, weight dequantize, matmul.
        fn nodes() -> Vec<NodeProto> {
            vec![
                node("DequantizeLinear", &["xq", "a_scale", "a_zp"], "a", &[]),
                node(
                    "DequantizeLinear",
                    &["w", "w_scale", "w_zp"],
                    "wd",
                    &[("axis", 1)],
                ),
                node("MatMul", &["a", "wd"], "y", &[]),
            ]
        }

        fn initializer<'g>(g: &'g mut GraphProto, name: &str) -> &'g mut TensorProto {
            g.initializer.iter_mut().find(|t| t.name == name).unwrap()
        }

        #[test]
        fn rewrites_matmul_against_dequantized_int8_weight() {
            let g = run(graph(nodes()), fuse_qdq_matmul);
            // The weight's DequantizeLinear is gone; the activation's stays.
            assert_eq!(op_types(&g), ["DequantizeLinear", "QLinearMatMulI8"]);
            assert_eq!(
                g.node[1].input,
                ["a", "a_scale", "a_zp", "w", "w_scale"],
                "the fused node carries both quantizations' parameters"
            );
            assert_eq!(g.node[1].output, ["y"]);
        }

        #[test]
        fn keeps_matmul_when_weight_zero_point_is_not_zero() {
            // An asymmetric weight needs a second correction term the kernel
            // does not compute.
            let mut g = graph(nodes());
            initializer(&mut g, "w_zp").raw_data = vec![3u8, 0u8];
            let g = run(g, fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }

        #[test]
        fn keeps_matmul_when_weight_is_uint8() {
            let mut g = graph(nodes());
            initializer(&mut g, "w").data_type = 2;
            let g = run(g, fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }

        #[test]
        fn keeps_matmul_when_weight_scale_runs_along_k() {
            // axis 0 scales each input channel, which cannot be lifted out of
            // the accumulation the way an output-channel scale can.
            let mut g = graph(nodes());
            g.node[1].attribute[0].i = 0;
            initializer(&mut g, "w_scale").dims = vec![4];
            let g = run(g, fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }

        #[test]
        fn fuses_when_weight_scale_is_per_tensor() {
            // A single scale has no axis to disagree about.
            let mut g = graph(nodes());
            g.node[1].attribute[0].i = 0;
            initializer(&mut g, "w_scale").dims = vec![];
            let g = run(g, fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"QLinearMatMulI8"));
        }

        #[test]
        fn keeps_matmul_when_dequantized_weight_has_another_consumer() {
            // Dropping the DequantizeLinear would leave the other consumer
            // without its input.
            let mut n = nodes();
            n.push(node("Relu", &["wd"], "other", &[]));
            let g = run(graph(n), fuse_qdq_matmul);
            assert!(op_types(&g).contains(&"MatMul"));
            assert_eq!(
                op_types(&g)
                    .iter()
                    .filter(|o| **o == "DequantizeLinear")
                    .count(),
                2
            );
        }

        #[test]
        fn keeps_matmul_when_activation_is_not_dequantized() {
            // Without the activation's scale and zero point there is nothing
            // to requantize against.
            let mut n = nodes();
            n[0] = node("Relu", &["xq"], "a", &[]);
            let g = run(graph(n), fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }

        #[test]
        fn keeps_matmul_when_activation_zero_point_is_int8() {
            // The integer kernel multiplies unsigned activations by signed
            // weights.
            let mut g = graph(nodes());
            initializer(&mut g, "a_zp").data_type = 3;
            let g = run(g, fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }

        #[test]
        fn keeps_matmul_when_activation_scale_is_per_channel() {
            let mut g = graph(nodes());
            initializer(&mut g, "a_scale").dims = vec![4];
            let g = run(g, fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }

        #[test]
        fn keeps_matmul_between_two_activations() {
            // Attention scores: both operands are dequantized activations, so
            // there is no static weight to pack.
            let mut n = nodes();
            n[1] = node("DequantizeLinear", &["vq", "a_scale", "a_zp"], "wd", &[]);
            let g = run(graph(n), fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }
    }
}
