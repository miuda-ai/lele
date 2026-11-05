//! Fusions of QDQ (QuantizeLinear / DequantizeLinear) graphs.

use crate::compiler::ir::{Binary, DType, Graph, NodeId, Op, ValueId};
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
///
/// A `MatMul` of two such activations (attention's queries by keys, and
/// probabilities by values) becomes a `QLinearMatMulAct`, which quantizes both
/// back to their codes and multiplies them in integers.
pub(crate) fn fuse_qdq_matmul(g: &mut Graph) {
    fuse_qdq_matmul_weight(g);
    fuse_qdq_matmul_act(g);
}

/// The scale and zero point of `v` if it is a `DequantizeLinear` of a
/// computed tensor, with a scalar scale and a scalar uint8 zero point: an
/// activation the integer kernels can quantize back exactly.
fn dequantized_activation(g: &Graph, v: ValueId) -> Option<(ValueId, ValueId)> {
    let dq = g.producer(v)?;
    if !matches!(dq.op, Op::DequantizeLinear(_)) || g.data(dq.input(0)?).is_some() {
        return None;
    }
    let (scale, zp) = (dq.input(1)?, dq.input(2)?);
    let numel = |t: &TensorProto| t.dims.iter().product::<i64>().max(1);
    let scalar_scale = g.data(scale).is_some_and(|t| numel(t) == 1);
    // UINT8 is the operand type `vpdpbusd` takes for A.
    let u8_zp = g
        .data(zp)
        .is_some_and(|t| DType::from_onnx(t.data_type) == DType::U8 && numel(t) == 1);
    (scalar_scale && u8_zp).then_some((scale, zp))
}

fn fuse_qdq_matmul_act(g: &mut Graph) {
    let mut rewrites = Vec::new();
    for (mm_id, mm) in g.nodes() {
        if !matches!(mm.op, Op::MatMul) {
            continue;
        }
        let (Some(a), Some(b)) = (mm.input(0), mm.input(1)) else {
            continue;
        };
        let (Some((a_scale, a_zp)), Some((b_scale, b_zp))) =
            (dequantized_activation(g, a), dequantized_activation(g, b))
        else {
            continue;
        };
        let inputs = [a, a_scale, a_zp, b, b_scale, b_zp].map(Some).to_vec();
        rewrites.push((mm_id, inputs));
    }
    for (mm_id, inputs) in rewrites {
        g.set_inputs(mm_id, inputs);
        g.node_mut(mm_id).op = Op::QLinearMatMulAct;
    }
}

fn fuse_qdq_matmul_weight(g: &mut Graph) {
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

/// Inputs of `QLinearMatMulI8` beyond the five it is made with: the scale and
/// zero point of the pair that follows the product, and the bias added after
/// it.
pub(crate) const I8_REQUANT: usize = 5;
pub(crate) const I8_BIAS: usize = 7;
/// The scale and zero point of the pair after a `QLinearMatMulAct`.
pub(crate) const ACT_REQUANT: usize = 6;

/// Folds the `QuantizeLinear` -> `DequantizeLinear` pairs (by now
/// `FakeQuantizeLinear`) around the integer matmuls into them. Each is a pass
/// over an activation, and all of them leave the result as it was, bit for
/// bit:
///
/// * a pair whose every reader is an integer matmul quantizing it with the
///   pair's own scale and zero point goes: quantizing the input gives the
///   codes the pair would have, and quantizing those again gives them back;
/// * the uint8 pair after a matmul's product, and a constant bias added after
///   that, become the product's epilogue, which rounds each step as the nodes
///   did.
pub(crate) fn fold_qdq_into_matmuls(g: &mut Graph) {
    drop_requantized_inputs(g);
    fold_output_pairs(g);
    fold_biases(g);
}

/// The scale and zero point of a per-tensor uint8 `FakeQuantizeLinear`.
fn u8_round_trip(g: &Graph, fq: NodeId) -> Option<(ValueId, ValueId)> {
    let n = g.node(fq);
    let Op::FakeQuantizeLinear(attrs) = &n.op else {
        return None;
    };
    let (scale, zp) = (n.input(1)?, n.input(2)?);
    let numel = |t: &TensorProto| t.dims.iter().product::<i64>().max(1);
    let scalar_scale = g
        .data(scale)
        .is_some_and(|t| DType::from_onnx(t.data_type) == DType::F32 && numel(t) == 1);
    let u8_zp = g
        .data(zp)
        .is_some_and(|t| DType::from_onnx(t.data_type) == DType::U8 && numel(t) == 1);
    let u8_out = attrs.output_dtype.is_none_or(|t| t == DType::U8);
    (scalar_scale && u8_zp && u8_out && attrs.block_size == 0).then_some((scale, zp))
}

/// The sole reader of `v`, unless `v` is also a graph output.
fn sole_reader(g: &Graph, v: ValueId) -> Option<NodeId> {
    match g.value(v).consumers() {
        &[c] if g.uses(v) == 1 => Some(c),
        _ => None,
    }
}

/// `node` quantizes every operand it reads from `v` with `scale` and `zp`.
fn quantizes(g: &Graph, node: NodeId, v: ValueId, scale: ValueId, zp: ValueId) -> bool {
    let n = g.node(node);
    // The quantized operands, each followed by its scale and zero point.
    let operands: &[usize] = match n.op {
        Op::QLinearMatMulI8 => &[0],
        Op::QLinearMatMulAct => &[0, 3],
        _ => return false,
    };
    n.inputs().iter().enumerate().all(|(i, &input)| {
        input != Some(v)
            || operands.contains(&i) && n.input(i + 1) == Some(scale) && n.input(i + 2) == Some(zp)
    })
}

fn drop_requantized_inputs(g: &mut Graph) {
    let mut drops = Vec::new();
    for (id, n) in g.nodes() {
        if !matches!(n.op, Op::FakeQuantizeLinear(_)) {
            continue;
        }
        let (Some(x), Some(y), Some(scale), Some(zp)) =
            (n.input(0), n.output(0), n.input(1), n.input(2))
        else {
            continue;
        };
        let readers = g.value(y).consumers();
        if readers.is_empty()
            || g.uses(y) != readers.len()
            || !readers.iter().all(|&r| quantizes(g, r, y, scale, zp))
        {
            continue;
        }
        drops.push((id, x, y));
    }
    for (id, x, y) in drops {
        let mut readers = g.value(y).consumers().to_vec();
        readers.dedup();
        for r in readers {
            let inputs = g
                .node(r)
                .inputs()
                .iter()
                .map(|&v| if v == Some(y) { Some(x) } else { v });
            g.set_inputs(r, inputs.collect());
        }
        g.remove_node(id);
    }
}

/// Moves the output of `consumer` onto `node` and removes `consumer`.
fn absorb(g: &mut Graph, node: NodeId, consumer: NodeId) {
    let out = g.node(consumer).output(0);
    g.remove_node(consumer);
    g.set_output(node, 0, out);
}

/// Sets input `i` of `node`, padding with omitted operands.
fn set_input(g: &mut Graph, node: NodeId, i: usize, v: ValueId) {
    let mut inputs = g.node(node).inputs().to_vec();
    if inputs.len() <= i {
        inputs.resize(i + 1, None);
    }
    inputs[i] = Some(v);
    g.set_inputs(node, inputs);
}

fn fold_output_pairs(g: &mut Graph) {
    let mut folds = Vec::new();
    for (id, n) in g.nodes() {
        let slot = match n.op {
            Op::QLinearMatMulI8 => I8_REQUANT,
            Op::QLinearMatMulAct => ACT_REQUANT,
            _ => continue,
        };
        let Some(fq) = n.output(0).and_then(|y| sole_reader(g, y)) else {
            continue;
        };
        if g.node(fq).input(0) != n.output(0) || n.input(slot).is_some() {
            continue;
        }
        if let Some((scale, zp)) = u8_round_trip(g, fq) {
            folds.push((id, fq, slot, scale, zp));
        }
    }
    for (id, fq, slot, scale, zp) in folds {
        absorb(g, id, fq);
        set_input(g, id, slot, scale);
        set_input(g, id, slot + 1, zp);
    }
}

fn fold_biases(g: &mut Graph) {
    let mut folds = Vec::new();
    for (id, n) in g.nodes() {
        if !matches!(n.op, Op::QLinearMatMulI8) || n.input(I8_BIAS).is_some() {
            continue;
        }
        let (Some(y), Some(w)) = (n.output(0), n.input(3)) else {
            continue;
        };
        let Some(add) = sole_reader(g, y) else {
            continue;
        };
        let a = g.node(add);
        if !matches!(a.op, Op::Binary(Binary::Add)) {
            continue;
        }
        let Some(bias) = (match (a.input(0), a.input(1)) {
            (Some(l), r) if l == y => r,
            (l, Some(r)) if r == y => l,
            _ => None,
        }) else {
            continue;
        };
        // One value per column of the weight, `[k, n]`.
        let n_cols = g.data(w).map(|t| t.dims[1]);
        let fits = g.data(bias).is_some_and(|t| {
            DType::from_onnx(t.data_type) == DType::F32 && n_cols.is_some_and(|n| t.dims == [n])
        });
        if bias != y && fits {
            folds.push((id, add, bias));
        }
    }
    for (id, add, bias) in folds {
        absorb(g, id, add);
        set_input(g, id, I8_BIAS, bias);
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
                        &[("output_dtype", DType::U8.to_onnx() as i64)],
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

        fn init(name: &str, dtype: DType, dims: &[i64], raw: Vec<u8>) -> TensorProto {
            TensorProto {
                name: name.to_string(),
                data_type: dtype.to_onnx(),
                dims: dims.to_vec(),
                raw_data: raw,
                ..Default::default()
            }
        }

        /// int8 weight, per-channel f32 scale, uint8 activation scale/zero
        /// point: the shapes `fuse_qdq_matmul` is meant to accept.
        fn initializers() -> Vec<TensorProto> {
            vec![
                init("w", DType::I8, &[4, 2], vec![1u8; 8]),
                init("w_scale", DType::F32, &[2], vec![0u8; 8]),
                init("w_zp", DType::I8, &[2], vec![0u8; 2]),
                init("a_scale", DType::F32, &[], vec![0u8; 4]),
                init("a_zp", DType::U8, &[], vec![7u8]),
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
            initializer(&mut g, "w").data_type = DType::U8.to_onnx();
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
            initializer(&mut g, "a_zp").data_type = DType::I8.to_onnx();
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

        /// Attention scores: both operands dequantized activations.
        fn activation_nodes() -> Vec<NodeProto> {
            let mut n = nodes();
            n[1] = node("DequantizeLinear", &["vq", "w_scale", "a_zp"], "wd", &[]);
            n
        }

        #[test]
        fn rewrites_matmul_between_two_dequantized_activations() {
            let mut g = graph(activation_nodes());
            initializer(&mut g, "w_scale").dims = vec![];
            let g = run(g, fuse_qdq_matmul);
            // Both DequantizeLinears stay, for `fuse_qdq` to collapse.
            assert_eq!(
                op_types(&g),
                ["DequantizeLinear", "DequantizeLinear", "QLinearMatMulAct"]
            );
            assert_eq!(
                g.node[2].input,
                ["a", "a_scale", "a_zp", "wd", "w_scale", "a_zp"]
            );
            assert_eq!(g.node[2].output, ["y"]);
        }

        #[test]
        fn keeps_matmul_between_activations_with_a_per_channel_scale() {
            // `w_scale` has two elements.
            let g = run(graph(activation_nodes()), fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }

        #[test]
        fn keeps_matmul_between_activations_with_an_int8_zero_point() {
            let mut g = graph(activation_nodes());
            initializer(&mut g, "w_scale").dims = vec![];
            initializer(&mut g, "a_zp").data_type = DType::I8.to_onnx();
            let g = run(g, fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }

        #[test]
        fn keeps_matmul_against_a_dequantized_uint8_weight() {
            // A constant B is no activation: folding dequantizes it once,
            // rather than the kernel quantizing it on every call.
            let mut n = nodes();
            n[1] = node("DequantizeLinear", &["w", "a_scale", "a_zp"], "wd", &[]);
            let mut g = graph(n);
            initializer(&mut g, "w").data_type = DType::U8.to_onnx();
            let g = run(g, fuse_qdq_matmul);
            assert_eq!(op_types(&g).last(), Some(&"MatMul"));
        }
    }

    mod fold_into_matmuls {
        use super::*;

        fn init(name: &str, dtype: DType, dims: &[i64]) -> TensorProto {
            // (The values do not matter, and no type is wider than 4 bytes.)
            let len = dims.iter().product::<i64>().max(1) as usize * 4;
            TensorProto {
                name: name.to_string(),
                data_type: dtype.to_onnx(),
                dims: dims.to_vec(),
                raw_data: vec![1; len],
                ..Default::default()
            }
        }

        /// `x`'s pair, an int8 `[4, 2]` weight, the product's pair and a bias.
        fn graph(extra: Vec<NodeProto>, outputs: &[&str]) -> GraphProto {
            let mut nodes = vec![
                node("QuantizeLinear", &["x", "s", "z"], "xq", &[]),
                node("DequantizeLinear", &["xq", "s", "z"], "a", &[]),
                node("DequantizeLinear", &["w", "w_scale"], "wd", &[("axis", 1)]),
                node("MatMul", &["a", "wd"], "y", &[]),
                node("QuantizeLinear", &["y", "ys", "yz"], "yq", &[]),
                node("DequantizeLinear", &["yq", "ys", "yz"], "yd", &[]),
                node("Add", &["bias", "yd"], "out", &[]),
            ];
            nodes.extend(extra);
            GraphProto {
                node: nodes,
                initializer: vec![
                    init("w", DType::I8, &[4, 2]),
                    init("w_scale", DType::F32, &[2]),
                    init("s", DType::F32, &[]),
                    init("z", DType::U8, &[]),
                    init("ys", DType::F32, &[]),
                    init("yz", DType::U8, &[]),
                    init("bias", DType::F32, &[2]),
                    init("bias_wide", DType::F32, &[4, 2]),
                ],
                output: outputs.iter().map(|o| output(o)).collect(),
                ..Default::default()
            }
        }

        fn fold(g: &mut Graph) {
            fuse_qdq_matmul(g);
            fuse_qdq(g);
            fold_qdq_into_matmuls(g);
        }

        #[test]
        fn folds_the_pairs_and_the_bias_into_an_int8_matmul() {
            let g = run(graph(vec![], &["out"]), fold);
            assert_eq!(op_types(&g), ["QLinearMatMulI8"]);
            assert_eq!(
                g.node[0].input,
                ["x", "s", "z", "w", "w_scale", "ys", "yz", "bias"]
            );
            assert_eq!(g.node[0].output, ["out"]);
        }

        #[test]
        fn keeps_an_input_pair_something_else_reads() {
            let g = run(
                graph(vec![node("Relu", &["a"], "r", &[])], &["out", "r"]),
                fold,
            );
            assert_eq!(
                op_types(&g),
                ["FakeQuantizeLinear", "QLinearMatMulI8", "Relu"]
            );
            assert_eq!(g.node[1].input[0], "a");
        }

        #[test]
        fn keeps_the_pair_after_a_product_something_else_reads() {
            let g = run(graph(vec![], &["out", "y"]), fold);
            assert_eq!(
                op_types(&g),
                ["QLinearMatMulI8", "FakeQuantizeLinear", "Add"]
            );
            assert_eq!(g.node[0].input.len(), 5);
        }

        #[test]
        fn keeps_a_bias_that_is_not_one_per_column() {
            let mut g = graph(vec![], &["out"]);
            g.node[6].input[0] = "bias_wide".to_string();
            let g = run(g, fold);
            assert_eq!(op_types(&g), ["QLinearMatMulI8", "Add"]);
            assert_eq!(g.node[0].input[5..], ["ys", "yz"]);
        }

        #[test]
        fn folds_the_pairs_around_a_product_of_activations() {
            let mut g = graph(
                vec![
                    node("QuantizeLinear", &["v", "s", "z"], "vq", &[]),
                    node("DequantizeLinear", &["vq", "s", "z"], "b", &[]),
                ],
                &["yd"],
            );
            g.node.retain(|n| n.op_type != "Add" && n.output[0] != "wd");
            g.node[2].input[1] = "b".to_string();
            let g = run(g, fold);
            assert_eq!(op_types(&g), ["QLinearMatMulAct"]);
            assert_eq!(g.node[0].input, ["x", "s", "z", "v", "s", "z", "ys", "yz"]);
            assert_eq!(g.node[0].output, ["yd"]);
        }
    }
}
