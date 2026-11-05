pub use self::alloc::Allocator;
pub(crate) use self::alloc::AnalysisData;
use crate::model::onnx_proto::{GraphProto, NodeProto};
use std::collections::HashMap;
use std::io::Write;
/// Handler callback for custom operator code generation.
/// Args:
/// - node: The ONNX NodeProto
/// - inputs: List of Rust variable names for inputs (e.g. `["op_1_out", "state_in"]`)
/// - outputs: List of Rust variable names for outputs (e.g. `["op_5_out"]`)
/// - buffer: The code string to access the output buffer (e.g. `"&mut ws.buf_3"` or `"&mut buf_X"`)
/// - writer: The output writer
/// - indent: Current indentation level
pub type OpHandler = Box<
    dyn Fn(&NodeProto, &[String], &[String], &str, &mut dyn Write, usize) -> std::io::Result<()>,
>;
pub type PatternMatcher = Box<dyn Fn(&[&NodeProto]) -> Option<usize>>;
/// Weights by sanitized name, as `(byte offset, byte length, shape, ONNX data
/// type)`.
pub type WeightTable = HashMap<String, (usize, usize, Vec<usize>, i32)>;
pub type PatternGenerator = Box<
    dyn Fn(
        &[&NodeProto],
        &WeightTable,
        Option<&Allocator>,
        &mut dyn Write,
        usize,
    ) -> std::io::Result<()>,
>;
pub struct Pattern {
    pub name: String,
    pub matcher: PatternMatcher,
    pub generator: PatternGenerator,
}
pub struct CompilationResult {
    pub code: String,
    pub weights: Vec<u8>,
    /// Content hash -> (offset, length) for every distinct weight blob written.
    /// Feed this back through [`Compiler::with_shared_weights`] so a second model
    /// that owns the same tensors reuses their bytes instead of duplicating them.
    pub content_hashes: HashMap<u64, (usize, usize)>,
}

/// A weights blob plus its content index, used to share tensors between models
/// (e.g. a prefill and a decode-step graph built from the same transformer).
pub struct SharedWeights {
    pub blob: Vec<u8>,
    pub content_hashes: HashMap<u64, (usize, usize)>,
}

impl SharedWeights {
    /// Seed a shared blob from a previously compiled model.
    pub fn from_result(result: &CompilationResult) -> Self {
        Self {
            blob: result.weights.clone(),
            content_hashes: result.content_hashes.clone(),
        }
    }
}
mod alloc;
pub mod generate;
pub mod ir;
pub mod ops;
mod passes;
mod weights;

pub(crate) use generate::*;

fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    result
}

/// Graph optimizations the compiler runs before code generation. All are off
/// in [`Optimizations::default`]; [`Optimizations::all`] turns every one on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Optimizations {
    /// Replaces nodes whose inputs are all known at compile time by
    /// initializers holding their result, computed with the kernels the
    /// generated code would run.
    pub constant_folding: bool,
    /// Infers tensor shapes and replaces shape computations (`Shape`,
    /// `Gather` of dimensions, ...) by constants where the dimensions are
    /// known, and `Reshape`/`Expand`/`ConstantOfShape` targets by
    /// dimensions of their inputs where they are not. With constant folding
    /// on, the results are folded further.
    pub shape_folding: bool,
    /// Removes nodes whose outputs nothing reads.
    pub dead_code_elimination: bool,
    /// Rewrites `MatMul` of a dequantized int8 weight into an integer
    /// matmul that keeps the weight in int8, and collapses
    /// `QuantizeLinear` → `DequantizeLinear` pairs into one node.
    pub qdq_fusion: bool,
    /// Fuses operator chains (LayerNorm, GELU, SiLU, Conv + bias/activation,
    /// MatMul + Add, dynamically quantized linear layers, ...) into single
    /// operators with fused kernels.
    pub operator_fusion: bool,
    /// Stores f16 weights read only as the right operand of matrix products
    /// in the order the matmul kernel reads them
    /// ([`pack_f16`](crate::kernels::matmul::pack_f16)), so that each product
    /// streams its weight from start to end. Not done when patterns are
    /// registered, whose generators read weights as stored in the model.
    pub weight_packing: bool,
}

impl Optimizations {
    /// Every optimization.
    pub fn all() -> Self {
        Self {
            constant_folding: true,
            shape_folding: true,
            dead_code_elimination: true,
            qdq_fusion: true,
            operator_fusion: true,
            weight_packing: true,
        }
    }
}

pub struct Compiler {
    pub(crate) overrides: HashMap<String, OpHandler>,
    pub(crate) model_name: String,
    pub(crate) patterns: Vec<Pattern>,
    pub(crate) custom_methods: Vec<String>,
    pub optimizations: Optimizations,
    /// Optional weights blob (and content index) reused from an earlier
    /// compilation, so identical tensors are not stored twice.
    pub(crate) shared_weights: std::cell::RefCell<Option<SharedWeights>>,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            overrides: HashMap::new(),
            model_name: "Model".to_string(),
            patterns: Vec::new(),
            custom_methods: Vec::new(),
            optimizations: Optimizations::default(),
            shared_weights: std::cell::RefCell::new(None),
        }
    }
    /// Reuse the tensors of an already compiled model. New (or different)
    /// tensors are appended after the shared blob; identical ones keep the
    /// offset they had in the shared blob, so several models can be loaded from
    /// a single weights file.
    pub fn with_shared_weights(self, shared: SharedWeights) -> Self {
        *self.shared_weights.borrow_mut() = Some(shared);
        self
    }
    /// Register a custom code generator for an operator type.
    /// The handler closure will be called instead of the default generation logic.
    pub fn with_override<F>(mut self, op: &str, handler: F) -> Self
    where
        F: Fn(&NodeProto, &[String], &[String], &str, &mut dyn Write, usize) -> std::io::Result<()>
            + 'static,
    {
        self.overrides.insert(op.to_string(), Box::new(handler));
        self
    }
    /// Register a pattern matcher and generator.
    pub fn with_pattern<M, G>(mut self, name: &str, matcher: M, generator: G) -> Self
    where
        M: Fn(&[&NodeProto]) -> Option<usize> + 'static,
        G: Fn(
                &[&NodeProto],
                &WeightTable,
                Option<&Allocator>,
                &mut dyn Write,
                usize,
            ) -> std::io::Result<()>
            + 'static,
    {
        self.patterns.push(Pattern {
            name: name.to_string(),
            matcher: Box::new(matcher),
            generator: Box::new(generator),
        });
        self
    }
    /// Add a custom method to the implementation block.
    pub fn with_custom_method(mut self, code: &str) -> Self {
        self.custom_methods.push(code.to_string());
        self
    }

    /// Enables every optimization ([`Optimizations::all`]).
    pub fn with_default_optimizations(self) -> Self {
        self.with_optimizations(Optimizations::all())
    }

    /// Sets the optimizations to run.
    pub fn with_optimizations(mut self, optimizations: Optimizations) -> Self {
        self.optimizations = optimizations;
        self
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.model_name = to_pascal_case(name);
        self
    }

    pub fn compile(
        &self,
        graph: &GraphProto,
    ) -> Result<CompilationResult, Box<dyn std::error::Error>> {
        let opt = self.optimizations;
        let mut ir = ir::Graph::from_onnx(graph.clone());
        if opt.qdq_fusion {
            // Must precede folding, which would otherwise consume the int8
            // weights.
            passes::fuse_qdq_matmul(&mut ir);
        }
        if opt.constant_folding {
            passes::fold_constants(&mut ir);
        }
        if opt.shape_folding {
            // Shape inference resolves the shape computations folding
            // could not, which may leave more to fold.
            passes::infer_dtypes(&mut ir);
            passes::fold_shapes(&mut ir);
            if opt.constant_folding {
                passes::fold_constants(&mut ir);
            }
        }
        if opt.dead_code_elimination {
            passes::eliminate_dead_code(&mut ir);
        }
        if opt.qdq_fusion {
            // Runs after folding so that only the activation pairs are left.
            passes::fuse_qdq(&mut ir);
        }
        if opt.operator_fusion {
            passes::fuse(&mut ir);
        }
        passes::infer_dtypes(&mut ir);
        let var_types = variable_types(&ir);
        let (allocator, analysis) = alloc::solve_allocation(&ir);
        // Patterns and overrides of the products would read packed weights as stored in
        // the model.
        let pack = opt.weight_packing
            && self.patterns.is_empty()
            && !["MatMul", "Linear"].iter().any(|op| self.overrides.contains_key(*op));
        // Start from a previously compiled blob when weights are shared. The
        // shared bytes stay where they are; anything new is appended after them,
        // so the resulting file is a superset of the first model's weights.
        let weights = weights::Weights::collect_shared(
            &ir,
            pack,
            self.shared_weights.borrow_mut().take(),
        );
        let (graph_inputs, graph_outputs) = (ir.input_infos(), ir.output_infos());
        let (ids, protos) = lower_nodes(&ir);
        let protos: Vec<&NodeProto> = protos.iter().collect();
        let nodes = Nodes {
            graph: &ir,
            ids: &ids,
            protos: &protos,
        };

        let mut chunk_functions = Vec::new(); // Will hold the code for fn chunk_0(...) { ... }
        let mut body_buffer = Vec::new(); // Will hold the code for forward() body
        let mut current_id = 0;
        generate_partitioned_graph(
            nodes,
            &mut chunk_functions,
            &mut body_buffer,
            2,
            &weights,
            Some(&allocator),
            Some(&analysis),
            &mut current_id,
            self,
            &graph_inputs,
            &graph_outputs,
            &var_types,
        )?;
        let body_str =
            String::from_utf8(body_buffer).map_err(|e| format!("UTF-8 error in body: {}", e))?;
        // Detect used buffers
        // We need to scan both body AND usage in chunk functions
        let mut all_code = body_str.clone();
        for func in &chunk_functions {
            all_code.push_str(func);
        }
        let mut used_buffers = std::collections::HashSet::new();
        let mut search_idx = 0;
        while let Some(idx) = all_code[search_idx..].find("ws.buf_") {
            let start = search_idx + idx + 7; // "ws.buf_".len()
            let mut end = start;
            while end < all_code.len() {
                let c = all_code.as_bytes()[end] as char;
                if !c.is_ascii_digit() {
                    break;
                }
                end += 1;
            }
            if end > start
                && let Ok(buf_idx) = all_code[start..end].parse::<usize>()
            {
                used_buffers.insert(buf_idx);
            }
            search_idx = end;
        }
        // 2. Generate Rust Code
        let mut code = Vec::new();
        // Header
        writeln!(&mut code, "// Auto-generated by lele_compiler")?;
        writeln!(&mut code, "#![allow(non_snake_case)]")?;
        writeln!(&mut code, "#![allow(unused_variables)]")?;
        writeln!(&mut code, "#![allow(dead_code)]")?;
        writeln!(&mut code, "#![allow(unused_parens)]")?;
        writeln!(&mut code, "#![allow(unused_mut)]")?;
        writeln!(&mut code, "#![allow(unused_imports)]")?;
        writeln!(&mut code, "#![allow(clippy::too_many_arguments)]")?;
        writeln!(&mut code, "use lele::tensor::TensorView;")?;
        writeln!(&mut code, "use lele::kernels::*;")?;
        writeln!(&mut code, "\n")?;
        let struct_name = &self.model_name;
        // Generate Workspace Struct
        writeln!(&mut code, "#[derive(Default)]")?;
        writeln!(&mut code, "pub struct {}Workspace {{", struct_name)?;
        for i in 0..allocator.num_buffers {
            if used_buffers.contains(&i) {
                writeln!(&mut code, "    pub buf_{}: Vec<f32>,", i)?;
            }
        }
        writeln!(&mut code, "}}")?;
        writeln!(&mut code, "impl {}Workspace {{", struct_name)?;
        writeln!(&mut code, "    pub fn new() -> Self {{")?;
        writeln!(&mut code, "        Self::default()")?;
        writeln!(&mut code, "    }}")?;
        writeln!(&mut code, "}}")?;
        // Main struct: wraps the weights and derefs to them, so the weight
        // accessors and fused ops in lele::runtime are called as `self.*`.
        writeln!(&mut code, "\npub struct {}<'a> {{", struct_name)?;
        writeln!(&mut code, "    weights: lele::runtime::Weights<'a>,")?;
        writeln!(&mut code, "}}")?;
        writeln!(
            &mut code,
            "\nimpl<'a> std::ops::Deref for {}<'a> {{",
            struct_name
        )?;
        writeln!(&mut code, "    type Target = lele::runtime::Weights<'a>;")?;
        writeln!(&mut code, "    fn deref(&self) -> &Self::Target {{")?;
        writeln!(&mut code, "        &self.weights")?;
        writeln!(&mut code, "    }}")?;
        writeln!(&mut code, "}}")?;
        writeln!(&mut code, "\nimpl<'a> {}<'a> {{", struct_name)?;
        writeln!(&mut code, "    pub fn new(data: &'a [u8]) -> Self {{")?;
        // f16 weights read as f32 are converted once, here, not on every read;
        // those the matmul kernels take as f16 are left as they are.
        let f16: Vec<_> = weights
            .f16_regions()
            .into_iter()
            .filter(|(offset, len)| {
                let read = format!("self.weight_f16({offset}, {len},");
                all_code.contains(&read) || self.custom_methods.iter().any(|m| m.contains(&read))
            })
            .collect();
        if f16.is_empty() {
            writeln!(
                &mut code,
                "        Self {{ weights: lele::runtime::Weights::new(data) }}"
            )?;
        } else {
            writeln!(
                &mut code,
                "        Self {{ weights: lele::runtime::Weights::new(data).with_f16(&{:?}) }}",
                f16
            )?;
        }
        writeln!(&mut code, "    }}")?;
        for method in &self.custom_methods {
            writeln!(&mut code, "{}", method)?;
        }
        // Generated Chunk Functions
        for func in &chunk_functions {
            writeln!(&mut code, "{}", func)?;
        }

        // Inference Function
        writeln!(&mut code, "\n    // Inference Entry Point")?;
        let args: Vec<String> = graph_inputs
            .iter()
            .map(|i| {
                let name = sanitize_name(&i.name);
                match var_types.get(&name) {
                    ElemType::F32 => format!("{}: TensorView<'a>", name),
                    ty => format!("{}: TensorView<'a, {}>", name, ty),
                }
            })
            .collect();
        // Generate return type
        let ret_types: Vec<String> = graph_outputs
            .iter()
            .map(|o| match var_types.get(&sanitize_name(&o.name)) {
                ElemType::F32 => "TensorView<'a>".to_string(),
                ty => format!("TensorView<'a, {}>", ty),
            })
            .collect();
        let ret_sig = if ret_types.len() == 1 {
            ret_types[0].clone()
        } else {
            format!("({})", ret_types.join(", "))
        };
        // forward (legacy)
        writeln!(
            &mut code,
            "    pub fn forward(&self, {}) -> {} {{",
            args.join(", "),
            ret_sig.replace("'a", "'static")
        )?;
        writeln!(
            &mut code,
            "        let mut ws = {}Workspace::new();",
            struct_name
        )?;
        let arg_names_str = graph_inputs
            .iter()
            .map(|i| sanitize_name(&i.name))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            &mut code,
            "        let res = self.forward_with_workspace(&mut ws, {});",
            arg_names_str
        )?;
        let ret_vals_owned: Vec<String> = if graph_outputs.len() == 1 {
            vec!["res.to_owned()".to_string()]
        } else {
            (0..graph_outputs.len())
                .map(|i| format!("res.{}.to_owned()", i))
                .collect()
        };
        let ret_str_owned = if ret_vals_owned.len() == 1 {
            ret_vals_owned[0].clone()
        } else {
            format!("({})", ret_vals_owned.join(", "))
        };
        writeln!(&mut code, "        {}", ret_str_owned)?;
        writeln!(&mut code, "    }}")?;

        // forward_with_workspace
        writeln!(
            &mut code,
            "    pub fn forward_with_workspace<'w>(&self, ws: &'w mut {}Workspace, {}) -> {} {{",
            struct_name,
            args.join(", ").replace("'a", "'w"),
            ret_sig.replace("'a", "'w")
        )?;
        // Generate Body
        code.extend_from_slice(body_str.as_bytes());
        // Return outputs
        let ret_vals: Vec<String> = graph_outputs
            .iter()
            .map(|o| {
                let name = sanitize_name(&o.name);
                if let Some(weight) = weights.get(&o.name) {
                    weight_expr(weight, var_types.get(&name))
                } else {
                    name
                }
            })
            .collect();
        if ret_vals.len() == 1 {
            writeln!(&mut code, "        {}", ret_vals[0])?;
        } else {
            writeln!(&mut code, "        ({})", ret_vals.join(", "))?;
        }
        writeln!(&mut code, "    }}")?; // End forward
        writeln!(&mut code, "}}")?; // End impl
        Ok(CompilationResult {
            code: String::from_utf8(code).map_err(|e| format!("UTF-8 error: {}", e))?,
            weights: weights.data,
            content_hashes: weights.by_content,
        })
    }
}
pub fn sanitize_name(name: &str) -> String {
    let s = name
        .replace(".", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "_");
    if s.chars().next().is_some_and(|c| c.is_numeric()) {
        format!("_{}", s)
    } else {
        s
    }
}

#[cfg(test)]
mod codegen_smoke_tests {
    use super::*;
    use crate::model::onnx_proto::{TensorProto, ValueInfoProto};

    fn graph(nodes: Vec<NodeProto>, inputs: &[&str]) -> GraphProto {
        let last = nodes.last().unwrap().output[0].clone();
        GraphProto {
            node: nodes,
            input: inputs
                .iter()
                .map(|name| ValueInfoProto {
                    name: name.to_string(),
                    ..Default::default()
                })
                .collect(),
            output: vec![ValueInfoProto {
                name: last,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn f32_weight(name: &str, dims: &[i64], raw: Vec<u8>) -> TensorProto {
        TensorProto {
            name: name.to_string(),
            data_type: 1,
            dims: dims.to_vec(),
            raw_data: raw,
            ..Default::default()
        }
    }

    fn relu_graph(weight_name: &str, weight: TensorProto) -> GraphProto {
        let mut g = graph(
            vec![NodeProto {
                op_type: "Relu".to_string(),
                input: vec![weight_name.to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            &[],
        );
        g.initializer.push(weight);
        g
    }

    /// Two models sharing a transformer (like MOSS's prefill and decode step)
    /// must store the identical weights once. `with_shared_weights` keeps their
    /// offsets and appends only what is new, so one file serves both.
    #[test]
    fn shared_weights_are_reused_across_models() {
        let bytes: Vec<u8> = (0..64u8).collect();

        let first = Compiler::new()
            .with_name("SharedA")
            .compile(&relu_graph("w", f32_weight("w", &[8, 8], bytes.clone())))
            .expect("first model should compile");

        // Same tensor under a different name, plus one genuinely new tensor
        // that the graph reads (unused initializers are not written).
        let mut second_graph = relu_graph("w2", f32_weight("w2", &[8, 8], bytes.clone()));
        second_graph
            .initializer
            .push(f32_weight("extra", &[8, 8], vec![9; 64]));
        second_graph.node.push(NodeProto {
            op_type: "Add".to_string(),
            input: vec!["y".to_string(), "extra".to_string()],
            output: vec!["z".to_string()],
            ..Default::default()
        });
        second_graph.output = vec![crate::model::onnx_proto::ValueInfoProto {
            name: "z".to_string(),
            ..Default::default()
        }];

        let second = Compiler::new()
            .with_name("SharedB")
            .with_shared_weights(SharedWeights::from_result(&first))
            .compile(&second_graph)
            .expect("second model should compile");

        assert!(
            second.weights.starts_with(&first.weights),
            "the shared blob must be reused as a prefix"
        );
        assert!(
            second.weights.len() > first.weights.len(),
            "the new tensor must be appended after the shared one"
        );
        for (hash, (offset, len)) in &first.content_hashes {
            assert_eq!(
                second.content_hashes.get(hash),
                Some(&(*offset, *len)),
                "identical content must keep its offset"
            );
        }
    }
}

#[cfg(test)]
mod conv_pattern_tests {
    use super::*;
    use crate::model::onnx_proto::{AttributeProto, TensorProto, ValueInfoProto};

    /// A depthwise 3x3 Conv with stride `[2, 1]` (as in PP-OCR's recognizer),
    /// followed by `next` (`Relu`, or `Add` of a bias initializer).
    fn graph(next: &str) -> GraphProto {
        let ints = |name: &str, v: &[i64]| AttributeProto {
            name: name.to_string(),
            ints: v.to_vec(),
            ..Default::default()
        };
        let mut group = ints("group", &[]);
        group.i = 4;
        let conv = NodeProto {
            op_type: "Conv".to_string(),
            input: vec!["x".to_string(), "w".to_string()],
            output: vec!["c".to_string()],
            attribute: vec![
                ints("kernel_shape", &[3, 3]),
                ints("strides", &[2, 1]),
                ints("dilations", &[1, 2]),
                ints("pads", &[1, 2, 1, 2]),
                group,
            ],
            ..Default::default()
        };
        let mut inputs = vec!["c".to_string()];
        if next == "Add" {
            inputs.push("b".to_string());
        }
        let after = NodeProto {
            op_type: next.to_string(),
            input: inputs,
            output: vec!["y".to_string()],
            ..Default::default()
        };
        let init = |name: &str, dims: &[i64]| TensorProto {
            name: name.to_string(),
            data_type: ir::DType::F32.to_onnx(),
            dims: dims.to_vec(),
            raw_data: vec![0u8; 4 * dims.iter().product::<i64>() as usize],
            ..Default::default()
        };
        GraphProto {
            node: vec![conv, after],
            initializer: vec![init("w", &[4, 1, 3, 3]), init("b", &[1, 4, 1, 1])],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// The fused Conv patterns used to keep only the first stride and
    /// dilation and repeat it, turning `[2, 1]` into `[2, 2]`.
    #[test]
    fn fused_conv_keeps_per_axis_strides_and_dilations() {
        for next in ["Relu", "Add"] {
            let result = Compiler::new()
                .with_name("ConvStrides")
                .with_default_optimizations()
                .compile(&graph(next))
                .expect("compilation should succeed");
            let call = result
                .code
                .lines()
                .find(|l| l.contains("lele::kernels::conv2d"))
                .unwrap_or_else(|| panic!("{next}: no conv2d call in\n{}", result.code));
            assert!(
                call.contains("&[1, 2], 4, &[1, 2, 1, 2], &[2, 1]"),
                "{next}: {call}"
            );
            // Both went through their fused pattern.
            let fused = if next == "Relu" {
                "conv2d_fused("
            } else {
                "Some(&self.weight_f32"
            };
            assert!(call.contains(fused), "{next}: {call}");
        }
    }
}
