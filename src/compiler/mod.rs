pub use self::alloc::Allocator;
pub(crate) use self::alloc::AnalysisData;
use crate::model::onnx_proto::{GraphProto, NodeProto};
use std::collections::HashMap;
use std::io::Write;
/// Code generation for the nodes of an operator type, registered with
/// [`Compiler::with_override`]: writes the code computing the node
/// [`Codegen`] describes.
pub type OpHandler = Box<dyn Fn(&Codegen, &mut dyn Write) -> std::io::Result<()>>;
/// A rewrite of the graph, registered with
/// [`Compiler::with_pass_before_fusion`] or
/// [`Compiler::with_pass_after_fusion`].
pub type Pass = Box<dyn Fn(&mut ir::Graph)>;
pub struct CompilationResult {
    pub code: String,
    pub weights: Vec<u8>,
}
mod alloc;
pub mod generate;
pub mod ir;
pub mod ops;
mod passes;
mod pattern;
mod specialize;
mod weights;

pub use generate::Codegen;
pub use specialize::InputValue;

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
    /// generated code would run, and `If` nodes whose condition is known by
    /// the branch they take.
    pub constant_folding: bool,
    /// Infers tensor shapes and replaces shape computations (`Shape`,
    /// `Gather` of dimensions, ...) by constants where the dimensions are
    /// known, and `Reshape`/`Expand`/`ConstantOfShape` targets by
    /// dimensions of their inputs where they are not. With constant folding
    /// on, the results are folded further.
    pub shape_folding: bool,
    /// Removes nodes whose outputs nothing reads.
    pub dead_code_elimination: bool,
    /// Removes nodes that leave their input as it is (`Identity`, a `Cast`
    /// to the input's type, a product by ones, a `Pad` by zeros, ...),
    /// merges transposes of transposes, and makes reshapes of the transposes
    /// that only move axes of size 1.
    pub simplification: bool,
    /// Rewrites `MatMul` of a dequantized int8 weight into an integer
    /// matmul that keeps the weight in int8, and collapses
    /// `QuantizeLinear` → `DequantizeLinear` pairs into one node.
    pub qdq_fusion: bool,
    /// Fuses operator chains (LayerNorm, GELU, SiLU, Conv + bias/activation,
    /// MatMul + Add, dynamically quantized linear layers, ...) into single
    /// operators with fused kernels.
    pub operator_fusion: bool,
    /// Merges nodes that run the same operator with the same attributes on
    /// the same inputs, so that each such computation runs once.
    pub common_subexpression_elimination: bool,
    /// Stores f16 weights read only as the right operand of matrix products,
    /// or as the weights of pointwise 1-D convolutions (transposed), in the
    /// order the matmul kernel reads them
    /// ([`pack_f16`](crate::kernels::matmul::pack_f16)), so that each product
    /// streams its weight from start to end. Not done for the weights a
    /// product with an [override](Compiler::with_override) reads, which
    /// reads them as stored in the model.
    pub weight_packing: bool,
}

impl Optimizations {
    /// Every optimization.
    pub fn all() -> Self {
        Self {
            constant_folding: true,
            shape_folding: true,
            dead_code_elimination: true,
            simplification: true,
            qdq_fusion: true,
            operator_fusion: true,
            common_subexpression_elimination: true,
            weight_packing: true,
        }
    }
}

pub struct Compiler {
    pub(crate) overrides: HashMap<String, OpHandler>,
    pub(crate) model_name: String,
    pub(crate) passes_before_fusion: Vec<Pass>,
    pub(crate) passes_after_fusion: Vec<Pass>,
    pub(crate) custom_methods: Vec<String>,
    pub(crate) specialization: specialize::Specialization,
    pub optimizations: Optimizations,
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
            passes_before_fusion: Vec::new(),
            passes_after_fusion: Vec::new(),
            custom_methods: Vec::new(),
            specialization: Default::default(),
            optimizations: Optimizations::default(),
        }
    }
    /// Generates the nodes of operator type `op` with `handler` instead of
    /// lele's kernels.
    pub fn with_override<F>(mut self, op: &str, handler: F) -> Self
    where
        F: Fn(&Codegen, &mut dyn Write) -> std::io::Result<()> + 'static,
    {
        self.overrides.insert(op.to_string(), Box::new(handler));
        self
    }

    /// Runs `pass` on the graph before lele fuses operators, after folding
    /// and simplification; passes run in the order they are registered.
    /// With [`Optimizations::operator_fusion`] off, `pass` runs all the same.
    pub fn with_pass_before_fusion<F>(mut self, pass: F) -> Self
    where
        F: Fn(&mut ir::Graph) + 'static,
    {
        self.passes_before_fusion.push(Box::new(pass));
        self
    }

    /// Runs `pass` on the graph after lele fuses operators, before common
    /// subexpressions are merged and weights packed; passes run in the order
    /// they are registered.
    pub fn with_pass_after_fusion<F>(mut self, pass: F) -> Self
    where
        F: Fn(&mut ir::Graph) + 'static,
    {
        self.passes_after_fusion.push(Box::new(pass));
        self
    }

    /// Replaces each run of nodes `matcher` finds by one node of operator
    /// type `name`, which `generator` generates
    /// ([`with_override`](Self::with_override)). The matching runs as a pass
    /// before fusion ([`with_pass_before_fusion`](Self::with_pass_before_fusion)).
    ///
    /// `matcher` is called with every node, in execution order, and returns
    /// the nodes of the run that node is part of, if any. The node replacing
    /// them runs where the last of them ran. It reads what they read of other
    /// nodes, the graph inputs and weights, and produces what of theirs is
    /// read by other nodes or is a graph output, both in the order the run
    /// reads or produces them first. A run is left as it is if it holds a
    /// node with a subgraph, or if a node outside it reads one of its
    /// results before its last node runs.
    pub fn with_pattern<M, G>(self, name: &str, matcher: M, generator: G) -> Self
    where
        M: Fn(&ir::Graph, ir::NodeId) -> Option<Vec<ir::NodeId>> + 'static,
        G: Fn(&Codegen, &mut dyn Write) -> std::io::Result<()> + 'static,
    {
        let op_type = name.to_string();
        self.with_pass_before_fusion(move |g| pattern::fuse_matches(g, &op_type, &matcher))
            .with_override(name, generator)
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

    /// Compiles the model for input `name` of shape `shape` alone. The passes
    /// know its dimensions, shape folding resolves what depends on them, and
    /// buffers whose sizes become known are reserved by the workspace up
    /// front. The generated `forward` panics on another shape.
    pub fn with_input_shape(mut self, name: &str, shape: &[usize]) -> Self {
        self.specialization
            .shapes
            .insert(name.to_string(), shape.to_vec());
        self
    }

    /// Compiles the model for the symbolic dimension `name` (an input's
    /// `dim_param`, such as `batch_size`) being `value`, in every input and
    /// tensor that has it. The generated `forward` panics on inputs where it
    /// is not.
    pub fn with_dim(mut self, name: &str, value: usize) -> Self {
        self.specialization.dims.insert(name.to_string(), value);
        self
    }

    /// Compiles input `name` in as a constant holding `value`, its elements
    /// in order, of the input's type: what depends on it alone is folded.
    /// Its shape must be fixed, by the model or by
    /// [`with_input_shape`](Self::with_input_shape). The generated `forward`
    /// still takes the input, and panics on another value.
    pub fn with_input_value(mut self, name: &str, value: impl Into<InputValue>) -> Self {
        self.specialization
            .values
            .insert(name.to_string(), value.into());
        self
    }

    pub fn compile(
        &self,
        graph: &GraphProto,
    ) -> Result<CompilationResult, Box<dyn std::error::Error>> {
        let opt = self.optimizations;
        let mut ir = ir::Graph::from_onnx(graph.clone());
        let inputs = self.specialization.apply(&mut ir)?;
        if opt.qdq_fusion {
            // Must precede folding, which would otherwise consume the int8
            // weights.
            passes::fuse_qdq_matmul(&mut ir);
            passes::fuse_weight_only_matmul(&mut ir);
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
                // Folding resolves shape computations inference left open
                // (an `Expand` target selected by `Where`), whose results a
                // second round of inference then knows the shapes of.
                passes::infer_dtypes(&mut ir);
                passes::fold_shapes(&mut ir);
                passes::fold_constants(&mut ir);
            }
        }
        // Each round takes out at least one `If`, whose branch may fold into
        // the condition of another.
        while opt.constant_folding && passes::inline_constant_ifs(&mut ir) {
            passes::fold_constants(&mut ir);
            if opt.shape_folding {
                passes::infer_dtypes(&mut ir);
                passes::fold_shapes(&mut ir);
                passes::fold_constants(&mut ir);
            }
        }
        if opt.simplification {
            // A cast is removed where its input has the type it casts to.
            passes::infer_dtypes(&mut ir);
            passes::simplify(&mut ir);
        }
        if opt.dead_code_elimination {
            passes::eliminate_dead_code(&mut ir);
        }
        if opt.qdq_fusion {
            // Runs after folding so that only the activation pairs are left.
            passes::fuse_qdq(&mut ir);
            passes::fold_qdq_into_matmuls(&mut ir);
        }
        if !self.passes_before_fusion.is_empty() {
            passes::infer_dtypes(&mut ir);
        }
        for pass in &self.passes_before_fusion {
            pass(&mut ir);
        }
        if opt.operator_fusion {
            // Some fusions apply to f32 alone.
            passes::infer_dtypes(&mut ir);
            passes::fuse(&mut ir);
            if opt.dead_code_elimination {
                // The constants fused chains read (an epsilon, a scale) are
                // left without readers, and would each keep a buffer.
                passes::eliminate_dead_code(&mut ir);
            }
        }
        if !self.passes_after_fusion.is_empty() {
            passes::infer_dtypes(&mut ir);
        }
        for pass in &self.passes_after_fusion {
            pass(&mut ir);
        }
        if opt.common_subexpression_elimination {
            // Runs after fusion, whose patterns may need a value to have a
            // single reader.
            passes::eliminate_common_subexpressions(&mut ir);
        }
        passes::infer_dtypes(&mut ir);
        let var_types = variable_types(&ir);
        // Casts between types held alike are left to the element types to
        // tell apart.
        passes::bypass_view_casts(&mut ir, &|name| var_types.get(&sanitize_name(name)));
        if opt.operator_fusion {
            // Norms the casts kept apart are now adjacent.
            passes::fuse_scale_norm_pairs(&mut ir);
        }
        let var_types = variable_types(&ir);
        let (allocator, analysis) =
            alloc::solve_allocation(&ir, &|name| var_types.get(&sanitize_name(name)));
        if opt.weight_packing {
            // Overrides read weights as stored in the model.
            passes::pack_weights(&mut ir, &|node| {
                self.overrides.contains_key(node.op.op_type())
            });
        }
        let weights = weights::Weights::collect(&ir);
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
        for (i, ty) in allocator.buffer_types.iter().enumerate() {
            if used_buffers.contains(&i) {
                writeln!(&mut code, "    pub buf_{}: Vec<{}>,", i, ty)?;
            }
        }
        writeln!(&mut code, "}}")?;
        // Buffers whose tensors all have fixed shapes are allocated once, at
        // their largest size, when the workspace is made.
        let mut used: Vec<usize> = used_buffers.iter().copied().collect();
        used.sort();
        let sized: Vec<(usize, usize)> = used
            .iter()
            .filter_map(|&b| Some((b, allocator.buffer_lens[b]?)))
            .collect();
        writeln!(&mut code, "impl {}Workspace {{", struct_name)?;
        if sized.len() == used.len() {
            let bytes: usize = sized
                .iter()
                .map(|&(b, len)| len * allocator.buffer_types[b].size())
                .sum();
            writeln!(
                &mut code,
                "    /// Bytes of the buffers `new` allocates. Every tensor's shape is fixed, so the buffers never grow."
            )?;
            writeln!(&mut code, "    pub const BYTES: usize = {bytes};")?;
        }
        writeln!(&mut code, "    pub fn new() -> Self {{")?;
        if sized.is_empty() {
            writeln!(&mut code, "        Self::default()")?;
        } else {
            writeln!(&mut code, "        Self {{")?;
            for (b, len) in &sized {
                writeln!(&mut code, "            buf_{b}: Vec::with_capacity({len}),")?;
            }
            if sized.len() < used.len() {
                writeln!(&mut code, "            ..Default::default()")?;
            }
            writeln!(&mut code, "        }}")?;
        }
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
        // f16, u8 and i8 weights read as f32, and i32 and bool ones read as
        // i64, are widened once, here, not on every read; those the kernels
        // take as they are stored are left so.
        let mut weights_expr = "lele::runtime::Weights::new(data)".to_string();
        let read_code = [all_code.as_str()]
            .into_iter()
            .chain(self.custom_methods.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        use ir::DType;
        for (dtype, ty) in [
            (DType::F16, "f16"),
            (DType::U8, "u8"),
            (DType::I8, "i8"),
            (DType::I32, "i32_i64"),
            (DType::Bool, "bool_i64"),
        ] {
            let read = weights.read_through(dtype, &format!("weight_{ty}"), &read_code);
            if !read.is_empty() {
                weights_expr += &format!(".with_{ty}(&{read:?})");
            }
        }
        // The weights of the integer products are packed here, numbered as the
        // code reads them.
        let prepared = weights.prepared.borrow();
        if prepared.qweights.is_empty() && prepared.conv_weights.is_empty() {
            writeln!(&mut code, "        Self {{ weights: {weights_expr} }}")?;
        } else {
            writeln!(&mut code, "        let weights = {weights_expr};")?;
            let mut with = String::new();
            for (name, packed) in [
                ("qweights", &prepared.qweights),
                ("conv_weights", &prepared.conv_weights),
            ] {
                if packed.is_empty() {
                    continue;
                }
                writeln!(&mut code, "        let {name} = vec![")?;
                for expr in packed {
                    writeln!(&mut code, "            {expr},")?;
                }
                writeln!(&mut code, "        ];")?;
                with += &format!(".with_{name}({name})");
            }
            writeln!(&mut code, "        Self {{ weights: weights{with} }}")?;
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
        // `forward` takes every input, those compiled in included.
        let input_type = |i: &specialize::Input| {
            i.fixed_type
                .unwrap_or_else(|| var_types.get(&sanitize_name(&i.name)))
        };
        let args: Vec<String> = inputs
            .iter()
            .map(|i| {
                let name = sanitize_name(&i.name);
                match input_type(i) {
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
        let arg_names_str = inputs
            .iter()
            .map(|i| sanitize_name(&i.name))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            &mut code,
            "        let res = self.forward_with_workspace(&mut ws, {});",
            arg_names_str
        )?;
        // Outputs owning their data, such as states updated in place, move out.
        let ret_vals_owned: Vec<String> = if graph_outputs.len() == 1 {
            vec!["res.into_owned()".to_string()]
        } else {
            (0..graph_outputs.len())
                .map(|i| format!("res.{}.into_owned()", i))
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
        for input in &inputs {
            let var = sanitize_name(&input.name);
            for check in input.checks(struct_name, &var, input_type(input)) {
                writeln!(&mut code, "        {check}")?;
            }
        }
        // Generate Body
        code.extend_from_slice(body_str.as_bytes());
        // Return outputs
        let ret_vals: Vec<String> = graph_outputs
            .iter()
            .map(|o| {
                let name = sanitize_name(&o.name);
                if let Some(weight) = weights.get(&o.name) {
                    weight.expr(var_types.get(&name))
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
        })
    }
}
pub fn sanitize_name(name: &str) -> String {
    let s = name
        .replace(".", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "_");
    // Tensors are named by what produced them: `where`, `type` and `match` do occur.
    const KEYWORDS: &[&str] = &[
        "Self", "abstract", "as", "async", "await", "become", "box", "break", "const", "continue",
        "crate", "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if",
        "impl", "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv",
        "pub", "ref", "return", "self", "static", "struct", "super", "trait", "true", "try",
        "type", "typeof", "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
    ];
    if s.chars().next().is_some_and(|c| c.is_numeric()) {
        format!("_{}", s)
    } else if KEYWORDS.contains(&s.as_str()) {
        format!("{}_", s)
    } else {
        s
    }
}

#[cfg(test)]
mod sanitize_tests {
    use super::sanitize_name;

    #[test]
    fn names_are_identifiers() {
        assert_eq!(sanitize_name("/a/b.c-d:e"), "_a_b_c_d_e");
        assert_eq!(sanitize_name("1x"), "_1x");
        assert_eq!(sanitize_name("where"), "where_");
        assert_eq!(sanitize_name("type"), "type_");
        assert_eq!(sanitize_name("wherever"), "wherever");
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
