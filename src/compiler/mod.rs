use crate::model::onnx_proto::{GraphProto, NodeProto, TensorProto};
use crate::model::tensor_to_array;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
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
pub type PatternGenerator = Box<
    dyn Fn(
        &[&NodeProto],
        &HashMap<String, (usize, usize, Vec<usize>)>,
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
}
pub struct Allocator {
    // Maps tensor name -> Buffer Index
    pub tensor_to_buffer: HashMap<String, usize>,
    pub num_buffers: usize,
}
// Helper to identify nodes uniquely across recursion
struct NodeVisitor<'a> {
    id_counter: usize,
    // Map tensor name -> List of Node IDs that use it
    uses: HashMap<String, Vec<usize>>,
    // Map Node ID -> NodeProto
    node_map: HashMap<usize, &'a NodeProto>,
    // Map Node ID -> outputs
    node_outputs: HashMap<usize, Vec<String>>,
    // Map Node ID -> inputs
    node_inputs: HashMap<usize, Vec<String>>,
    // Map alias -> sources (e.g. If_out -> [Then_out, Else_out])
    aliases: HashMap<String, Vec<String>>,
}
impl<'a> NodeVisitor<'a> {
    fn new() -> Self {
        Self {
            id_counter: 0,
            uses: HashMap::new(),
            node_map: HashMap::new(),
            node_outputs: HashMap::new(),
            node_inputs: HashMap::new(),
            aliases: HashMap::new(),
        }
    }
    fn visit(&mut self, graph: &'a GraphProto) {
        for node in &graph.node {
            let id = self.id_counter;
            self.id_counter += 1;
            self.node_map.insert(id, node);
            self.node_outputs.insert(id, node.output.clone());
            self.node_inputs.insert(id, node.input.clone());
            for input in &node.input {
                self.uses.entry(input.clone()).or_default().push(id);
            }
            // Recurse for subgraphs (If, Loop, Scan) in DETERMINISTIC ORDER matching generation
            if node.op_type == "If" {
                let then_g = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "then_branch")
                    .and_then(|a| a.g.as_ref());
                let else_g = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "else_branch")
                    .and_then(|a| a.g.as_ref());
                // Record aliases
                for (i, out_name) in node.output.iter().enumerate() {
                    let mut sources = Vec::new();
                    if let Some(g) = then_g {
                        if i < g.output.len() {
                            sources.push(g.output[i].name.clone());
                        }
                    }
                    if let Some(g) = else_g {
                        if i < g.output.len() {
                            sources.push(g.output[i].name.clone());
                        }
                    }
                    if !sources.is_empty() {
                        self.aliases.insert(out_name.clone(), sources);
                    }
                }
                if let Some(g) = then_g {
                    self.visit(g);
                }
                if let Some(g) = else_g {
                    self.visit(g);
                }
            } else if node.op_type == "Loop" {
                let body = node
                    .attribute
                    .iter()
                    .find(|a| a.name == "body")
                    .and_then(|a| a.g.as_ref());
                // For Loop, output [i] corresponds to scan_output.
                // The main outputs are N iteration outputs + K scan outputs?
                // For simplicity, just aliasing implicit body outputs is harder because naming convention relies on scan_outputs.
                // Silero VAD mostly uses If, let's stick to If for now. Code generation for Loop is complex.
                if let Some(g) = body {
                    self.visit(g);
                }
            } else {
                // Record Zero-Copy Aliases
                if matches!(
                    node.op_type.as_str(),
                    "Squeeze" | "Unsqueeze" | "Reshape" | "Identity" | "Flatten" | "Cast"
                ) && !node.input.is_empty()
                    && !node.output.is_empty()
                {
                    self.aliases
                        .insert(node.output[0].clone(), vec![node.input[0].clone()]);
                }
                for attr in &node.attribute {
                    if let Some(g) = &attr.g {
                        self.visit(g);
                    }
                }
            }
        }
    }
}
pub(crate) struct AnalysisData {
    pub death_time: HashMap<String, usize>,
    pub uses: HashMap<String, Vec<usize>>,
}
fn solve_allocation(graph: &GraphProto, outputs: &[String]) -> (Allocator, AnalysisData) {
    let mut visitor = NodeVisitor::new();
    visitor.visit(graph);
    // Add graph outputs as "uses" at infinity (max_id)
    let max_id = visitor.id_counter;
    for out in outputs {
        visitor.uses.entry(out.clone()).or_default().push(max_id);
    }
    // Liveness Analysis
    let mut death_time = HashMap::new();
    for (name, uses) in &visitor.uses {
        if let Some(max) = uses.iter().max() {
            death_time.insert(name.clone(), *max);
        }
    }
    // Propagate aliases backwards
    // If output X aliases Y, then Y must live as long as X.
    let mut changed = true;
    while changed {
        changed = false;
        for (alias, sources) in &visitor.aliases {
            let alias_death = *death_time.get(alias).unwrap_or(&0);
            for source in sources {
                let source_death = *death_time.get(source).unwrap_or(&0);
                if alias_death > source_death {
                    death_time.insert(source.clone(), alias_death);
                    changed = true;
                }
            }
        }
    }
    let mut free_buffers: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    let mut active_allocations: HashMap<String, usize> = HashMap::new();
    let mut all_allocations: HashMap<String, usize> = HashMap::new();
    let mut max_buffers = 0;
    for id in 0..visitor.id_counter {
        // Release buffers
        let mut to_remove = Vec::new();
        for (tensor, &buf_idx) in &active_allocations {
            let die_at = *death_time.get(tensor).unwrap_or(&usize::MAX);
            if die_at < id {
                to_remove.push((tensor.clone(), buf_idx));
            }
        }
        for (t, b) in to_remove {
            active_allocations.remove(&t);
            free_buffers.push(Reverse(b));
        }
        // Allocate output buffers
        if let Some(node_outputs) = visitor.node_outputs.get(&id) {
            for out in node_outputs {
                if out.is_empty() {
                    continue;
                }
                // Check if this output aliases a source
                let mut alias_idx = None;
                if let Some(sources) = visitor.aliases.get(out) {
                    // Try to find the buffer of the source
                    // Since we propagated liveness, the source buffer should be active
                    // For single source aliases (Identity etc)
                    if let Some(src) = sources.first() {
                        if let Some(&idx) = active_allocations.get(src) {
                            alias_idx = Some(idx);
                        } else if let Some(&_idx) = all_allocations.get(src) {
                            // Source might be dead but consistent?
                            // But we need it to be active to reserve it.
                            // If it's not active, maybe it was freed?
                            // But we propagated lifetime, so it SHOULD be active.
                            // Unless it's a constant or weight which has no buffer?
                            // Weights used to be handled separately.
                            // If source is not in active/all allocations, it might be an input or weight.
                            // In that case, we can't alias into a workspace buffer number??
                            // Wait, weights don't use workspace.
                            // Input tensors don't use workspace logic here?
                            // If source is weight, Alias is view of weight.
                            // The generator will handle view of weight.
                            // We just need to ensure we don't allocate a NEW buffer for it.
                        }
                    }
                }
                if let Some(idx) = alias_idx {
                    // It's an alias to an existing buffer.
                    // Do NOT mark as active allocation (owner responsible for freeing)
                    all_allocations.insert(out.clone(), idx);
                } else {
                    // Allocate new
                    // Check inputs to avoid reusing their buffers for the same node (Borrow Checker)
                    let mut input_bufs = std::collections::HashSet::new();
                    if let Some(inps) = visitor.node_inputs.get(&id) {
                        input_bufs
                            .extend(inps.iter().filter_map(|n| all_allocations.get(n)).cloned());
                    }
                    // Also check inputs of the previous node to avoid immediate reuse in fused ops (e.g. Conv+Relu)
                    if id > 0 {
                        if let Some(inps) = visitor.node_inputs.get(&(id - 1)) {
                            input_bufs.extend(
                                inps.iter().filter_map(|n| all_allocations.get(n)).cloned(),
                            );
                        }
                    }
                    let mut deferred = Vec::new();
                    let mut selected_buf = None;
                    while let Some(Reverse(b)) = free_buffers.pop() {
                        if input_bufs.contains(&b) {
                            deferred.push(b);
                        } else {
                            selected_buf = Some(b);
                            break;
                        }
                    }
                    for b in deferred {
                        free_buffers.push(Reverse(b));
                    }
                    let buf_idx = if let Some(b) = selected_buf {
                        b
                    } else {
                        let b = max_buffers;
                        max_buffers += 1;
                        b
                    };
                    active_allocations.insert(out.clone(), buf_idx);
                    all_allocations.insert(out.clone(), buf_idx);
                }
            }
        }
    }
    (
        Allocator {
            tensor_to_buffer: all_allocations,
            num_buffers: max_buffers,
        },
        AnalysisData {
            death_time,
            uses: visitor.uses,
        },
    )
}
pub mod generate;
pub mod patterns;
pub mod ops;

pub(crate) use generate::*;

pub struct Compiler {
    pub(crate) overrides: HashMap<String, OpHandler>,
    pub(crate) model_name: String,
    pub(crate) patterns: Vec<Pattern>,
    pub(crate) custom_methods: Vec<String>,
    pub constant_folding: bool,
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
            constant_folding: false,
        }
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
                &HashMap<String, (usize, usize, Vec<usize>)>,
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

    pub fn with_default_optimizations(mut self) -> Self {
        self = self.with_custom_method(include_str!("snippets/default_methods.rs"));
        for p in patterns::get_default_patterns() {
            self.patterns.push(p);
        }
        self
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.model_name = name.to_string();
        self
    }

    pub fn with_constant_folding(mut self, enabled: bool) -> Self {
        self.constant_folding = enabled;
        self
    }

    fn fold_constants(&self, graph: &mut GraphProto) {
        if !self.constant_folding {
            return;
        }
        let mut constants: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        // 1. Load initializers
        for init in &graph.initializer {
            if let Ok((data, shape)) = tensor_to_array(init) {
                if !data.is_empty() {
                    constants.insert(init.name.clone(), (data, shape));
                }
            }
        }
        // 2. Load existing Constant nodes
        for node in &graph.node {
            if node.op_type == "Constant" {
                if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
                    if let Some(t) = &attr.t {
                        if let Ok((data, shape)) = tensor_to_array(t) {
                            if !data.is_empty() {
                                constants.insert(node.output[0].clone(), (data, shape));
                            }
                        }
                    }
                }
            }
        }
        let mut folded_indices = std::collections::HashSet::new();
        let mut new_initializers = Vec::new();
        // 3. Simple single-pass folding (can be iterative but single-pass handles most ONNX patterns which are DAG)
        for (i, node) in graph.node.iter().enumerate() {
            let mut all_inputs_const = true;
            if node.input.is_empty() && node.op_type != "Constant" {
                all_inputs_const = false;
            }
            for input in &node.input {
                if !constants.contains_key(input) {
                    all_inputs_const = false;
                    break;
                }
            }
            if !all_inputs_const || node.op_type == "Constant" {
                continue;
            }
            // Try folding common ops
            let result = match node.op_type.as_str() {
                "Shape" => {
                    let (_, shape) = &constants[&node.input[0]];
                    let data: Vec<f32> = shape.iter().map(|&s| s as f32).collect();
                    let out_shape = vec![shape.len()];
                    Some((data, out_shape))
                }
                "Unsqueeze" => {
                    let axes = if node.input.len() > 1 {
                        constants
                            .get(&node.input[1])
                            .map(|(d, _)| d.iter().map(|&x| x as i64).collect::<Vec<_>>())
                    } else {
                        node.attribute
                            .iter()
                            .find(|a| a.name == "axes")
                            .map(|a| a.ints.clone())
                    };
                    if let Some(axes) = axes {
                        let (data, shape) = &constants[&node.input[0]];
                        let mut new_shape = shape.clone();
                        let mut sorted_axes = axes.clone();
                        sorted_axes.sort();
                        for &ax in &sorted_axes {
                            let idx = if ax < 0 {
                                (new_shape.len() + 1) as i64 + ax
                            } else {
                                ax
                            } as usize;
                            if idx <= new_shape.len() {
                                new_shape.insert(idx, 1);
                            }
                        }
                        Some((data.clone(), new_shape))
                    } else {
                        None
                    }
                }
                "Squeeze" => {
                    let axes = if node.input.len() > 1 {
                        constants
                            .get(&node.input[1])
                            .map(|(d, _)| d.iter().map(|&x| x as i64).collect::<Vec<_>>())
                    } else {
                        node.attribute
                            .iter()
                            .find(|a| a.name == "axes")
                            .map(|a| a.ints.clone())
                    };
                    let (data, shape) = &constants[&node.input[0]];
                    let mut new_shape = Vec::new();
                    if let Some(axes) = axes {
                        let axes_set: std::collections::HashSet<usize> = axes
                            .iter()
                            .map(|&a| {
                                if a < 0 {
                                    (shape.len() as i64 + a) as usize
                                } else {
                                    a as usize
                                }
                            })
                            .collect();
                        for (idx, &d) in shape.iter().enumerate() {
                            if d != 1 || !axes_set.contains(&idx) {
                                new_shape.push(d);
                            }
                        }
                    } else {
                        for &d in shape.iter() {
                            if d != 1 {
                                new_shape.push(d);
                            }
                        }
                    }
                    Some((data.clone(), new_shape))
                }
                "Concat" => {
                    let axis_attr = node
                        .attribute
                        .iter()
                        .find(|a| a.name == "axis")
                        .map(|a| a.i)
                        .unwrap_or(0);
                    let mut all_data = Vec::new();
                    let mut first_shape = None;
                    let mut total_axis_dim = 0;
                    for input in &node.input {
                        let (data, shape) = &constants[input];
                        let axis = if axis_attr < 0 {
                            (shape.len() as i64 + axis_attr) as usize
                        } else {
                            axis_attr as usize
                        };
                        if first_shape.is_none() {
                            first_shape = Some(shape.clone());
                        }
                        total_axis_dim += shape[axis];
                        all_data.push((data, shape, axis));
                    }
                    if let Some(mut target_shape) = first_shape {
                        let axis = all_data[0].2;
                        target_shape[axis] = total_axis_dim;
                        // Simple 1D/2D concat for shapes (usually Concat is used for shape tensors)
                        if target_shape.len() == 1 {
                            let mut merged = Vec::new();
                            for (d, _, _) in all_data {
                                merged.extend_from_slice(d);
                            }
                            Some((merged, target_shape))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "Cast" => {
                    let (data, shape) = &constants[&node.input[0]];
                    Some((data.clone(), shape.clone()))
                }
                "ConstantOfShape" => {
                    let (data, _) = &constants[&node.input[0]];
                    let val = node
                        .attribute
                        .iter()
                        .find(|a| a.name == "value")
                        .and_then(|a| a.t.as_ref())
                        .and_then(|t| {
                            if !t.float_data.is_empty() {
                                Some(t.float_data[0])
                            } else if !t.double_data.is_empty() {
                                Some(t.double_data[0] as f32)
                            } else if !t.int32_data.is_empty() {
                                Some(t.int32_data[0] as f32)
                            } else if !t.int64_data.is_empty() {
                                Some(t.int64_data[0] as f32)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0.0);
                    let shape: Vec<usize> = data.iter().map(|&x| x as usize).collect();
                    let size: usize = shape.iter().product();
                    let res_data = vec![val; size];
                    Some((res_data, shape))
                }
                _ => None,
            };
            if let Some((data, shape)) = result {
                let out_name = &node.output[0];
                constants.insert(out_name.clone(), (data.clone(), shape.clone()));
                let new_init = TensorProto {
                    name: out_name.clone(),
                    dims: shape.iter().map(|&s| s as i64).collect(),
                    data_type: 1, // FLOAT
                    float_data: data,
                    ..Default::default()
                };
                new_initializers.push(new_init);
                folded_indices.insert(i);
            }
        }
        // Apply
        if !folded_indices.is_empty() {
            graph.initializer.extend(new_initializers);
            let mut idx = 0;
            graph.node.retain(|_| {
                let keep = !folded_indices.contains(&idx);
                idx += 1;
                keep
            });
        }
    }

    pub fn compile(
        &self,
        graph: &GraphProto,
    ) -> Result<CompilationResult, Box<dyn std::error::Error>> {
        let mut graph_storage;
        let graph = if self.constant_folding {
            graph_storage = graph.clone();
            self.fold_constants(&mut graph_storage);
            &graph_storage
        } else {
            graph
        };
        let mut weights_data = Vec::new();
        let mut offset_map = Vec::new();
        let mut current_offset = 0usize;
        let mut int64_map = HashMap::new();
        // 1. Collect Weights
        collect_weights(
            graph,
            &mut weights_data,
            &mut offset_map,
            &mut current_offset,
            &mut int64_map,
        )?;
        // Prepare weight lookup map for body generation
        let mut weight_lookup = HashMap::new();
        for (name, offset, len, shape) in &offset_map {
            let safe_name = sanitize_name(name);
            weight_lookup
                .entry(safe_name)
                .or_insert((*offset, *len, shape.clone()));
        }
        // --- OPTIMIZATION: Reorder ConstantOfShape -> Concat pattern ---
        // We create a modified list of nodes to pass to body generation
        let mut nodes: Vec<&NodeProto> = graph.node.iter().collect();
        // Find Concat nodes that can be optimized
        let mut changes = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            if node.op_type == "Concat" && node.input.len() >= 2 {
                // Check inputs
                for input_name in &node.input {
                    // Check if this input comes from a ConstantOfShape
                    if let Some(pos) = nodes
                        .iter()
                        .position(|n| !n.output.is_empty() && n.output[0] == *input_name)
                    {
                        if pos < i && pos != i - 1 {
                            // If it's before and not immediately before
                            let producer = nodes[pos];
                            if producer.op_type == "ConstantOfShape" {
                                // Check if producer depends on weights (so it can be moved anywhere)
                                // Or constants?
                                // ConstantOfShape input[0] is shape. Check if it's a weight or initializer.
                                let shape_input = &producer.input[0];
                                // Check against known weights/initializers
                                // We have weight_lookup keyed by sanitized name, but let's check raw name in offset_map?
                                // Actually sanitize_name(shape_input) lookup works.
                                if weight_lookup.contains_key(&sanitize_name(shape_input))
                                    || int64_map.contains_key(shape_input)
                                {
                                    // Safe to move!
                                    changes.push((pos, i));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Apply moves.
        // To avoid index invalidation, we rebuild the list
        if !changes.is_empty() {
            let mut new_nodes = Vec::new();
            let mut moved_indices = std::collections::HashSet::new();
            for (from, _) in &changes {
                moved_indices.insert(*from);
            }
            // Iterate original list
            for (i, node) in nodes.iter().enumerate() {
                // Check if `i` is a target destination for some moved node
                // We want to insert 'from' node right BEFORE 'to' node.
                // Find all moves where 'to' == i
                for (from, to) in &changes {
                    if *to == i && moved_indices.contains(from) {
                        new_nodes.push(nodes[*from]);
                        // Remove from set to avoid dups if multiple targets? (Should not happen for unique unique producer)
                    }
                }
                if !moved_indices.contains(&i) {
                    new_nodes.push(node);
                }
            }
            nodes = new_nodes;
        }
        // --- END OPTIMIZATION ---
        // Run Allocation/Liveness FIRST
        let output_names: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();
        // Warn: Allocation uses graph.node internally!
        // We need to pass our reordered nodes to solve_allocation?
        // solve_allocation takes &GraphProto.
        // We can create a temporary GraphProto
        let mut temp_graph = graph.clone();
        temp_graph.node = nodes.iter().map(|&n| n.clone()).collect();
        let (allocator, analysis) = solve_allocation(&temp_graph, &output_names);
        // Generate Body FIRST to determine used buffers
        // We use generate_partitioned_graph instead of generate_graph_body to split the code
        let mut chunk_functions = Vec::new(); // Will hold the code for fn chunk_0(...) { ... }
        let mut body_buffer = Vec::new(); // Will hold the code for forward() body
        let mut current_id = 0;
        generate_partitioned_graph(
            &nodes,
            &mut chunk_functions,
            &mut body_buffer,
            2,
            &weight_lookup,
            &int64_map,
            Some(&allocator),
            Some(&analysis),
            &mut current_id,
            self,
            &graph.input,
            &graph.output,
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
            if end > start {
                if let Ok(buf_idx) = all_code[start..end].parse::<usize>() {
                    used_buffers.insert(buf_idx);
                }
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
        // Main Struct (Silero style: holds data reference)
        writeln!(&mut code, "\npub struct {}<'a> {{", struct_name)?;
        writeln!(&mut code, "    data: &'a [u8],")?;
        writeln!(&mut code, "    _phantom: std::marker::PhantomData<&'a ()>,")?;
        writeln!(&mut code, "}}")?;
        writeln!(&mut code, "\nimpl<'a> {}<'a> {{", struct_name)?;
        writeln!(&mut code, "    pub fn new(data: &'a [u8]) -> Self {{")?;
        writeln!(
            &mut code,
            "        Self {{ data, _phantom: std::marker::PhantomData }}"
        )?;
        writeln!(&mut code, "    }}")?;
        for method in &self.custom_methods {
            writeln!(&mut code, "{}", method)?;
        }
        // Generated Chunk Functions
        for func in &chunk_functions {
            writeln!(&mut code, "{}", func)?;
        }
        // Helper for weights (Hardcoded offset logic)
        writeln!(&mut code, "    // Inlined removed to reduce code bloat")?;
        writeln!(&mut code, "    fn weight(&self, offset: usize, len: usize, shape: &'a [usize]) -> TensorView<'a> {{")?;
        writeln!(
            &mut code,
            "        let slice = &self.data[offset..offset+len];"
        )?;
        writeln!(&mut code, "        let f32_slice = unsafe {{ std::slice::from_raw_parts(slice.as_ptr() as *const f32, slice.len() / 4) }};")?;
        writeln!(&mut code, "        TensorView::new(f32_slice, shape)")?;
        writeln!(&mut code, "    }}")?;
        // Inference Function
        writeln!(&mut code, "\n    // Inference Entry Point")?;
        let args: Vec<String> = graph
            .input
            .iter()
            .map(|i| format!("{}: TensorView<'a>", sanitize_name(&i.name)))
            .collect();
        // Generate return type
        let ret_types: Vec<String> = (0..graph.output.len())
            .map(|_| "TensorView<'a>".to_string())
            .collect();
        let ret_sig = if ret_types.len() == 1 {
            "TensorView<'a>".to_string()
        } else {
            format!("({})", ret_types.join(", "))
        };
        writeln!(
            &mut code,
            "    pub fn forward(&self, {}) -> {} {{",
            args.join(", "),
            ret_sig
        )?;
        writeln!(
            &mut code,
            "        let mut ws = {}Workspace::new();",
            struct_name
        )?;
        // Generate Body
        code.extend_from_slice(body_str.as_bytes());
        // Return outputs
        let ret_vals: Vec<String> = graph
            .output
            .iter()
            .map(|o| {
                let name = sanitize_name(&o.name);
                if let Some((offset, len, shape)) = weight_lookup.get(&name) {
                    format!("self.weight({}, {}, &{:?}).to_owned()", offset, len, shape)
                } else {
                    format!("{}.to_owned()", name)
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
            weights: weights_data,
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

fn collect_weights(
    graph: &GraphProto,
    bin_data: &mut Vec<u8>,
    offset_map: &mut Vec<(String, usize, usize, Vec<usize>)>,
    current_offset: &mut usize,
    _int64_map: &mut HashMap<String, (Vec<i64>, Vec<usize>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initializers
    for init in &graph.initializer {
        // Try reading as float array (converts everything to f32)
        match tensor_to_array(init) {
            Ok((data, shape)) => {
                if !data.is_empty() {
                    let slice = data.as_slice();
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
                    };
                    bin_data.write_all(bytes)?;
                    offset_map.push((init.name.clone(), *current_offset, bytes.len(), shape));
                    *current_offset += bytes.len();
                }
            }
            Err(_) => {
                // Ignore errors
            }
        }
    }
    // 2. Constants
    for node in &graph.node {
        if node.op_type == "Constant" {
            if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
                if let Some(t) = &attr.t {
                    // Convert everything to f32 weights to avoid inlining large/many constants code
                    if let Ok((data, shape)) = tensor_to_array(t) {
                        if !data.is_empty() {
                            let slice = data.as_slice();
                            let bytes: &[u8] = unsafe {
                                std::slice::from_raw_parts(
                                    slice.as_ptr() as *const u8,
                                    slice.len() * 4,
                                )
                            };
                            bin_data.write_all(bytes)?;
                            if let Some(out_name) = node.output.first() {
                                offset_map.push((
                                    out_name.clone(),
                                    *current_offset,
                                    bytes.len(),
                                    shape,
                                ));
                                *current_offset += bytes.len();
                            }
                        }
                    }
                }
            }
        }
        // 3. Recurse
        for attr in &node.attribute {
            if let Some(g) = &attr.g {
                collect_weights(g, bin_data, offset_map, current_offset, _int64_map)?;
            }
        }
    }
    Ok(())
}
