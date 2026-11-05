//! Typed graph IR that optimization passes run on.
//!
//! An ONNX [`GraphProto`] refers to tensors by name and leaves every
//! operator's attributes as an untyped list. [`Graph`] interns the names into
//! [`Value`]s that know their producer and consumers, and parses each node's
//! operator into an [`Op`]. Operators without a typed variant yet parse to
//! [`Op::Onnx`], so any model round-trips and variants can be added one at a
//! time.
//!
//! Every [`Node`] keeps the [`NodeProto`] it was built from. Lowering
//! ([`Graph::into_onnx`]) writes the node's current operator, attributes and
//! operands back into that proto, so code generation, op overrides and
//! patterns, which all work on `NodeProto`s, see the rewritten graph without
//! changes.

mod op;

pub use op::{
    Activation, Binary, ConvAttrs, DType, Op, PoolAttrs, QuantAttrs, Reduction, TargetDim, Unary,
};

use crate::model::onnx_proto::{
    GraphProto, NodeProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto,
    tensor_shape_proto, type_proto,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(u32);

/// One dimension of a tensor shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Fixed(i64),
    Sym(String),
    Unknown,
}

/// A tensor: a graph input, an initializer, a node output, or a value a
/// subgraph captures from an enclosing graph.
#[derive(Clone, Debug)]
pub struct Value {
    pub name: String,
    pub dtype: DType,
    /// `None` when the rank is unknown.
    pub shape: Option<Vec<Dim>>,
    /// Contents of an initializer.
    pub data: Option<TensorProto>,
    producer: Option<NodeId>,
    /// One entry per use, so a node that reads the value twice is listed
    /// twice. A node whose subgraph reads the value is listed once.
    consumers: Vec<NodeId>,
}

impl Value {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            dtype: DType::Undefined,
            shape: None,
            data: None,
            producer: None,
            consumers: Vec::new(),
        }
    }

    pub fn producer(&self) -> Option<NodeId> {
        self.producer
    }

    pub fn consumers(&self) -> &[NodeId] {
        &self.consumers
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    /// The node this one was built from. Lowering overwrites its op type,
    /// operands and the attributes `op` models; everything else, such as the
    /// node name and unmodelled attributes, is kept as is.
    pub proto: NodeProto,
    /// `None` is an omitted optional operand (`""` in ONNX).
    inputs: Vec<Option<ValueId>>,
    outputs: Vec<Option<ValueId>>,
    /// Values of enclosing scopes that the node's subgraphs read.
    captures: Vec<ValueId>,
}

impl Node {
    pub fn inputs(&self) -> &[Option<ValueId>] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Option<ValueId>] {
        &self.outputs
    }

    pub fn input(&self, i: usize) -> Option<ValueId> {
        self.inputs.get(i).copied().flatten()
    }

    pub fn output(&self, i: usize) -> Option<ValueId> {
        self.outputs.get(i).copied().flatten()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    /// Indexed by [`NodeId`]. Removed nodes leave `None` so ids stay valid.
    nodes: Vec<Option<Node>>,
    /// Execution order. May still list removed nodes, which are skipped.
    order: Vec<NodeId>,
    values: Vec<Value>,
    by_name: HashMap<String, ValueId>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    /// Initializers in model order, which fixes the weights file layout.
    initializers: Vec<ValueId>,
    value_info: Vec<ValueId>,
    /// Graph name, doc string and other fields the IR does not model.
    proto: GraphProto,
}

impl Graph {
    pub fn from_onnx(mut proto: GraphProto) -> Self {
        let mut g = Graph::default();
        for info in std::mem::take(&mut proto.input) {
            let v = g.intern(&info.name);
            g.set_type(v, &info);
            g.inputs.push(v);
        }
        for tensor in std::mem::take(&mut proto.initializer) {
            let v = g.intern(&tensor.name);
            let value = &mut g.values[v.0 as usize];
            if value.dtype == DType::Undefined {
                value.dtype = DType::from_onnx(tensor.data_type);
            }
            if value.shape.is_none() {
                value.shape = Some(tensor.dims.iter().map(|&d| Dim::Fixed(d)).collect());
            }
            value.data = Some(tensor);
            g.initializers.push(v);
        }
        for node in std::mem::take(&mut proto.node) {
            g.push_node(node);
        }
        for info in std::mem::take(&mut proto.output) {
            let v = g.intern(&info.name);
            g.set_type(v, &info);
            g.outputs.push(v);
        }
        for info in std::mem::take(&mut proto.value_info) {
            let v = g.intern(&info.name);
            g.set_type(v, &info);
            g.value_info.push(v);
        }
        g.proto = proto;
        g
    }

    pub fn into_onnx(self) -> GraphProto {
        let info = |v: &ValueId| self.value_info_proto(*v);
        let input = self.inputs.iter().map(info).collect();
        let output = self.outputs.iter().map(info).collect();
        // Skip values a pass has disconnected.
        let value_info = self
            .value_info
            .iter()
            .filter(|&&v| {
                let value = self.value(v);
                value.producer.is_some() || !value.consumers.is_empty()
            })
            .map(info)
            .collect();
        let mut proto = GraphProto {
            input,
            output,
            value_info,
            ..self.proto
        };
        let mut values = self.values;
        proto.initializer = self
            .initializers
            .iter()
            .filter_map(|v| {
                let value = &mut values[v.0 as usize];
                let mut tensor = value.data.take()?;
                tensor.name = value.name.clone();
                Some(tensor)
            })
            .collect();
        let mut nodes = self.nodes;
        proto.node = self
            .order
            .iter()
            .filter_map(|id| nodes[id.0 as usize].take())
            .map(|node| lower_node(node, &values))
            .collect();
        proto
    }

    /// `node` as the [`NodeProto`] [`Self::into_onnx`] would produce, its
    /// subgraphs lowered too.
    pub fn node_proto(&self, id: NodeId) -> NodeProto {
        lower_node(self.node(id).clone(), &self.values)
    }

    /// Graph inputs as ONNX value infos.
    pub fn input_infos(&self) -> Vec<ValueInfoProto> {
        self.inputs
            .iter()
            .map(|&v| self.value_info_proto(v))
            .collect()
    }

    /// Graph outputs as ONNX value infos.
    pub fn output_infos(&self) -> Vec<ValueInfoProto> {
        self.outputs
            .iter()
            .map(|&v| self.value_info_proto(v))
            .collect()
    }

    /// Ids of the live nodes in execution order. A snapshot, so the graph can
    /// be edited while iterating it.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes().map(|(id, _)| id).collect()
    }

    /// Live nodes in execution order.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.order
            .iter()
            .filter_map(|&id| Some((id, self.nodes[id.0 as usize].as_ref()?)))
    }

    /// Replaces the execution order.
    ///
    /// # Panics
    /// If `order` is not a permutation of the live nodes.
    pub fn set_order(&mut self, order: Vec<NodeId>) {
        let mut seen = vec![false; self.nodes.len()];
        for id in &order {
            let i = id.0 as usize;
            assert!(
                self.nodes[i].is_some() && !std::mem::replace(&mut seen[i], true),
                "order lists {id:?} twice or after its removal"
            );
        }
        assert_eq!(
            order.len(),
            self.nodes.iter().flatten().count(),
            "order misses live nodes"
        );
        self.order = order;
    }

    /// Whether `id` has not been removed.
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes[id.0 as usize].is_some()
    }

    /// # Panics
    /// If the node has been removed.
    pub fn node(&self, id: NodeId) -> &Node {
        self.nodes[id.0 as usize]
            .as_ref()
            .expect("node was removed")
    }

    /// Operands are not reachable from here; they are edited through
    /// [`Self::set_inputs`] and [`Self::set_output`], which keep producers
    /// and consumers in sync.
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        self.nodes[id.0 as usize]
            .as_mut()
            .expect("node was removed")
    }

    /// Every value of this graph, subgraphs excluded.
    pub fn value_ids(&self) -> impl Iterator<Item = ValueId> + use<> {
        (0..self.values.len() as u32).map(ValueId)
    }

    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.0 as usize]
    }

    pub fn value_mut(&mut self, id: ValueId) -> &mut Value {
        &mut self.values[id.0 as usize]
    }

    pub fn find_value(&self, name: &str) -> Option<ValueId> {
        self.by_name.get(name).copied()
    }

    /// The node producing `v`, if it is a node output.
    pub fn producer(&self, v: ValueId) -> Option<&Node> {
        self.value(v).producer.map(|n| self.node(n))
    }

    /// The initializer contents of `v`.
    pub fn data(&self, v: ValueId) -> Option<&TensorProto> {
        self.value(v).data.as_ref()
    }

    /// Initializers in model order.
    pub fn initializers(&self) -> &[ValueId] {
        &self.initializers
    }

    /// Turns `v` into an initializer holding `tensor`, appended after the
    /// existing ones.
    ///
    /// # Panics
    /// If `v` has a producer or already holds data.
    pub fn add_initializer(&mut self, v: ValueId, tensor: TensorProto) {
        let value = &mut self.values[v.0 as usize];
        assert!(
            value.producer.is_none() && value.data.is_none(),
            "{} is already defined",
            value.name
        );
        value.dtype = DType::from_onnx(tensor.data_type);
        value.shape = Some(tensor.dims.iter().map(|&d| Dim::Fixed(d)).collect());
        value.data = Some(tensor);
        self.initializers.push(v);
    }

    /// Adds an initializer holding `tensor`, named `name` or, if a value has
    /// that name, `name` with a number appended.
    pub fn new_initializer(&mut self, name: &str, mut tensor: TensorProto) -> ValueId {
        let mut unique = name.to_string();
        let mut n = 1;
        while self.by_name.contains_key(&unique) {
            unique = format!("{name}_{n}");
            n += 1;
        }
        tensor.name = unique.clone();
        let v = self.intern(&unique);
        self.add_initializer(v, tensor);
        v
    }

    /// Turns graph input `v` into an initializer holding `tensor`, replacing
    /// the default value an initializer of its name gave it.
    pub fn fix_input(&mut self, v: ValueId, tensor: TensorProto) {
        self.inputs.retain(|&i| i != v);
        let value = &mut self.values[v.0 as usize];
        value.dtype = DType::from_onnx(tensor.data_type);
        value.shape = Some(tensor.dims.iter().map(|&d| Dim::Fixed(d)).collect());
        if value.data.replace(tensor).is_none() {
            self.initializers.push(v);
        }
    }

    /// Replaces `node` by the nodes of `branch`, one of its subgraphs, run
    /// where it ran: `branch` reads the values it captures from this graph,
    /// and its outputs become the node's, through `Identity` nodes. Returns
    /// false, leaving the graph as it was, if a value `branch` defines has the
    /// name of a value of this graph.
    pub fn inline_subgraph(&mut self, node: NodeId, branch: &Graph) -> bool {
        let defined: Vec<ValueId> = branch
            .value_ids()
            .filter(|&v| branch.value(v).producer.is_some() || branch.value(v).data.is_some())
            .collect();
        if defined
            .iter()
            .any(|&v| self.by_name.contains_key(&branch.value(v).name))
        {
            return false;
        }
        let outputs = self.node(node).outputs.clone();
        let pos = self.order.iter().position(|&n| n == node).unwrap();
        self.remove_node(node);
        let first = self.order.len();
        for &v in branch.initializers() {
            let value = branch.value(v);
            let new = self.intern(&value.name);
            self.add_initializer(new, value.data.clone().unwrap());
        }
        for (id, _) in branch.nodes() {
            self.push_node(branch.node_proto(id));
        }
        for (&out, &src) in outputs.iter().zip(&branch.outputs) {
            if let Some(out) = out {
                self.push_node(NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec![branch.value(src).name.clone()],
                    output: vec![self.value(out).name.clone()],
                    ..Default::default()
                });
            }
        }
        for &v in &defined {
            let (value, new) = (branch.value(v), self.by_name[&branch.value(v).name]);
            let new = &mut self.values[new.0 as usize];
            new.dtype = value.dtype;
            new.shape = value.shape.clone();
        }
        let inlined: Vec<NodeId> = self.order.drain(first..).collect();
        self.order.splice(pos..pos, inlined);
        true
    }

    /// Number of reads of `v`, counting each graph output it is as one.
    pub fn uses(&self, v: ValueId) -> usize {
        self.value(v).consumers.len() + self.outputs.iter().filter(|&&o| o == v).count()
    }

    /// Replaces all inputs of `node`.
    pub fn set_inputs(&mut self, node: NodeId, inputs: Vec<Option<ValueId>>) {
        let old = std::mem::replace(&mut self.node_mut(node).inputs, inputs.clone());
        for v in old.into_iter().flatten() {
            self.unlink_consumer(v, node);
        }
        for v in inputs.into_iter().flatten() {
            self.values[v.0 as usize].consumers.push(node);
        }
    }

    /// Makes `node` produce `value` as its output `i`. The previous output
    /// is left without a producer.
    ///
    /// # Panics
    /// If `value` already has a producer.
    pub fn set_output(&mut self, node: NodeId, i: usize, value: Option<ValueId>) {
        if let Some(v) = value {
            assert!(
                self.value(v).producer.is_none(),
                "{} already has a producer",
                self.value(v).name
            );
            self.values[v.0 as usize].producer = Some(node);
        }
        let old = std::mem::replace(&mut self.node_mut(node).outputs[i], value);
        if let Some(v) = old {
            self.values[v.0 as usize].producer = None;
        }
    }

    /// Adds a node running right before `anchor` and returns its id.
    /// `proto` carries the name and any attributes `op` does not model.
    ///
    /// # Panics
    /// If one of `outputs` already has a producer.
    pub fn insert_node(
        &mut self,
        anchor: NodeId,
        op: Op,
        proto: NodeProto,
        inputs: Vec<Option<ValueId>>,
        outputs: Vec<Option<ValueId>>,
    ) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        for &v in outputs.iter().flatten() {
            let value = &mut self.values[v.0 as usize];
            assert!(
                value.producer.is_none(),
                "{} already has a producer",
                value.name
            );
            value.producer = Some(id);
        }
        for &v in inputs.iter().flatten() {
            self.values[v.0 as usize].consumers.push(id);
        }
        self.nodes.push(Some(Node {
            op,
            proto: NodeProto {
                input: Vec::new(),
                output: Vec::new(),
                ..proto
            },
            inputs,
            outputs,
            captures: Vec::new(),
        }));
        let pos = self
            .order
            .iter()
            .position(|&n| n == anchor)
            .expect("anchor is not scheduled");
        self.order.insert(pos, id);
        id
    }

    /// Replaces `nodes` by one node running `op` where the last of them ran.
    /// The new node takes `inputs` and produces what the last node produced.
    /// `proto` carries the name and any attributes `op` does not model.
    ///
    /// The caller must make sure nothing outside `nodes` reads their other
    /// outputs.
    pub fn fuse(
        &mut self,
        nodes: &[NodeId],
        op: Op,
        proto: NodeProto,
        inputs: Vec<Option<ValueId>>,
    ) -> NodeId {
        let last = *nodes
            .iter()
            .max_by_key(|&&n| self.order.iter().position(|&o| o == n))
            .expect("nothing to fuse");
        let outputs = self.node(last).outputs.clone();
        for &n in nodes {
            self.remove_node(n);
        }
        let id = self.insert_node(last, op, proto, inputs, outputs);
        self.order.retain(|&n| n != last);
        id
    }

    /// Removes `node`. Its outputs are left without a producer.
    pub fn remove_node(&mut self, node: NodeId) {
        let n = self.nodes[node.0 as usize]
            .take()
            .expect("node was removed");
        for v in n.inputs.into_iter().flatten().chain(n.captures) {
            self.unlink_consumer(v, node);
        }
        for v in n.outputs.into_iter().flatten() {
            self.values[v.0 as usize].producer = None;
        }
    }

    /// Values this graph reads from enclosing scopes.
    pub fn captures(&self) -> impl Iterator<Item = ValueId> + '_ {
        let inputs: HashSet<ValueId> = self.inputs.iter().copied().collect();
        (0..self.values.len() as u32)
            .map(ValueId)
            .filter(move |&v| {
                let value = self.value(v);
                value.producer.is_none()
                    && value.data.is_none()
                    && !inputs.contains(&v)
                    && (!value.consumers.is_empty() || self.outputs.contains(&v))
            })
    }

    fn intern(&mut self, name: &str) -> ValueId {
        if let Some(&v) = self.by_name.get(name) {
            return v;
        }
        let v = ValueId(self.values.len() as u32);
        self.values.push(Value::new(name));
        self.by_name.insert(name.to_string(), v);
        v
    }

    fn intern_operand(&mut self, name: &str) -> Option<ValueId> {
        (!name.is_empty()).then(|| self.intern(name))
    }

    fn push_node(&mut self, mut proto: NodeProto) {
        let id = NodeId(self.nodes.len() as u32);
        // Hand graph attributes to the parser. An untyped operator keeps the
        // ones it did not take as IR too; a typed one leaves them in the
        // proto.
        let mut graphs: Vec<(String, GraphProto)> = proto
            .attribute
            .iter_mut()
            .filter_map(|a| Some((a.name.clone(), a.g.take()?)))
            .collect();
        let mut op = Op::parse(&mut proto, |name| {
            let i = graphs.iter().position(|(n, _)| n == name)?;
            Some(Graph::from_onnx(graphs.swap_remove(i).1))
        });
        for (name, g) in graphs {
            if let Op::Onnx { graphs, .. } = &mut op {
                graphs.push((name, Graph::from_onnx(g)));
            } else if let Some(a) = proto.attribute.iter_mut().find(|a| a.name == name) {
                a.g = Some(g);
            }
        }

        let mut captured: Vec<String> = Vec::new();
        for sub in op.subgraphs() {
            captured.extend(sub.captures().map(|v| sub.value(v).name.clone()));
        }
        for a in &proto.attribute {
            for g in a.g.iter().chain(&a.graphs) {
                free_names(g, &mut captured);
            }
        }
        captured.sort();
        captured.dedup();

        let inputs: Vec<_> = proto.input.iter().map(|n| self.intern_operand(n)).collect();
        let outputs: Vec<_> = proto
            .output
            .iter()
            .map(|n| self.intern_operand(n))
            .collect();
        let captures: Vec<_> = captured.iter().map(|n| self.intern(n)).collect();
        for &v in inputs.iter().flatten().chain(&captures) {
            self.values[v.0 as usize].consumers.push(id);
        }
        for &v in outputs.iter().flatten() {
            self.values[v.0 as usize].producer = Some(id);
        }
        proto.input.clear();
        proto.output.clear();
        self.nodes.push(Some(Node {
            op,
            proto,
            inputs,
            outputs,
            captures,
        }));
        self.order.push(id);
    }

    fn unlink_consumer(&mut self, v: ValueId, node: NodeId) {
        let consumers = &mut self.values[v.0 as usize].consumers;
        if let Some(i) = consumers.iter().position(|&n| n == node) {
            consumers.remove(i);
        }
    }

    fn set_type(&mut self, v: ValueId, info: &ValueInfoProto) {
        let Some(type_proto::Value::TensorType(t)) =
            info.r#type.as_ref().and_then(|t| t.value.as_ref())
        else {
            return;
        };
        let value = &mut self.values[v.0 as usize];
        value.dtype = DType::from_onnx(t.elem_type);
        value.shape = t.shape.as_ref().map(|s| {
            s.dim
                .iter()
                .map(|d| match &d.value {
                    Some(tensor_shape_proto::dimension::Value::DimValue(n)) => Dim::Fixed(*n),
                    Some(tensor_shape_proto::dimension::Value::DimParam(s)) => Dim::Sym(s.clone()),
                    None => Dim::Unknown,
                })
                .collect()
        });
    }

    fn value_info_proto(&self, v: ValueId) -> ValueInfoProto {
        let value = self.value(v);
        let typed = value.dtype != DType::Undefined || value.shape.is_some();
        let shape = value.shape.as_ref().map(|dims| TensorShapeProto {
            dim: dims
                .iter()
                .map(|d| tensor_shape_proto::Dimension {
                    value: match d {
                        Dim::Fixed(n) => Some(tensor_shape_proto::dimension::Value::DimValue(*n)),
                        Dim::Sym(s) => {
                            Some(tensor_shape_proto::dimension::Value::DimParam(s.clone()))
                        }
                        Dim::Unknown => None,
                    },
                    ..Default::default()
                })
                .collect(),
        });
        ValueInfoProto {
            name: value.name.clone(),
            r#type: typed.then(|| TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: value.dtype.to_onnx(),
                    shape,
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

/// Writes `node`'s operator, operands and modelled attributes into its proto.
fn lower_node(node: Node, values: &[Value]) -> NodeProto {
    let name =
        |v: &Option<ValueId>| v.map_or_else(String::new, |v| values[v.0 as usize].name.clone());
    let mut p = node.proto;
    p.op_type = node.op.op_type().to_string();
    if let Op::Onnx { domain, .. } = &node.op {
        p.domain = domain.clone();
    }
    p.input = node.inputs.iter().map(name).collect();
    p.output = node.outputs.iter().map(name).collect();
    node.op.into_attrs(&mut p.attribute);
    p
}

/// Appends the names `g` reads but does not define, i.e. those it captures
/// from enclosing scopes.
fn free_names(g: &GraphProto, out: &mut Vec<String>) {
    let mut defined: HashSet<&str> = g.input.iter().map(|i| i.name.as_str()).collect();
    defined.extend(g.initializer.iter().map(|t| t.name.as_str()));
    defined.extend(
        g.node
            .iter()
            .flat_map(|n| n.output.iter().map(String::as_str)),
    );
    let mut used: Vec<String> = g.output.iter().map(|o| o.name.clone()).collect();
    for node in &g.node {
        used.extend(node.input.iter().cloned());
        for a in &node.attribute {
            for sub in a.g.iter().chain(&a.graphs) {
                free_names(sub, &mut used);
            }
        }
    }
    out.extend(
        used.into_iter()
            .filter(|n| !n.is_empty() && !defined.contains(n.as_str())),
    );
}
