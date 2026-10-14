//! Layout of the weights file.

use super::ir::{DType, Graph, Op};
use super::{WeightTable, sanitize_name};
use crate::model::onnx_proto::TensorProto;
use crate::model::{tensor_to_array, tensor_to_vec_u8};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Where a weight lives in the weights file.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WeightRef {
    /// Byte offset.
    pub offset: usize,
    /// Length in bytes.
    pub len: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// The weights file, with where each weight landed in it.
#[derive(Default)]
pub(crate) struct Weights {
    pub data: Vec<u8>,
    /// Weights by sanitized name.
    lookup: HashMap<String, WeightRef>,
    /// `lookup` in the form [`Compiler::with_pattern`](super::Compiler::with_pattern)
    /// generators take, built on first use.
    pattern_table: OnceLock<WeightTable>,
    /// Values of int64 and int32 weights by name, for code generation to
    /// inline.
    pub int64: HashMap<String, (Vec<i64>, Vec<usize>)>,
    /// Offset and length of each distinct content already written.
    by_content: HashMap<u64, (usize, usize)>,
}

impl Weights {
    /// Lays out the initializers and `Constant` values of `g` and its
    /// subgraphs: per graph, its initializers in model order, then each
    /// node's constant followed by its subgraphs' weights, in attribute
    /// order. Each weight
    /// starts 16-byte aligned; identical contents are stored once.
    pub(crate) fn collect(g: &Graph) -> Self {
        let mut weights = Weights::default();
        weights.add_graph(g);
        weights
    }

    /// The weight holding the tensor `name`.
    pub(crate) fn get(&self, name: &str) -> Option<&WeightRef> {
        self.lookup.get(&sanitize_name(name))
    }

    /// The weights as pattern generators take them.
    pub(crate) fn pattern_table(&self) -> &WeightTable {
        self.pattern_table.get_or_init(|| {
            self.lookup
                .iter()
                .map(|(name, w)| {
                    let entry = (w.offset, w.len, w.shape.clone(), w.dtype.to_onnx());
                    (name.clone(), entry)
                })
                .collect()
        })
    }

    /// The values of `name`, if it is an int64 or int32 weight.
    pub(crate) fn ints(&self, name: &str) -> Option<&[i64]> {
        self.int64.get(name).map(|(ints, _)| ints.as_slice())
    }

    /// Whether the tensor `name` is read from the weights rather than
    /// computed.
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.get(name).is_some() || self.int64.contains_key(name)
    }

    fn add_graph(&mut self, g: &Graph) {
        for &v in g.initializers() {
            // Those that folding left without readers are not written.
            if let Some(t) = g.data(v).filter(|_| g.uses(v) > 0) {
                self.add(&g.value(v).name, t);
            }
        }
        for (_, node) in g.nodes() {
            if let Op::Constant { value } = &node.op
                && let Some(out) = node.output(0)
            {
                self.add(&g.value(out).name, value);
            }
            // In the order the node lists them.
            let mut subgraphs = node.op.named_subgraphs();
            subgraphs
                .sort_by_key(|(name, _)| node.proto.attribute.iter().position(|a| a.name == *name));
            for (_, sub) in subgraphs {
                self.add_graph(sub);
            }
        }
    }

    fn add(&mut self, name: &str, t: &TensorProto) {
        let dtype = DType::from_onnx(t.data_type);
        if matches!(dtype, DType::I64 | DType::I32)
            && let Ok((data, dims)) = tensor_to_array(t)
        {
            let ints = data.iter().map(|&x| x as i64).collect();
            self.int64.insert(name.to_string(), (ints, dims));
        }
        let Ok((bytes, shape, _)) = tensor_to_vec_u8(t) else {
            return;
        };
        if bytes.is_empty() {
            return;
        }
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut hasher);
            hasher.finish()
        };
        let (offset, len) = *self.by_content.entry(hash).or_insert_with(|| {
            self.data.resize(self.data.len().next_multiple_of(16), 0);
            let offset = self.data.len();
            self.data.extend_from_slice(&bytes);
            (offset, bytes.len())
        });
        // The first weight of a name wins.
        self.lookup.entry(sanitize_name(name)).or_insert(WeightRef {
            offset,
            len,
            shape,
            dtype,
        });
    }
}
