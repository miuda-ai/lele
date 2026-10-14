//! Layout of the weights file.

use super::ir::{Graph, Op};
use super::sanitize_name;
use crate::model::onnx_proto::TensorProto;
use crate::model::{tensor_to_array, tensor_to_vec_u8};
use std::collections::HashMap;

/// Where a weight lives in the weights file: byte offset, byte length,
/// shape and ONNX data type.
pub(crate) type WeightRef = (usize, usize, Vec<usize>, i32);

/// The weights file, with where each weight landed in it.
#[derive(Default)]
pub(crate) struct Weights {
    pub data: Vec<u8>,
    /// Weights by sanitized name.
    pub lookup: HashMap<String, WeightRef>,
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

    fn add_graph(&mut self, g: &Graph) {
        for &v in g.initializers() {
            if let Some(t) = g.data(v) {
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
        if (t.data_type == 7 || t.data_type == 6)
            && let Ok((data, dims)) = tensor_to_array(t)
        {
            let ints = data.iter().map(|&x| x as i64).collect();
            self.int64.insert(name.to_string(), (ints, dims));
        }
        let Ok((bytes, shape, data_type)) = tensor_to_vec_u8(t) else {
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
        self.lookup
            .entry(sanitize_name(name))
            .or_insert((offset, len, shape, data_type));
    }
}
