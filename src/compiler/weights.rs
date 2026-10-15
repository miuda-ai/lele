//! Layout of the weights file.

use super::generate::ElemType;
use super::ir::{DType, Graph, Layout, Op};
use super::sanitize_name;
use crate::model::onnx_proto::TensorProto;
use crate::model::{tensor_to_array, tensor_to_vec_u8};
use std::cell::RefCell;
use std::collections::HashMap;

/// Where a weight lives in the weights file. Only this module knows where:
/// code generation reads a weight through the expressions it builds.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WeightRef {
    /// Byte offset.
    offset: usize,
    /// Length in bytes.
    len: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// An f16 matrix stored as [`pack_f16`](crate::kernels::matmul::pack_f16)
    /// lays it out (or lays out its transpose: see
    /// [`Layout::PackedF16`]).
    pub packed: bool,
    /// The value of a rank-0 weight, which code reads inline: it is not
    /// written to the weights file, and has no offset there.
    pub scalar: Option<Scalar>,
}

impl WeightRef {
    /// The call of `lele::runtime::Weights` method `accessor` reading this
    /// weight, on `self` in a model method. A scalar, which is not in the
    /// weights file, is read inline, as the type `accessor` reads.
    pub(crate) fn read(&self, accessor: &str) -> String {
        match self.scalar {
            Some(scalar) => scalar.read(accessor),
            None => format!(
                "self.{accessor}({}, {}, &{:?})",
                self.offset, self.len, self.shape
            ),
        }
    }

    /// The byte offset and length of this weight, as the arguments the
    /// runtime methods reading raw bytes (`pack_qweights`, ...) start with.
    ///
    /// # Panics
    /// If this is a scalar, which is not in the weights file.
    pub(crate) fn bytes(&self) -> String {
        assert!(
            self.scalar.is_none(),
            "a scalar weight has no bytes in the file"
        );
        format!("{}, {}", self.offset, self.len)
    }

    /// The expression reading this weight as `target`, through the accessor
    /// widening its type to `target`.
    pub(crate) fn expr(&self, target: ElemType) -> String {
        let accessor = match target {
            ElemType::I64 => match self.dtype {
                DType::I32 => "weight_i32_i64",
                DType::Bool => "weight_bool_i64",
                _ => "weight_i64",
            },
            ElemType::F32 => match self.dtype {
                DType::U8 => "weight_u8",
                DType::I8 => "weight_i8",
                DType::I32 => "weight_i32_f32",
                DType::I64 => "weight_i64_f32",
                DType::F16 => "weight_f16",
                _ => "weight_f32",
            },
        };
        self.read(accessor)
    }
}

/// The value of a rank-0 weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scalar {
    /// Of an int64, int32, 8-bit or bool weight.
    Int(i64),
    /// Of an f32 or f16 weight.
    Float(f32),
}

impl Scalar {
    /// The scalar inline, as the type `lele::runtime::Weights` method
    /// `accessor` reads a weight as.
    fn read(self, accessor: &str) -> String {
        let (int, float) = match self {
            Scalar::Int(v) => (v, v as f32),
            Scalar::Float(v) => (v as i64, v),
        };
        let float = match float {
            v if v.is_finite() => format!("{v:?}f32"),
            v if v.is_nan() => "f32::NAN".to_string(),
            v if v > 0.0 => "f32::INFINITY".to_string(),
            _ => "f32::NEG_INFINITY".to_string(),
        };
        match accessor {
            "weight_i64" | "weight_i32_i64" | "weight_bool_i64" => {
                format!("lele::scalar({int}i64)")
            }
            "weight_i32" => format!("lele::scalar({int}i32)"),
            "weight_f32" | "weight_f16" | "weight_u8" | "weight_i8" | "weight_i32_f32"
            | "weight_i64_f32" => format!("lele::scalar({float})"),
            _ => panic!("a scalar weight is not read through {accessor}"),
        }
    }
}

/// The weights file, with where each weight landed in it.
#[derive(Default)]
pub(crate) struct Weights {
    pub data: Vec<u8>,
    /// Weights by sanitized name.
    lookup: HashMap<String, WeightRef>,
    /// Values of int64 and int32 weights by name, for code generation to
    /// inline.
    pub int64: HashMap<String, (Vec<i64>, Vec<usize>)>,
    /// Offset and length of each distinct content already written.
    by_content: HashMap<u64, (usize, usize)>,
    /// Weights of the integer products, packed when the model is created.
    pub prepared: RefCell<Prepared>,
}

/// Weights generated code prepares for the integer kernels when the model is
/// created, as the expressions preparing each, in the order code numbers them.
#[derive(Default)]
pub(crate) struct Prepared {
    pub qweights: Vec<String>,
    pub conv_weights: Vec<String>,
}

/// The number of `expr` in `prepared`, added if new.
fn number(prepared: &mut Vec<String>, expr: String) -> usize {
    prepared.iter().position(|e| *e == expr).unwrap_or_else(|| {
        prepared.push(expr);
        prepared.len() - 1
    })
}

impl Weights {
    /// The number of the [`QWeights`](crate::kernels::QWeights) that `expr`
    /// prepares from `weights` when the model is created.
    pub(crate) fn qweights(&self, expr: String) -> usize {
        number(&mut self.prepared.borrow_mut().qweights, expr)
    }

    /// The number of the [`ConvWeights`](crate::kernels::ConvWeights) that
    /// `expr` prepares from `weights` when the model is created.
    pub(crate) fn conv_weights(&self, expr: String) -> usize {
        number(&mut self.prepared.borrow_mut().conv_weights, expr)
    }

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

    /// The values of `name`, if it is an int64 or int32 weight.
    pub(crate) fn ints(&self, name: &str) -> Option<&[i64]> {
        self.int64.get(name).map(|(ints, _)| ints.as_slice())
    }

    /// Byte offset and length of every weight of `dtype` stored as it is (not
    /// packed) that `code` reads through `accessor`, in file order, for the
    /// runtime to widen once at load.
    pub(crate) fn read_through(
        &self,
        dtype: DType,
        accessor: &str,
        code: &str,
    ) -> Vec<(usize, usize)> {
        let mut regions: Vec<_> = self
            .lookup
            .values()
            .filter(|w| w.dtype == dtype && !w.packed)
            .map(|w| (w.offset, w.len))
            .collect();
        regions.sort_unstable();
        regions.dedup();
        regions.retain(|(offset, len)| code.contains(&format!("self.{accessor}({offset}, {len},")));
        regions
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
                let value = g.value(v);
                self.add(&value.name, t, value.layout == Layout::PackedF16);
            }
        }
        for (_, node) in g.nodes() {
            if let Op::Constant { value } = &node.op
                && let Some(out) = node.output(0)
            {
                self.add(&g.value(out).name, value, false);
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

    fn add(&mut self, name: &str, t: &TensorProto, packed: bool) {
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
        let scalar = match dtype {
            _ if !shape.is_empty() => None,
            DType::I64 => bytes.first_chunk().map(|&b| Scalar::Int(i64::from_le_bytes(b))),
            DType::I32 => bytes.first_chunk().map(|&b| Scalar::Int(i32::from_le_bytes(b).into())),
            DType::Bool => Some(Scalar::Int((bytes[0] != 0).into())),
            DType::U8 => Some(Scalar::Int(bytes[0].into())),
            DType::I8 => Some(Scalar::Int((bytes[0] as i8).into())),
            DType::F32 => bytes.first_chunk().map(|&b| Scalar::Float(f32::from_le_bytes(b))),
            DType::F16 => bytes
                .first_chunk()
                .map(|&b| Scalar::Float(half::f16::from_le_bytes(b).to_f32())),
            _ => None,
        };
        // Code reads scalars inline, so they are not written.
        let (offset, len) = if scalar.is_some() {
            (0, 0)
        } else {
            let hash = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                bytes.hash(&mut hasher);
                hasher.finish()
            };
            *self.by_content.entry(hash).or_insert_with(|| {
                self.data.resize(self.data.len().next_multiple_of(16), 0);
                let offset = self.data.len();
                self.data.extend_from_slice(&bytes);
                (offset, bytes.len())
            })
        };
        // The first weight of a name wins.
        self.lookup.entry(sanitize_name(name)).or_insert(WeightRef {
            offset,
            len,
            shape,
            dtype,
            packed,
            scalar,
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::compiler::Compiler;
    use crate::compiler::ir::DType;
    use crate::model::onnx_proto::{GraphProto, NodeProto, TensorProto, ValueInfoProto};

    /// Code reads scalar weights inline, whatever their type, and the weights
    /// file leaves them out.
    #[test]
    fn scalars_are_not_written() {
        let info = |name: &str| ValueInfoProto {
            name: name.to_string(),
            ..Default::default()
        };
        let tensor = |name: &str, dtype: DType, dims: Vec<i64>, raw_data: Vec<u8>| TensorProto {
            name: name.to_string(),
            data_type: dtype.to_onnx(),
            dims,
            raw_data,
            ..Default::default()
        };
        let node = |op: &str, input: &[&str], output: &str| NodeProto {
            op_type: op.to_string(),
            input: input.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        };
        let row: Vec<u8> = [7.0f32, 9.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let graph = GraphProto {
            node: vec![
                node("Add", &["x", "s"], "a"),
                node("Add", &["a", "v"], "b"),
                node("Identity", &["q"], "c"),
            ],
            initializer: vec![
                tensor("s", DType::F32, vec![], 2.5f32.to_le_bytes().to_vec()),
                tensor("v", DType::F32, vec![2], row.clone()),
                tensor("q", DType::U8, vec![], vec![3]),
            ],
            input: vec![info("x")],
            output: vec![info("b"), info("c")],
            ..Default::default()
        };
        let result = Compiler::new().compile(&graph).unwrap();
        let code = &result.code;
        assert!(code.contains("lele::scalar(2.5f32)"), "{code}");
        assert!(code.contains("lele::scalar(3.0f32)"), "{code}");
        assert_eq!(result.weights, row);
    }
}
