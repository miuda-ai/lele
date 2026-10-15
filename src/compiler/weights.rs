//! Layout of the weights file.

use super::ir::{ConvAttrs, DType, Graph, Op, ValueId};
use super::{WeightTable, sanitize_name};
use crate::kernels::matmul::PACK_COLS;
use crate::model::onnx_proto::TensorProto;
use crate::model::{tensor_to_array, tensor_to_vec_u8};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
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
    /// An f16 matrix stored as [`pack_f16`](crate::kernels::matmul::pack_f16)
    /// lays it out (or lays out its transpose: see [`Packing`]).
    pub packed: bool,
    /// The value of a rank-0 weight, which code reads inline.
    pub scalar: Option<Scalar>,
}

/// The value of a rank-0 weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scalar {
    /// Of an int64, int32, 8-bit or bool weight.
    Int(i64),
    /// Of an f32 or f16 weight.
    Float(f32),
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
    /// starts 16-byte aligned; identical contents are stored once. With
    /// `pack`, f16 initializers only read as the right operand of `MatMul`
    /// and `Linear`, or as the weights of pointwise `Conv`s, are packed
    /// ([`packable`]).
    pub(crate) fn collect(g: &Graph, pack: bool) -> Self {
        let mut weights = Weights::default();
        weights.add_graph(g, pack);
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

    /// Byte offset and length of every weight of `dtype` stored as it is (not
    /// packed), in file order, for the runtime to widen once at load.
    pub(crate) fn regions(&self, dtype: DType) -> Vec<(usize, usize)> {
        let mut regions: Vec<_> = self
            .lookup
            .values()
            .filter(|w| w.dtype == dtype && !w.packed)
            .map(|w| (w.offset, w.len))
            .collect();
        regions.sort_unstable();
        regions.dedup();
        regions
    }

    /// Whether the tensor `name` is read from the weights rather than
    /// computed.
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.get(name).is_some() || self.int64.contains_key(name)
    }

    fn add_graph(&mut self, g: &Graph, pack: bool) {
        let mut sub_reads = HashSet::new();
        if pack {
            for (_, node) in g.nodes() {
                for sub in node.op.subgraphs() {
                    names_read(sub, &mut sub_reads);
                }
            }
        }
        for &v in g.initializers() {
            // Those that folding left without readers are not written.
            if let Some(t) = g.data(v).filter(|_| g.uses(v) > 0) {
                let name = &g.value(v).name;
                let packing = (pack && !sub_reads.contains(name))
                    .then(|| packable(g, v))
                    .flatten();
                self.add(name, t, packing);
            }
        }
        for (_, node) in g.nodes() {
            if let Op::Constant { value } = &node.op
                && let Some(out) = node.output(0)
            {
                self.add(&g.value(out).name, value, None);
            }
            // In the order the node lists them.
            let mut subgraphs = node.op.named_subgraphs();
            subgraphs
                .sort_by_key(|(name, _)| node.proto.attribute.iter().position(|a| a.name == *name));
            for (_, sub) in subgraphs {
                self.add_graph(sub, pack);
            }
        }
    }

    fn add(&mut self, name: &str, t: &TensorProto, packing: Option<Packing>) {
        let dtype = DType::from_onnx(t.data_type);
        if matches!(dtype, DType::I64 | DType::I32)
            && let Ok((data, dims)) = tensor_to_array(t)
        {
            let ints = data.iter().map(|&x| x as i64).collect();
            self.int64.insert(name.to_string(), (ints, dims));
        }
        let Ok((mut bytes, shape, _)) = tensor_to_vec_u8(t) else {
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
        if let Some(packing) = packing {
            let mut halves: Vec<half::f16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| half::f16::from_le_bytes(b))
                .collect();
            let (k, n) = match packing {
                Packing::Matrix => (shape[0], shape[1]),
                Packing::Transposed => {
                    let (m, k) = (shape[0], shape[1]);
                    halves = (0..k * m).map(|i| halves[i % m * k + i / m]).collect();
                    (k, m)
                }
            };
            let packed = crate::kernels::matmul::pack_f16(&halves, k, n);
            bytes = packed.iter().flat_map(|h| h.to_le_bytes()).collect();
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
            packed: packing.is_some(),
            scalar,
        });
    }
}

/// How an f16 weight is packed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Packing {
    /// A `[k, n]` right operand, packed as it is.
    Matrix,
    /// The `[m, k, 1]` weights of a pointwise `Conv` (a left operand), packed
    /// as their transpose `[k, m]`.
    Transposed,
}

/// How initializer `v` of `g` can be stored packed, if it can: an f16 matrix
/// with a multiple of [`PACK_COLS`] columns, every read of which (in `g`) is
/// as the right operand of a `MatMul` or `Linear`; or f16 weights `[m, k, 1]`,
/// `m` a multiple of [`PACK_COLS`], every read of which is by a `Conv` code
/// generation computes with `conv1d_pointwise`.
fn packable(g: &Graph, v: ValueId) -> Option<Packing> {
    let value = g.value(v);
    let shape = g.data(v).map(|t| t.dims.as_slice()).unwrap_or_default();
    let cols = |n: i64| n > 0 && n as usize % PACK_COLS == 0;
    let packing = match *shape {
        [k, n] if k > 0 && cols(n) => Packing::Matrix,
        [m, k, 1] if k > 1 && cols(m) => Packing::Transposed,
        _ => return None,
    };
    let read_by = |id| {
        let node = g.node(id);
        let op_reads = match (&node.op, packing) {
            (Op::MatMul | Op::Linear, Packing::Matrix) => true,
            (Op::Conv(conv), Packing::Transposed) => is_pointwise(conv),
            _ => false,
        };
        op_reads
            && node.input(1) == Some(v)
            && node.inputs().iter().filter(|&&i| i == Some(v)).count() == 1
    };
    (value.dtype == DType::F16
        && value.consumers().len() == g.uses(v)
        && value.consumers().iter().all(|&id| read_by(id)))
    .then_some(packing)
}

/// Whether a `Conv` with weights `[m, k, 1]`, `k > 1`, is one code generation
/// computes with `conv1d_pointwise`: one group, stride and dilation 1, no
/// padding.
fn is_pointwise(conv: &ConvAttrs) -> bool {
    let ones = |v: &Option<Vec<i64>>| v.iter().flatten().all(|&x| x == 1);
    matches!(conv.auto_pad.as_deref(), None | Some("NOTSET" | ""))
        && conv.group.unwrap_or(1) == 1
        && ones(&conv.strides)
        && ones(&conv.dilations)
        && conv.pads.iter().flatten().all(|&p| p == 0)
}

/// Adds the names of the values read by the nodes of `g` and its subgraphs,
/// and of its outputs, to `names`.
fn names_read(g: &Graph, names: &mut HashSet<String>) {
    for (_, node) in g.nodes() {
        for v in node.inputs().iter().flatten() {
            names.insert(g.value(*v).name.clone());
        }
        for sub in node.op.subgraphs() {
            names_read(sub, names);
        }
    }
    names.extend(g.output_infos().into_iter().map(|o| o.name));
}
