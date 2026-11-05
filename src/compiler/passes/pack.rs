//! Packing of matrix product weights.

use super::super::ir::{ConvAttrs, DType, Graph, Layout, Node, Op, ValueId};
use crate::kernels::matmul::PACK_COLS;
use crate::model::onnx_proto::TensorProto;
use crate::model::tensor_to_vec_u8;
use std::collections::HashSet;

/// Lays out the f16 initializers of `g` and its subgraphs read only as the
/// right operand of `MatMul`, `Linear` and `SwiGlu`, or as the weights of
/// pointwise `Conv`s, as the matmul kernel reads them
/// ([`pack_f16`](crate::kernels::matmul::pack_f16)), so that each product
/// streams its weight from start to end ([`Layout::Packed`]); and so the
/// integer codes read only by `QWeightMatMul`s. Reads by nodes `skip` holds,
/// which code generation leaves to someone else, keep an initializer as it is
/// stored, as do reads by subgraphs.
pub(crate) fn pack_weights(g: &mut Graph, skip: &dyn Fn(&Node) -> bool) {
    let mut sub_reads = HashSet::new();
    for (_, node) in g.nodes() {
        for sub in node.op.subgraphs() {
            names_read(sub, &mut sub_reads);
        }
    }
    let packed: Vec<(ValueId, TensorProto)> = g
        .initializers()
        .iter()
        .filter(|&&v| g.uses(v) > 0 && !sub_reads.contains(&g.value(v).name))
        .filter_map(|&v| Some((v, pack(g.data(v)?, packable(g, v, skip)?)?)))
        .collect();
    for (v, tensor) in packed {
        g.set_data(v, tensor);
        g.value_mut(v).layout = Layout::Packed;
    }
    for id in g.node_ids() {
        for sub in g.node_mut(id).op.subgraphs_mut() {
            pack_weights(sub, skip);
        }
    }
}

/// How a weight is packed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Packing {
    /// A `[k, n]` f16 right operand, packed as it is.
    Matrix,
    /// The `[m, k, 1]` f16 weights of a pointwise `Conv` (a left operand),
    /// packed as their transpose `[k, m]`.
    Transposed,
    /// The `[k, n]` one-byte integer codes of a `QWeightMatMul`.
    Codes,
}

/// `t` packed as `packing` says, its dims kept.
fn pack(t: &TensorProto, packing: Packing) -> Option<TensorProto> {
    let (bytes, shape, _) = tensor_to_vec_u8(t).ok()?;
    let halves = || -> Vec<half::f16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| half::f16::from_le_bytes(b))
            .collect()
    };
    let f16_bytes = |h: Vec<half::f16>| h.iter().flat_map(|h| h.to_le_bytes()).collect();
    let raw_data = match packing {
        Packing::Matrix => f16_bytes(crate::kernels::matmul::pack_f16(&halves(), shape[0], shape[1])),
        Packing::Transposed => {
            let (m, k) = (shape[0], shape[1]);
            let halves = halves();
            let transposed: Vec<_> = (0..k * m).map(|i| halves[i % m * k + i / m]).collect();
            f16_bytes(crate::kernels::matmul::pack_f16(&transposed, k, m))
        }
        Packing::Codes => crate::kernels::matmul::pack_codes(&bytes, shape[0], shape[1]),
    };
    Some(TensorProto {
        data_type: t.data_type,
        dims: t.dims.clone(),
        raw_data,
        ..Default::default()
    })
}

/// How initializer `v` of `g` can be stored packed, if it can: an f16 matrix
/// with a multiple of [`PACK_COLS`] columns, every read of which (in `g`) is
/// as the right operand of a `MatMul`, `Linear` or `SwiGlu`; or f16 weights
/// `[m, k, 1]`, `m` a multiple of [`PACK_COLS`], every read of which is by a
/// `Conv` or `ConvAct` code generation computes with `conv1d_pointwise`; or a
/// matrix of int8 or uint8 codes, likewise wide, read only as the weight of
/// `QWeightMatMul`s. None if `skip` holds one of the nodes reading it.
fn packable(g: &Graph, v: ValueId, skip: &dyn Fn(&Node) -> bool) -> Option<Packing> {
    let value = g.value(v);
    let shape = g.data(v).map(|t| t.dims.as_slice()).unwrap_or_default();
    let cols = |n: i64| n > 0 && n as usize % PACK_COLS == 0;
    let codes = matches!(value.dtype, DType::I8 | DType::U8);
    let packing = match *shape {
        [k, n] if k > 0 && cols(n) && codes => Packing::Codes,
        [k, n] if k > 0 && cols(n) => Packing::Matrix,
        [m, k, 1] if k > 1 && cols(m) => Packing::Transposed,
        _ => return None,
    };
    let read_by = |id| {
        let node = g.node(id);
        let op_reads = match (&node.op, packing) {
            (Op::MatMul | Op::Linear { .. } | Op::SwiGlu, Packing::Matrix) => true,
            (Op::Conv(conv) | Op::ConvAct { conv, .. }, Packing::Transposed) => is_pointwise(conv),
            (Op::QWeightMatMul, Packing::Codes) => true,
            _ => false,
        };
        op_reads
            && !skip(node)
            && node.input(1) == Some(v)
            && node.inputs().iter().filter(|&&i| i == Some(v)).count() == 1
    };
    ((value.dtype == DType::F16 || codes)
        && value.layout == Layout::Stored
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
