//! Typed operators of the IR.

use super::Graph;
use crate::model::onnx_proto::{AttributeProto, NodeProto, TensorProto};

/// ONNX tensor element type (`TensorProto.DataType`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DType {
    #[default]
    Undefined,
    F32,
    U8,
    I8,
    U16,
    I16,
    I32,
    I64,
    String,
    Bool,
    F16,
    F64,
    U32,
    U64,
    BF16,
    /// Any other `TensorProto.DataType` (complex, float8, int4, ...).
    Other(i32),
}

impl DType {
    pub fn from_onnx(data_type: i32) -> Self {
        match data_type {
            0 => Self::Undefined,
            1 => Self::F32,
            2 => Self::U8,
            3 => Self::I8,
            4 => Self::U16,
            5 => Self::I16,
            6 => Self::I32,
            7 => Self::I64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::F16,
            11 => Self::F64,
            12 => Self::U32,
            13 => Self::U64,
            16 => Self::BF16,
            other => Self::Other(other),
        }
    }

    pub fn to_onnx(self) -> i32 {
        match self {
            Self::Undefined => 0,
            Self::F32 => 1,
            Self::U8 => 2,
            Self::I8 => 3,
            Self::U16 => 4,
            Self::I16 => 5,
            Self::I32 => 6,
            Self::I64 => 7,
            Self::String => 8,
            Self::Bool => 9,
            Self::F16 => 10,
            Self::F64 => 11,
            Self::U32 => 12,
            Self::U64 => 13,
            Self::BF16 => 16,
            Self::Other(other) => other,
        }
    }
}

/// Attributes shared by `QuantizeLinear`, `DequantizeLinear` and the fused
/// `FakeQuantizeLinear`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantAttrs {
    pub axis: i64,
    pub block_size: i64,
    /// Explicit `output_dtype` (opset 21+). When absent, `QuantizeLinear`
    /// takes its type from the zero point, or defaults to uint8.
    pub output_dtype: Option<DType>,
}

impl QuantAttrs {
    fn parse(node: &NodeProto) -> Self {
        Self {
            axis: attr_i(node, "axis").unwrap_or(1),
            block_size: attr_i(node, "block_size").unwrap_or(0),
            output_dtype: attr_i(node, "output_dtype").map(|t| DType::from_onnx(t as i32)),
        }
    }

    fn write(&self, attrs: &mut Vec<AttributeProto>) {
        set_attr_i(attrs, "axis", self.axis, 1);
        set_attr_i(attrs, "block_size", self.block_size, 0);
        set_attr_opt_i(
            attrs,
            "output_dtype",
            self.output_dtype.map(|t| t.to_onnx() as i64),
        );
    }
}

/// Attributes of `Conv`, `ConvInteger` and `ConvTranspose`, also carried by
/// the fused [`Op::ConvAct`]. Each is `None` when the node omits it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConvAttrs {
    pub auto_pad: Option<String>,
    pub dilations: Option<Vec<i64>>,
    pub group: Option<i64>,
    pub kernel_shape: Option<Vec<i64>>,
    pub pads: Option<Vec<i64>>,
    pub strides: Option<Vec<i64>>,
}

impl ConvAttrs {
    fn parse(node: &NodeProto) -> Self {
        Self {
            auto_pad: attr_s(node, "auto_pad"),
            dilations: attr_ints(node, "dilations"),
            group: attr_i(node, "group"),
            kernel_shape: attr_ints(node, "kernel_shape"),
            pads: attr_ints(node, "pads"),
            strides: attr_ints(node, "strides"),
        }
    }

    fn write(self, attrs: &mut Vec<AttributeProto>) {
        set_attr_opt_s(attrs, "auto_pad", self.auto_pad);
        set_attr_ints(attrs, "dilations", self.dilations);
        set_attr_opt_i(attrs, "group", self.group);
        set_attr_ints(attrs, "kernel_shape", self.kernel_shape);
        set_attr_ints(attrs, "pads", self.pads);
        set_attr_ints(attrs, "strides", self.strides);
    }
}

/// Attributes shared by `MaxPool` and `AveragePool`. Each but `ceil_mode` is
/// `None` when the node omits it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoolAttrs {
    pub auto_pad: Option<String>,
    pub ceil_mode: bool,
    pub dilations: Option<Vec<i64>>,
    pub kernel_shape: Option<Vec<i64>>,
    pub pads: Option<Vec<i64>>,
    pub strides: Option<Vec<i64>>,
}

impl PoolAttrs {
    fn parse(node: &NodeProto) -> Self {
        Self {
            auto_pad: attr_s(node, "auto_pad"),
            ceil_mode: attr_i(node, "ceil_mode").unwrap_or(0) != 0,
            dilations: attr_ints(node, "dilations"),
            kernel_shape: attr_ints(node, "kernel_shape"),
            pads: attr_ints(node, "pads"),
            strides: attr_ints(node, "strides"),
        }
    }

    fn write(self, attrs: &mut Vec<AttributeProto>) {
        set_attr_opt_s(attrs, "auto_pad", self.auto_pad);
        set_attr_i(attrs, "ceil_mode", self.ceil_mode as i64, 0);
        set_attr_ints(attrs, "dilations", self.dilations);
        set_attr_ints(attrs, "kernel_shape", self.kernel_shape);
        set_attr_ints(attrs, "pads", self.pads);
        set_attr_ints(attrs, "strides", self.strides);
    }
}

/// Activation fused into a preceding operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    /// `x * sigmoid(x)`.
    Silu,
}

impl Activation {
    pub fn name(self) -> &'static str {
        match self {
            Activation::Relu => "Relu",
            Activation::Silu => "Silu",
        }
    }
}

/// One dimension of a shape known at compile time up to dimensions of the
/// node's inputs: the target of [`Op::ReshapeTo`], [`Op::ExpandTo`] and
/// [`Op::ConstantOfShapeTo`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetDim {
    Fixed(i64),
    /// Dimension `axis` of input `input`.
    Of {
        input: usize,
        axis: usize,
    },
    /// Whatever the element count leaves; only in [`Op::ReshapeTo`], at
    /// most once.
    Infer,
}

/// An enum of operators sharing one [`Op`] variant, each named by its
/// `op_type`.
macro_rules! op_kinds {
    ($(#[$meta:meta])* $name:ident { $($kind:ident = $op_type:literal,)* }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name {
            $($kind,)*
        }

        impl $name {
            fn from_op_type(op_type: &str) -> Option<Self> {
                match op_type {
                    $($op_type => Some(Self::$kind),)*
                    _ => None,
                }
            }

            pub fn op_type(self) -> &'static str {
                match self {
                    $(Self::$kind => $op_type,)*
                }
            }
        }
    };
}

op_kinds! {
    /// Reduction of a `Reduce*` operator.
    Reduction {
        L2 = "ReduceL2",
        Max = "ReduceMax",
        Mean = "ReduceMean",
        Prod = "ReduceProd",
        Sum = "ReduceSum",
    }
}

op_kinds! {
    /// Elementwise operator of one operand, without attributes.
    Unary {
        Ceil = "Ceil",
        Cos = "Cos",
        Erf = "Erf",
        Exp = "Exp",
        Floor = "Floor",
        Log = "Log",
        Neg = "Neg",
        Not = "Not",
        Reciprocal = "Reciprocal",
        Relu = "Relu",
        Round = "Round",
        Sigmoid = "Sigmoid",
        Sin = "Sin",
        Softplus = "Softplus",
        Sqrt = "Sqrt",
        Tanh = "Tanh",
    }
}

op_kinds! {
    /// Elementwise operator of two broadcast operands, without attributes.
    /// `Mod`'s `fmod` attribute stays in the proto.
    Binary {
        Add = "Add",
        And = "And",
        Div = "Div",
        Equal = "Equal",
        Greater = "Greater",
        GreaterOrEqual = "GreaterOrEqual",
        Less = "Less",
        LessOrEqual = "LessOrEqual",
        Mod = "Mod",
        Mul = "Mul",
        Or = "Or",
        Pow = "Pow",
        PRelu = "PRelu",
        Sub = "Sub",
        Xor = "Xor",
    }
}

impl Binary {
    /// Whether the operator yields booleans.
    pub fn is_predicate(self) -> bool {
        matches!(
            self,
            Self::And
                | Self::Equal
                | Self::Greater
                | Self::GreaterOrEqual
                | Self::Less
                | Self::LessOrEqual
                | Self::Or
                | Self::Xor
        )
    }
}

/// An operator together with its parsed attributes.
///
/// A typed variant owns the attributes it models: they are parsed once when
/// the graph is built and written back to the node's [`NodeProto`] when it is
/// lowered. Attributes a variant does not model, and every attribute of an
/// [`Op::Onnx`] node, stay in the proto untouched.
#[derive(Clone, Debug)]
pub enum Op {
    // Standard ONNX operators.
    ArgMax {
        axis: i64,
        keepdims: bool,
    },
    AveragePool {
        pool: PoolAttrs,
        count_include_pad: bool,
    },
    /// `momentum` and `training_mode` stay in the proto.
    BatchNormalization {
        epsilon: f32,
    },
    Binary(Binary),
    Cast {
        to: DType,
    },
    /// Opset 11+ form, bounds are operands. The opset 6 attributes stay in
    /// the proto.
    Clip,
    Concat {
        axis: i64,
    },
    /// Only the `value` form; `value_float`, `value_ints` etc. stay
    /// [`Op::Onnx`].
    Constant {
        value: TensorProto,
    },
    ConstantOfShape {
        /// Fill value; zero f32 when absent.
        value: Option<TensorProto>,
    },
    Conv(ConvAttrs),
    ConvInteger(ConvAttrs),
    /// `output_padding` and `output_shape` stay in the proto.
    ConvTranspose(ConvAttrs),
    CumSum {
        exclusive: bool,
        reverse: bool,
    },
    DynamicQuantizeLinear,
    Einsum {
        equation: String,
    },
    Expand,
    Flatten {
        axis: i64,
    },
    Gather {
        axis: i64,
    },
    GatherElements {
        axis: i64,
    },
    Gelu {
        /// `"none"` (the default, exact erf form) or `"tanh"`.
        approximate: Option<String>,
    },
    Gemm {
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    },
    GlobalAveragePool,
    /// The other attributes (`hidden_size`, `direction`, ...) stay in the
    /// proto.
    Gru {
        linear_before_reset: bool,
    },
    HardSigmoid {
        alpha: f32,
        beta: f32,
    },
    Identity,
    LayerNormalization {
        axis: i64,
        epsilon: f32,
    },
    LeakyRelu {
        alpha: f32,
    },
    /// Softmax in log space.
    LogSoftmax {
        /// Absent means 1 before opset 13 and -1 from opset 13.
        axis: Option<i64>,
    },
    /// Its attributes (`hidden_size`, `direction`, ...) stay in the proto.
    Lstm,
    MatMul,
    MatMulInteger,
    /// Elementwise maximum of any number of operands.
    Max,
    /// `storage_order` stays in the proto.
    MaxPool(PoolAttrs),
    Pad {
        /// `"constant"` (the default), `"reflect"`, `"edge"` or `"wrap"`.
        mode: Option<String>,
    },
    QuantizeLinear(QuantAttrs),
    DequantizeLinear(QuantAttrs),
    Range,
    /// `ReduceL2`, `ReduceMax`, `ReduceMean`, `ReduceProd` and `ReduceSum`.
    /// `noop_with_empty_axes` stays in the proto.
    Reduce {
        reduction: Reduction,
        /// Attribute up to opset 12 for `ReduceSum` and 17 for the others;
        /// later opsets take the axes as an operand.
        axes: Option<Vec<i64>>,
        keepdims: bool,
    },
    /// `allowzero` stays in the proto.
    Reshape,
    /// The other attributes (`cubic_coeff_a`, `antialias`, ...) stay in the
    /// proto.
    Resize {
        /// `"half_pixel"` when absent.
        coordinate_transformation_mode: Option<String>,
        /// `"nearest"` when absent.
        mode: Option<String>,
        /// `"round_prefer_floor"` when absent.
        nearest_mode: Option<String>,
    },
    Shape {
        start: Option<i64>,
        end: Option<i64>,
    },
    Size,
    /// Opset 10+ form, all parameters are operands. The opset 1 attributes
    /// stay in the proto.
    Slice,
    Softmax {
        /// Absent means 1 before opset 13 and -1 from opset 13.
        axis: Option<i64>,
    },
    /// `num_outputs` (opset 18) stays in the proto.
    Split {
        axis: i64,
        /// Opset 1-12 attribute; later opsets take the sizes as an operand.
        split: Option<Vec<i64>>,
    },
    Squeeze {
        /// Opset 1-12 attribute; later opsets take the axes as an operand.
        axes: Option<Vec<i64>>,
    },
    /// `onesided` stays in the proto.
    Stft,
    Tile,
    TopK {
        axis: i64,
        largest: bool,
        sorted: bool,
    },
    Transpose {
        /// Absent means reversing the axes.
        perm: Option<Vec<i64>>,
    },
    Unary(Unary),
    Unsqueeze {
        /// Opset 1-12 attribute; later opsets take the axes as an operand.
        axes: Option<Vec<i64>>,
    },
    Where,
    If {
        then_branch: Box<Graph>,
        else_branch: Box<Graph>,
    },
    Loop {
        body: Box<Graph>,
    },

    // lele operators, created by passes.
    /// `QuantizeLinear` → `DequantizeLinear` round trip in one step. Inputs
    /// are those of the `QuantizeLinear`.
    FakeQuantizeLinear(QuantAttrs),
    /// `MatMul` of a uint8-quantized activation against a symmetric int8
    /// weight. Inputs: `[a, a_scale, a_zero_point, w, w_scale]`.
    QLinearMatMulI8,
    /// `MatMul` of two uint8-quantized activations (attention's scores and
    /// weighted values in static-QDQ models). Inputs:
    /// `[a, a_scale, a_zero_point, b, b_scale, b_zero_point]`.
    QLinearMatMulAct,
    /// `Conv` followed by an activation. Inputs are those of the `Conv`.
    ConvAct {
        conv: ConvAttrs,
        act: Activation,
    },
    /// `x * sigmoid(x)`.
    Silu,
    /// Snake activation, `x + beta * sin(alpha * x)^2` with one `alpha` and
    /// `beta` per channel (the second to last axis of `x`). Inputs:
    /// `[x, alpha, beta]`.
    Snake,
    /// `x / (ReduceL2(x, last axis) * scale + epsilon) * gamma` (ScaleNorm).
    /// Inputs: `[x, gamma]`.
    ScaleNorm {
        scale: f32,
        epsilon: f32,
    },
    /// SimAM attention over the planes of the last `axes` axes:
    /// `x * sigmoid(d / (scale * (ReduceSum(d) / n + lambda)) + shift)` with
    /// `d = (x - ReduceMean(x))^2`. Inputs: `[x, n]`, `n` a scalar.
    SimAm {
        axes: usize,
        lambda: f32,
        scale: f32,
        shift: f32,
    },
    /// `MatMul` + `Add`. Inputs: `[a, b, bias]`.
    Linear,
    /// Dynamically quantized linear layer: `DynamicQuantizeLinear` of the
    /// activation, `MatMulInteger` against a quantized weight, rescale and
    /// bias, optionally ReLU. Inputs:
    /// `[x, weight, weight_scale, weight_zero_point, bias]`.
    QLinearDynamic {
        relu: bool,
    },
    /// `Concat([weight, ConstantOfShape(shape, value)], axis = 0)`. Inputs:
    /// `[shape, weight]`.
    EmbeddingConcat {
        value: Option<TensorProto>,
    },
    /// `Reshape` to a target known at compile time up to dimensions of the
    /// inputs. Inputs: `[x, ...]`, then the tensors other than `x` whose
    /// dimensions the target takes.
    ReshapeTo {
        dims: Vec<TargetDim>,
    },
    /// `Expand` to a target known at compile time up to dimensions of the
    /// inputs. Inputs as in [`Op::ReshapeTo`].
    ExpandTo {
        dims: Vec<TargetDim>,
    },
    /// `ConstantOfShape` of a shape known at compile time up to dimensions
    /// of the inputs, which are the tensors it takes them from.
    ConstantOfShapeTo {
        value: Option<TensorProto>,
        dims: Vec<TargetDim>,
    },

    /// Any operator without a typed variant yet. Its graph attributes are
    /// built into `graphs` so passes see into them; every other attribute
    /// stays in the proto.
    Onnx {
        op_type: String,
        domain: String,
        graphs: Vec<(String, Graph)>,
    },
}

impl Op {
    /// Parses `node`'s operator. `subgraph` hands over the graph attribute of
    /// the given name, already built. Tensor attributes a variant takes are
    /// moved out of `node`.
    pub(super) fn parse(
        node: &mut NodeProto,
        mut subgraph: impl FnMut(&str) -> Option<Graph>,
    ) -> Self {
        let fallback = |node: &NodeProto| Op::Onnx {
            op_type: node.op_type.clone(),
            domain: node.domain.clone(),
            graphs: Vec::new(),
        };
        if !(node.domain.is_empty() || node.domain == "ai.onnx") {
            return fallback(node);
        }
        if let Some(reduction) = Reduction::from_op_type(&node.op_type) {
            return Op::Reduce {
                reduction,
                axes: attr_ints(node, "axes"),
                keepdims: attr_i(node, "keepdims").unwrap_or(1) != 0,
            };
        }
        if let Some(unary) = Unary::from_op_type(&node.op_type) {
            return Op::Unary(unary);
        }
        if let Some(binary) = Binary::from_op_type(&node.op_type) {
            return Op::Binary(binary);
        }
        match node.op_type.as_str() {
            "Clip" => Op::Clip,
            "DynamicQuantizeLinear" => Op::DynamicQuantizeLinear,
            "Expand" => Op::Expand,
            "GlobalAveragePool" => Op::GlobalAveragePool,
            "Identity" => Op::Identity,
            "LSTM" => Op::Lstm,
            "MatMulInteger" => Op::MatMulInteger,
            "Max" => Op::Max,
            "Range" => Op::Range,
            "Reshape" => Op::Reshape,
            "Size" => Op::Size,
            "STFT" => Op::Stft,
            "Tile" => Op::Tile,
            "Where" => Op::Where,
            "ArgMax" => Op::ArgMax {
                axis: attr_i(node, "axis").unwrap_or(0),
                keepdims: attr_i(node, "keepdims").unwrap_or(1) != 0,
            },
            "AveragePool" => Op::AveragePool {
                pool: PoolAttrs::parse(node),
                count_include_pad: attr_i(node, "count_include_pad").unwrap_or(0) != 0,
            },
            "BatchNormalization" => Op::BatchNormalization {
                epsilon: attr_f(node, "epsilon").unwrap_or(1e-5),
            },
            "Cast" => match attr_i(node, "to") {
                Some(to) => Op::Cast {
                    to: DType::from_onnx(to as i32),
                },
                None => fallback(node),
            },
            "Concat" => Op::Concat {
                axis: attr_i(node, "axis").unwrap_or(0),
            },
            "Constant" => match take_attr_t(node, "value") {
                Some(value) => Op::Constant { value },
                None => fallback(node),
            },
            "ConstantOfShape" => Op::ConstantOfShape {
                value: take_attr_t(node, "value"),
            },
            "Conv" => Op::Conv(ConvAttrs::parse(node)),
            "ConvInteger" => Op::ConvInteger(ConvAttrs::parse(node)),
            "ConvTranspose" => Op::ConvTranspose(ConvAttrs::parse(node)),
            "CumSum" => Op::CumSum {
                exclusive: attr_i(node, "exclusive").unwrap_or(0) != 0,
                reverse: attr_i(node, "reverse").unwrap_or(0) != 0,
            },
            "Einsum" => match attr_s(node, "equation") {
                Some(equation) => Op::Einsum { equation },
                None => fallback(node),
            },
            "Flatten" => Op::Flatten {
                axis: attr_i(node, "axis").unwrap_or(1),
            },
            "Gather" => Op::Gather {
                axis: attr_i(node, "axis").unwrap_or(0),
            },
            "GatherElements" => Op::GatherElements {
                axis: attr_i(node, "axis").unwrap_or(0),
            },
            "Gelu" => Op::Gelu {
                approximate: attr_s(node, "approximate"),
            },
            "Gemm" => Op::Gemm {
                alpha: attr_f(node, "alpha").unwrap_or(1.0),
                beta: attr_f(node, "beta").unwrap_or(1.0),
                trans_a: attr_i(node, "transA").unwrap_or(0) != 0,
                trans_b: attr_i(node, "transB").unwrap_or(0) != 0,
            },
            "GRU" => Op::Gru {
                linear_before_reset: attr_i(node, "linear_before_reset").unwrap_or(0) != 0,
            },
            "HardSigmoid" => Op::HardSigmoid {
                alpha: attr_f(node, "alpha").unwrap_or(0.2),
                beta: attr_f(node, "beta").unwrap_or(0.5),
            },
            "LayerNormalization" => Op::LayerNormalization {
                axis: attr_i(node, "axis").unwrap_or(-1),
                epsilon: attr_f(node, "epsilon").unwrap_or(1e-5),
            },
            "LeakyRelu" => Op::LeakyRelu {
                alpha: attr_f(node, "alpha").unwrap_or(0.01),
            },
            "LogSoftmax" => Op::LogSoftmax {
                axis: attr_i(node, "axis"),
            },
            "MatMul" => Op::MatMul,
            "MaxPool" => Op::MaxPool(PoolAttrs::parse(node)),
            "Pad" => Op::Pad {
                mode: attr_s(node, "mode"),
            },
            "QuantizeLinear" => Op::QuantizeLinear(QuantAttrs::parse(node)),
            "DequantizeLinear" => Op::DequantizeLinear(QuantAttrs::parse(node)),
            "Shape" => Op::Shape {
                start: attr_i(node, "start"),
                end: attr_i(node, "end"),
            },
            "Resize" => Op::Resize {
                coordinate_transformation_mode: attr_s(node, "coordinate_transformation_mode"),
                mode: attr_s(node, "mode"),
                nearest_mode: attr_s(node, "nearest_mode"),
            },
            "Slice" => Op::Slice,
            "Softmax" => Op::Softmax {
                axis: attr_i(node, "axis"),
            },
            "Split" => Op::Split {
                axis: attr_i(node, "axis").unwrap_or(0),
                split: attr_ints(node, "split"),
            },
            "Squeeze" => Op::Squeeze {
                axes: attr_ints(node, "axes"),
            },
            "TopK" => Op::TopK {
                axis: attr_i(node, "axis").unwrap_or(-1),
                largest: attr_i(node, "largest").unwrap_or(1) != 0,
                sorted: attr_i(node, "sorted").unwrap_or(1) != 0,
            },
            "Transpose" => Op::Transpose {
                perm: attr_ints(node, "perm"),
            },
            "Unsqueeze" => Op::Unsqueeze {
                axes: attr_ints(node, "axes"),
            },
            "If" => match (subgraph("then_branch"), subgraph("else_branch")) {
                (Some(t), Some(e)) => Op::If {
                    then_branch: Box::new(t),
                    else_branch: Box::new(e),
                },
                _ => fallback(node),
            },
            "Loop" => match subgraph("body") {
                Some(body) => Op::Loop {
                    body: Box::new(body),
                },
                None => fallback(node),
            },
            _ => fallback(node),
        }
    }

    /// The `op_type` the node lowers to.
    pub fn op_type(&self) -> &str {
        match self {
            Op::ArgMax { .. } => "ArgMax",
            Op::AveragePool { .. } => "AveragePool",
            Op::BatchNormalization { .. } => "BatchNormalization",
            Op::Binary(binary) => binary.op_type(),
            Op::Cast { .. } => "Cast",
            Op::Clip => "Clip",
            Op::DynamicQuantizeLinear => "DynamicQuantizeLinear",
            Op::Expand => "Expand",
            Op::GlobalAveragePool => "GlobalAveragePool",
            Op::Identity => "Identity",
            Op::Lstm => "LSTM",
            Op::MatMulInteger => "MatMulInteger",
            Op::Max => "Max",
            Op::Range => "Range",
            Op::Reshape => "Reshape",
            Op::Size => "Size",
            Op::Stft => "STFT",
            Op::Tile => "Tile",
            Op::Unary(unary) => unary.op_type(),
            Op::Where => "Where",
            Op::Concat { .. } => "Concat",
            Op::Constant { .. } => "Constant",
            Op::ConstantOfShape { .. } => "ConstantOfShape",
            Op::Conv(_) => "Conv",
            Op::ConvInteger(_) => "ConvInteger",
            Op::ConvTranspose(_) => "ConvTranspose",
            Op::CumSum { .. } => "CumSum",
            Op::Einsum { .. } => "Einsum",
            Op::Flatten { .. } => "Flatten",
            Op::Gather { .. } => "Gather",
            Op::GatherElements { .. } => "GatherElements",
            Op::Gelu { .. } => "Gelu",
            Op::Gemm { .. } => "Gemm",
            Op::Gru { .. } => "GRU",
            Op::HardSigmoid { .. } => "HardSigmoid",
            Op::LayerNormalization { .. } => "LayerNormalization",
            Op::LeakyRelu { .. } => "LeakyRelu",
            Op::LogSoftmax { .. } => "LogSoftmax",
            Op::MatMul => "MatMul",
            Op::MaxPool(_) => "MaxPool",
            Op::Pad { .. } => "Pad",
            Op::QuantizeLinear(_) => "QuantizeLinear",
            Op::DequantizeLinear(_) => "DequantizeLinear",
            Op::Reduce { reduction, .. } => reduction.op_type(),
            Op::Resize { .. } => "Resize",
            Op::Shape { .. } => "Shape",
            Op::Slice => "Slice",
            Op::Softmax { .. } => "Softmax",
            Op::Split { .. } => "Split",
            Op::Squeeze { .. } => "Squeeze",
            Op::TopK { .. } => "TopK",
            Op::Transpose { .. } => "Transpose",
            Op::Unsqueeze { .. } => "Unsqueeze",
            Op::If { .. } => "If",
            Op::Loop { .. } => "Loop",
            Op::FakeQuantizeLinear(_) => "FakeQuantizeLinear",
            Op::QLinearMatMulI8 => "QLinearMatMulI8",
            Op::QLinearMatMulAct => "QLinearMatMulAct",
            Op::ConvAct { .. } => "ConvAct",
            Op::Silu => "Silu",
            Op::Snake => "Snake",
            Op::ScaleNorm { .. } => "ScaleNorm",
            Op::SimAm { .. } => "SimAm",
            Op::Linear => "Linear",
            Op::QLinearDynamic { .. } => "QLinearDynamic",
            Op::EmbeddingConcat { .. } => "EmbeddingConcat",
            Op::ReshapeTo { .. } => "ReshapeTo",
            Op::ExpandTo { .. } => "ExpandTo",
            Op::ConstantOfShapeTo { .. } => "ConstantOfShapeTo",
            Op::Onnx { op_type, .. } => op_type,
        }
    }

    /// Subgraphs owned by this operator.
    pub fn subgraphs(&self) -> Vec<&Graph> {
        match self {
            Op::If {
                then_branch,
                else_branch,
            } => vec![then_branch, else_branch],
            Op::Loop { body } => vec![body],
            Op::Onnx { graphs, .. } => graphs.iter().map(|(_, g)| g).collect(),
            _ => Vec::new(),
        }
    }

    /// Subgraphs owned by this operator, with the name of their attribute.
    pub fn named_subgraphs(&self) -> Vec<(&str, &Graph)> {
        match self {
            Op::If {
                then_branch,
                else_branch,
            } => vec![("then_branch", then_branch), ("else_branch", else_branch)],
            Op::Loop { body } => vec![("body", body)],
            Op::Onnx { graphs, .. } => graphs.iter().map(|(n, g)| (n.as_str(), g)).collect(),
            _ => Vec::new(),
        }
    }

    pub fn subgraphs_mut(&mut self) -> Vec<&mut Graph> {
        match self {
            Op::If {
                then_branch,
                else_branch,
            } => vec![then_branch, else_branch],
            Op::Loop { body } => vec![body],
            Op::Onnx { graphs, .. } => graphs.iter_mut().map(|(_, g)| g).collect(),
            _ => Vec::new(),
        }
    }

    /// Writes the attributes this operator models into `attrs`, lowering its
    /// subgraphs.
    pub(super) fn into_attrs(self, attrs: &mut Vec<AttributeProto>) {
        match self {
            Op::Cast { to } => set_attr_opt_i(attrs, "to", Some(to.to_onnx() as i64)),
            Op::Concat { axis } => set_attr_i(attrs, "axis", axis, 0),
            Op::Constant { value } => set_attr_t(attrs, "value", Some(value)),
            Op::ConstantOfShape { value } | Op::EmbeddingConcat { value } => {
                set_attr_t(attrs, "value", value)
            }
            Op::Conv(conv) => conv.write(attrs),
            Op::ReshapeTo { dims } | Op::ExpandTo { dims } => write_target(attrs, &dims),
            Op::ConstantOfShapeTo { value, dims } => {
                set_attr_t(attrs, "value", value);
                write_target(attrs, &dims);
            }
            Op::ConvAct { conv, act } => {
                conv.write(attrs);
                set_attr(attrs, "activation", 3 /* STRING */, |a| {
                    a.s = act.name().as_bytes().to_vec()
                });
            }
            Op::ArgMax { axis, keepdims } => {
                set_attr_i(attrs, "axis", axis, 0);
                set_attr_i(attrs, "keepdims", keepdims as i64, 1);
            }
            Op::AveragePool {
                pool,
                count_include_pad,
            } => {
                pool.write(attrs);
                set_attr_i(attrs, "count_include_pad", count_include_pad as i64, 0);
            }
            Op::BatchNormalization { epsilon } => set_attr_f(attrs, "epsilon", epsilon, 1e-5),
            Op::ConvInteger(conv) | Op::ConvTranspose(conv) => conv.write(attrs),
            Op::CumSum { exclusive, reverse } => {
                set_attr_i(attrs, "exclusive", exclusive as i64, 0);
                set_attr_i(attrs, "reverse", reverse as i64, 0);
            }
            Op::Einsum { equation } => set_attr_opt_s(attrs, "equation", Some(equation)),
            Op::Flatten { axis } => set_attr_i(attrs, "axis", axis, 1),
            Op::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
            } => {
                set_attr_f(attrs, "alpha", alpha, 1.0);
                set_attr_f(attrs, "beta", beta, 1.0);
                set_attr_i(attrs, "transA", trans_a as i64, 0);
                set_attr_i(attrs, "transB", trans_b as i64, 0);
            }
            Op::Gru {
                linear_before_reset,
            } => set_attr_i(attrs, "linear_before_reset", linear_before_reset as i64, 0),
            Op::MaxPool(pool) => pool.write(attrs),
            Op::Resize {
                coordinate_transformation_mode,
                mode,
                nearest_mode,
            } => {
                set_attr_opt_s(
                    attrs,
                    "coordinate_transformation_mode",
                    coordinate_transformation_mode,
                );
                set_attr_opt_s(attrs, "mode", mode);
                set_attr_opt_s(attrs, "nearest_mode", nearest_mode);
            }
            Op::Gather { axis } | Op::GatherElements { axis } => set_attr_i(attrs, "axis", axis, 0),
            Op::Gelu { approximate } => set_attr_opt_s(attrs, "approximate", approximate),
            Op::HardSigmoid { alpha, beta } => {
                set_attr_f(attrs, "alpha", alpha, 0.2);
                set_attr_f(attrs, "beta", beta, 0.5);
            }
            Op::LayerNormalization { axis, epsilon } => {
                set_attr_i(attrs, "axis", axis, -1);
                set_attr_f(attrs, "epsilon", epsilon, 1e-5);
            }
            Op::LeakyRelu { alpha } => set_attr_f(attrs, "alpha", alpha, 0.01),
            Op::Softmax { axis } | Op::LogSoftmax { axis } => set_attr_opt_i(attrs, "axis", axis),
            Op::Pad { mode } => set_attr_opt_s(attrs, "mode", mode),
            Op::Reduce { axes, keepdims, .. } => {
                set_attr_ints(attrs, "axes", axes);
                set_attr_i(attrs, "keepdims", keepdims as i64, 1);
            }
            Op::Split { axis, split } => {
                set_attr_i(attrs, "axis", axis, 0);
                set_attr_ints(attrs, "split", split);
            }
            Op::TopK {
                axis,
                largest,
                sorted,
            } => {
                set_attr_i(attrs, "axis", axis, -1);
                set_attr_i(attrs, "largest", largest as i64, 1);
                set_attr_i(attrs, "sorted", sorted as i64, 1);
            }
            Op::Transpose { perm } => set_attr_ints(attrs, "perm", perm),
            Op::QLinearDynamic { relu } => set_attr_i(attrs, "relu", relu as i64, 0),
            Op::ScaleNorm { scale, epsilon } => {
                set_attr_f(attrs, "scale", scale, 1.0);
                set_attr_f(attrs, "epsilon", epsilon, 0.0);
            }
            Op::SimAm {
                axes,
                lambda,
                scale,
                shift,
            } => {
                set_attr_i(attrs, "axes", axes as i64, 0);
                set_attr_f(attrs, "lambda", lambda, 0.0);
                set_attr_f(attrs, "scale", scale, 1.0);
                set_attr_f(attrs, "shift", shift, 0.0);
            }
            Op::QuantizeLinear(a) | Op::DequantizeLinear(a) | Op::FakeQuantizeLinear(a) => {
                a.write(attrs)
            }
            Op::Shape { start, end } => {
                set_attr_opt_i(attrs, "start", start);
                set_attr_opt_i(attrs, "end", end);
            }
            Op::Squeeze { axes } | Op::Unsqueeze { axes } => set_attr_ints(attrs, "axes", axes),
            Op::If {
                then_branch,
                else_branch,
            } => {
                set_attr_g(attrs, "then_branch", *then_branch);
                set_attr_g(attrs, "else_branch", *else_branch);
            }
            Op::Loop { body } => set_attr_g(attrs, "body", *body),
            Op::Onnx { graphs, .. } => {
                for (name, g) in graphs {
                    set_attr_g(attrs, &name, g);
                }
            }
            Op::Binary(_)
            | Op::Clip
            | Op::DynamicQuantizeLinear
            | Op::Expand
            | Op::GlobalAveragePool
            | Op::Identity
            | Op::Lstm
            | Op::MatMul
            | Op::MatMulInteger
            | Op::Max
            | Op::Range
            | Op::Reshape
            | Op::Size
            | Op::Slice
            | Op::Stft
            | Op::Tile
            | Op::Unary(_)
            | Op::Where
            | Op::QLinearMatMulI8
            | Op::QLinearMatMulAct
            | Op::Silu
            | Op::Snake
            | Op::Linear => {}
        }
    }
}

fn attr_i(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

fn attr_f(node: &NodeProto, name: &str) -> Option<f32> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.f)
}

fn attr_s(node: &NodeProto, name: &str) -> Option<String> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
}

fn attr_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

fn take_attr_t(node: &mut NodeProto, name: &str) -> Option<TensorProto> {
    node.attribute
        .iter_mut()
        .find(|a| a.name == name)
        .and_then(|a| a.t.take())
}

/// Sets attribute `name` to `value`, adding it if absent unless it would
/// only restate `default`, so that an unchanged node lowers to exactly the
/// proto it was parsed from.
fn set_attr_i(attrs: &mut Vec<AttributeProto>, name: &str, value: i64, default: i64) {
    match attrs.iter_mut().find(|a| a.name == name) {
        Some(a) => a.i = value,
        None if value != default => attrs.push(AttributeProto {
            name: name.to_string(),
            r#type: 2, // INT
            i: value,
            ..Default::default()
        }),
        None => {}
    }
}

/// [`set_attr_i`] for a float attribute.
fn set_attr_f(attrs: &mut Vec<AttributeProto>, name: &str, value: f32, default: f32) {
    match attrs.iter_mut().find(|a| a.name == name) {
        Some(a) => a.f = value,
        None if value != default => attrs.push(AttributeProto {
            name: name.to_string(),
            r#type: 1, // FLOAT
            f: value,
            ..Default::default()
        }),
        None => {}
    }
}

/// Sets attribute `name` to `value`, or removes it if `value` is `None`.
fn set_attr_opt_s(attrs: &mut Vec<AttributeProto>, name: &str, value: Option<String>) {
    match value {
        Some(s) => set_attr(attrs, name, 3 /* STRING */, |a| a.s = s.into_bytes()),
        None => attrs.retain(|a| a.name != name),
    }
}

/// Sets attribute `name` to `value`, or removes it if `value` is `None`.
fn set_attr_opt_i(attrs: &mut Vec<AttributeProto>, name: &str, value: Option<i64>) {
    match value {
        Some(v) => set_attr(attrs, name, 2 /* INT */, |a| a.i = v),
        None => attrs.retain(|a| a.name != name),
    }
}

fn set_attr_ints(attrs: &mut Vec<AttributeProto>, name: &str, value: Option<Vec<i64>>) {
    match value {
        Some(v) => set_attr(attrs, name, 7 /* INTS */, |a| a.ints = v),
        None => attrs.retain(|a| a.name != name),
    }
}

fn set_attr_t(attrs: &mut Vec<AttributeProto>, name: &str, value: Option<TensorProto>) {
    match value {
        Some(t) => set_attr(attrs, name, 4 /* TENSOR */, |a| a.t = Some(t)),
        None => attrs.retain(|a| a.name != name),
    }
}

/// Writes a [`TargetDim`] target: `shape` holds the fixed sizes, -1 to
/// infer and 0 where `dim_inputs` and `dim_axes` name the input dimension to
/// take.
fn write_target(attrs: &mut Vec<AttributeProto>, dims: &[TargetDim]) {
    let mut shape = Vec::new();
    let mut inputs = Vec::new();
    let mut axes = Vec::new();
    for d in dims {
        let (n, input, axis) = match *d {
            TargetDim::Fixed(n) => (n, -1, -1),
            TargetDim::Of { input, axis } => (0, input as i64, axis as i64),
            TargetDim::Infer => (-1, -1, -1),
        };
        shape.push(n);
        inputs.push(input);
        axes.push(axis);
    }
    set_attr_ints(attrs, "shape", Some(shape));
    set_attr_ints(attrs, "dim_inputs", Some(inputs));
    set_attr_ints(attrs, "dim_axes", Some(axes));
}

fn set_attr_g(attrs: &mut Vec<AttributeProto>, name: &str, graph: Graph) {
    let g = graph.into_onnx();
    set_attr(attrs, name, 5 /* GRAPH */, |a| a.g = Some(g));
}

/// Updates attribute `name` with `set`, adding it with type `ty` if absent.
fn set_attr(
    attrs: &mut Vec<AttributeProto>,
    name: &str,
    ty: i32,
    set: impl FnOnce(&mut AttributeProto),
) {
    match attrs.iter_mut().find(|a| a.name == name) {
        Some(a) => set(a),
        None => {
            let mut a = AttributeProto {
                name: name.to_string(),
                r#type: ty,
                ..Default::default()
            };
            set(&mut a);
            attrs.push(a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::onnx_proto::GraphProto;

    fn int(name: &str, i: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: 2,
            i,
            ..Default::default()
        }
    }

    fn float(name: &str, f: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: 1,
            f,
            ..Default::default()
        }
    }

    fn ints(name: &str, ints: &[i64]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: 7,
            ints: ints.to_vec(),
            ..Default::default()
        }
    }

    fn string(name: &str, s: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: 3,
            s: s.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    /// Typed operators lower to the node they were parsed from, whether it
    /// spells out its attributes, defaults included, or leaves them out.
    #[test]
    fn typed_ops_round_trip() {
        let cases: Vec<(&str, Vec<AttributeProto>)> = vec![
            ("ArgMax", vec![int("axis", 0), int("keepdims", 0)]),
            (
                "AveragePool",
                vec![
                    ints("kernel_shape", &[3, 3]),
                    ints("pads", &[1, 1, 1, 1]),
                    int("count_include_pad", 1),
                    int("ceil_mode", 0),
                ],
            ),
            ("BatchNormalization", vec![float("epsilon", 1e-3)]),
            (
                "ConvInteger",
                vec![int("group", 1), ints("strides", &[2, 2])],
            ),
            (
                "ConvTranspose",
                vec![ints("strides", &[2]), ints("output_padding", &[1])],
            ),
            ("CumSum", vec![int("exclusive", 1), int("reverse", 0)]),
            ("Einsum", vec![string("equation", "bs,d->bsd")]),
            ("Gemm", vec![float("alpha", 1.0), int("transB", 1)]),
            (
                "GRU",
                vec![int("hidden_size", 8), int("linear_before_reset", 1)],
            ),
            (
                "MaxPool",
                vec![
                    string("auto_pad", "SAME_UPPER"),
                    ints("kernel_shape", &[2, 2]),
                    int("ceil_mode", 1),
                    ints("dilations", &[1, 1]),
                ],
            ),
            (
                "Resize",
                vec![
                    string("coordinate_transformation_mode", "asymmetric"),
                    string("mode", "nearest"),
                    string("nearest_mode", "floor"),
                ],
            ),
            ("Flatten", vec![int("axis", 2)]),
            ("Gather", vec![int("axis", 1)]),
            ("GatherElements", vec![int("axis", 0)]),
            ("HardSigmoid", vec![float("alpha", 0.2), float("beta", 0.6)]),
            ("LeakyRelu", vec![float("alpha", 0.1)]),
            ("LogSoftmax", vec![int("axis", 1)]),
            ("Pad", vec![string("mode", "reflect")]),
            ("ReduceMean", vec![ints("axes", &[-1]), int("keepdims", 1)]),
            (
                "ReduceSum",
                vec![int("keepdims", 0), int("noop_with_empty_axes", 1)],
            ),
            ("Softmax", vec![int("axis", -1)]),
            ("Split", vec![int("axis", 1), ints("split", &[2, 3])]),
            (
                "TopK",
                vec![int("axis", 1), int("largest", 0), int("sorted", 1)],
            ),
            ("Transpose", vec![ints("perm", &[0, 2, 1])]),
        ];
        let mut nodes = Vec::new();
        for (op_type, attribute) in cases {
            nodes.push(NodeProto {
                op_type: op_type.to_string(),
                input: vec!["x".to_string()],
                output: vec![format!("{op_type}_set")],
                attribute,
                ..Default::default()
            });
            // An equation is required.
            if op_type != "Einsum" {
                nodes.push(NodeProto {
                    op_type: op_type.to_string(),
                    input: vec!["x".to_string()],
                    output: vec![format!("{op_type}_default")],
                    ..Default::default()
                });
            }
        }
        let g = Graph::from_onnx(GraphProto {
            node: nodes.clone(),
            ..Default::default()
        });
        for (_, node) in g.nodes() {
            assert!(
                !matches!(node.op, Op::Onnx { .. }),
                "{} is untyped",
                node.op.op_type()
            );
        }
        assert_eq!(g.into_onnx().node, nodes);
    }
}
