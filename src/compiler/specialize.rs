//! Specializing a model to the inputs it runs on: the shapes of inputs, the
//! values of named dimensions, and inputs whose values never change.

use super::generate::ElemType;
use super::ir::{DType, Dim, Graph};
use crate::model::onnx_proto::TensorProto;
use std::collections::{BTreeMap, HashMap};

/// The value of an input compiled into a model, its elements in order.
#[derive(Clone, Debug, PartialEq)]
pub enum InputValue {
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

impl From<i64> for InputValue {
    fn from(v: i64) -> Self {
        Self::Ints(vec![v])
    }
}

impl From<Vec<i64>> for InputValue {
    fn from(v: Vec<i64>) -> Self {
        Self::Ints(v)
    }
}

impl From<f64> for InputValue {
    fn from(v: f64) -> Self {
        Self::Floats(vec![v])
    }
}

impl From<Vec<f64>> for InputValue {
    fn from(v: Vec<f64>) -> Self {
        Self::Floats(v)
    }
}

impl InputValue {
    fn len(&self) -> usize {
        match self {
            Self::Ints(v) => v.len(),
            Self::Floats(v) => v.len(),
        }
    }

    /// A tensor of type `dtype` and shape `dims` holding these elements.
    fn tensor(&self, name: &str, dtype: DType, dims: &[i64]) -> Result<TensorProto, String> {
        let mut t = TensorProto {
            name: name.to_string(),
            data_type: dtype.to_onnx(),
            dims: dims.to_vec(),
            ..Default::default()
        };
        let floats = || -> Vec<f64> {
            match self {
                Self::Ints(v) => v.iter().map(|&x| x as f64).collect(),
                Self::Floats(v) => v.clone(),
            }
        };
        match (dtype, self) {
            (DType::F32, _) => t.float_data = floats().iter().map(|&x| x as f32).collect(),
            (DType::F64, _) => t.double_data = floats(),
            (DType::I64, Self::Ints(v)) => t.int64_data = v.clone(),
            (DType::I32, Self::Ints(v)) => {
                t.int32_data = v
                    .iter()
                    .map(|&x| i32::try_from(x).map_err(|_| format!("{x} is not an int32")))
                    .collect::<Result<_, _>>()?
            }
            (DType::Bool, Self::Ints(v)) if v.iter().all(|&x| x == 0 || x == 1) => {
                t.int32_data = v.iter().map(|&x| x as i32).collect()
            }
            _ => return Err(format!("{self:?} cannot be a {dtype:?} tensor")),
        }
        Ok(t)
    }

    /// The elements as a Rust slice literal of `ty`.
    fn literal(&self, ty: ElemType) -> String {
        let elems: Vec<String> = match (self, ty) {
            (Self::Ints(v), ElemType::I64) => v.iter().map(|x| format!("{x}i64")).collect(),
            (Self::Ints(v), ElemType::F32) => {
                v.iter().map(|&x| format!("{:?}f32", x as f32)).collect()
            }
            (Self::Floats(v), _) => v.iter().map(|&x| format!("{:?}f32", x as f32)).collect(),
        };
        format!("&[{}]", elems.join(", "))
    }
}

/// The inputs a model is compiled for, set with
/// [`Compiler::with_input_shape`](super::Compiler::with_input_shape),
/// [`Compiler::with_dim`](super::Compiler::with_dim) and
/// [`Compiler::with_input_value`](super::Compiler::with_input_value).
#[derive(Clone, Debug, Default)]
pub(crate) struct Specialization {
    pub shapes: BTreeMap<String, Vec<usize>>,
    pub dims: BTreeMap<String, usize>,
    pub values: BTreeMap<String, InputValue>,
}

/// An input of the generated `forward`, and what it checks of it.
pub(crate) struct Input {
    pub name: String,
    /// The type of an input compiled in, which the graph no longer has.
    pub fixed_type: Option<ElemType>,
    /// The dimensions the input must have, `None` for any, if any are fixed.
    pub shape: Option<Vec<Option<usize>>>,
    pub value: Option<InputValue>,
}

impl Input {
    /// The statements checking the input at the start of `forward`.
    pub fn checks(&self, model: &str, var: &str, ty: ElemType) -> Vec<String> {
        let mut checks = Vec::new();
        if let Some(shape) = &self.shape {
            let dims: Vec<String> = shape.iter().map(|d| format!("{d:?}")).collect();
            checks.push(format!(
                "lele::runtime::check_input_shape({model:?}, {:?}, &{var}.shape, &[{}]);",
                self.name,
                dims.join(", ")
            ));
        }
        if let Some(value) = &self.value {
            checks.push(format!(
                "lele::runtime::check_input_value({model:?}, {:?}, &{var}.data, {});",
                self.name,
                value.literal(ty)
            ));
        }
        checks
    }
}

impl Specialization {
    /// Specializes `g`: fixes the shapes and named dimensions of its inputs
    /// and turns the inputs of fixed value into initializers. Returns the
    /// inputs `forward` takes, those compiled in included, in order.
    pub fn apply(&self, g: &mut Graph) -> Result<Vec<Input>, String> {
        let original = g.inputs.clone();
        let find = |g: &Graph, name: &str| {
            original
                .iter()
                .copied()
                .find(|&v| g.value(v).name == name)
                .ok_or_else(|| format!("the model has no input {name}"))
        };
        let declares = |sym: &str| {
            original.iter().any(|&v| {
                (g.value(v).shape.iter().flatten()).any(|d| matches!(d, Dim::Sym(s) if s == sym))
            })
        };
        if let Some(sym) = self.dims.keys().find(|s| !declares(s)) {
            return Err(format!("no input of the model has a dimension named {sym}"));
        }
        let mut bound: HashMap<String, usize> = self.dims.clone().into_iter().collect();
        let mut bind = |sym: &str, n: usize, input: &str| match bound.insert(sym.to_string(), n) {
            Some(m) if m != n => Err(format!(
                "input {input} makes dimension {sym} {n}, where it is {m}"
            )),
            _ => Ok(()),
        };
        for (name, shape) in &self.shapes {
            let v = find(g, name)?;
            if let Some(declared) = &g.value(v).shape {
                if declared.len() != shape.len() {
                    return Err(format!(
                        "input {name} has {} dimensions, not {}",
                        declared.len(),
                        shape.len()
                    ));
                }
                for (d, &n) in declared.iter().zip(shape) {
                    match d {
                        Dim::Fixed(m) if *m != n as i64 => {
                            return Err(format!(
                                "input {name} has shape {declared:?}, not {shape:?}"
                            ));
                        }
                        Dim::Sym(s) => bind(s, n, name)?,
                        _ => {}
                    }
                }
            }
            g.value_mut(v).shape = Some(shape.iter().map(|&n| Dim::Fixed(n as i64)).collect());
        }
        let affected: Vec<bool> = original
            .iter()
            .map(|&v| {
                let value = g.value(v);
                self.shapes.contains_key(&value.name)
                    || self.values.contains_key(&value.name)
                    || value
                        .shape
                        .iter()
                        .flatten()
                        .any(|d| matches!(d, Dim::Sym(s) if bound.contains_key(s)))
            })
            .collect();
        bind_dims(g, &bound);

        let mut inputs: Vec<Input> = original
            .iter()
            .zip(affected)
            .map(|(&v, affected)| {
                let value = g.value(v);
                let shape = value.shape.as_ref().filter(|_| affected).map(|dims| {
                    dims.iter()
                        .map(|d| match d {
                            Dim::Fixed(n) => usize::try_from(*n).ok(),
                            _ => None,
                        })
                        .collect()
                });
                Input {
                    name: value.name.clone(),
                    fixed_type: None,
                    shape,
                    value: None,
                }
            })
            .collect();
        for (name, value) in &self.values {
            let v = find(g, name)?;
            let i = original.iter().position(|&o| o == v).unwrap();
            let dims = inputs[i]
                .shape
                .clone()
                .and_then(|s| {
                    s.into_iter()
                        .map(|d| d.map(|n| n as i64))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| format!("input {name} needs a fixed shape to be given a value"))?;
            let count: i64 = dims.iter().product();
            if value.len() as i64 != count {
                return Err(format!(
                    "input {name} of shape {dims:?} has {count} elements, not {}",
                    value.len()
                ));
            }
            let dtype = g.value(v).dtype;
            let tensor = value
                .tensor(name, dtype, &dims)
                .map_err(|e| format!("input {name}: {e}"))?;
            inputs[i].fixed_type = Some(ElemType::of(dtype));
            inputs[i].value = Some(value.clone());
            g.fix_input(v, tensor);
        }
        Ok(inputs)
    }
}

/// Replaces the symbolic dimensions `bound` gives a value by that value, in
/// the shapes of every value of `g` and its subgraphs.
fn bind_dims(g: &mut Graph, bound: &HashMap<String, usize>) {
    if bound.is_empty() {
        return;
    }
    for v in g.value_ids() {
        for d in g.value_mut(v).shape.iter_mut().flatten() {
            if let Dim::Sym(s) = d
                && let Some(&n) = bound.get(s.as_str())
            {
                *d = Dim::Fixed(n as i64);
            }
        }
    }
    for id in g.node_ids() {
        for sub in g.node_mut(id).op.subgraphs_mut() {
            bind_dims(sub, bound);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::Compiler;
    use crate::model::onnx_proto::{
        GraphProto, NodeProto, TensorShapeProto, TypeProto, ValueInfoProto, tensor_shape_proto,
        type_proto,
    };

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto {
            op_type: op.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn input(name: &str, dtype: DType, dims: &[&str]) -> ValueInfoProto {
        let dim = |d: &&str| tensor_shape_proto::Dimension {
            value: Some(match d.parse::<i64>() {
                Ok(n) => tensor_shape_proto::dimension::Value::DimValue(n),
                Err(_) => tensor_shape_proto::dimension::Value::DimParam(d.to_string()),
            }),
            ..Default::default()
        };
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: dtype.to_onnx(),
                    shape: Some(TensorShapeProto {
                        dim: dims.iter().map(dim).collect(),
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// `y = Relu(x)`, `y` plus a `[7, 1, 4]` constant, and the positions
    /// along the first dimension of `x`, for `x` of shape `[N, 4]`, and
    /// whether the sample rate `sr` is 16000.
    fn graph() -> GraphProto {
        let int = |name: &str, v: i64| TensorProto {
            name: name.to_string(),
            data_type: DType::I64.to_onnx(),
            int64_data: vec![v],
            ..Default::default()
        };
        let mut constant = node("Constant", &[], "c");
        constant.attribute = vec![crate::model::onnx_proto::AttributeProto {
            name: "value".to_string(),
            t: Some(TensorProto {
                data_type: DType::F32.to_onnx(),
                dims: vec![7, 1, 4],
                float_data: vec![0.5; 28],
                ..Default::default()
            }),
            ..Default::default()
        }];
        GraphProto {
            node: vec![
                node("Relu", &["x"], "y"),
                constant,
                node("Add", &["c", "y"], "z"),
                node("Shape", &["x"], "s"),
                node("Gather", &["s", "zero"], "n"),
                node("Range", &["zero", "n", "one"], "positions"),
                node("Equal", &["sr", "rate"], "is_16k"),
            ],
            initializer: vec![int("zero", 0), int("one", 1), int("rate", 16000)],
            input: vec![
                input("x", DType::F32, &["N", "4"]),
                input("sr", DType::I64, &[]),
            ],
            output: ["z", "positions", "is_16k"]
                .iter()
                .map(|o| ValueInfoProto {
                    name: o.to_string(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn code(c: Compiler) -> String {
        c.with_default_optimizations()
            .compile(&graph())
            .unwrap()
            .code
    }

    /// Fixing a dimension, by its name or by an input's shape, folds what
    /// depends on it, and `forward` checks the input.
    #[test]
    fn fixed_shapes_fold_and_are_checked() {
        let general = code(Compiler::new());
        assert!(general.contains("kernels::shape(&x)"), "{general}");
        assert!(!general.contains("check_input"), "{general}");
        let check =
            r#"lele::runtime::check_input_shape("Model", "x", &x.shape, &[Some(2), Some(4)]);"#;
        for c in [
            Compiler::new().with_dim("N", 2),
            Compiler::new().with_input_shape("x", &[2, 4]),
            Compiler::new()
                .with_input_shape("x", &[2, 4])
                .with_dim("N", 2),
        ] {
            let code = code(c);
            assert!(!code.contains("kernels::shape("), "{code}");
            assert!(code.contains(check), "{code}");
            assert!(!code.contains(r#""sr""#), "{code}");
        }
    }

    /// With every shape fixed, the workspace allocates each buffer when made,
    /// at the size of its largest tensor. A constant, read from the weights,
    /// takes none.
    #[test]
    fn fixed_shapes_size_the_workspace() {
        // Only the buffer of the scalars `n` and `is_16k` has a fixed size.
        let general = code(Compiler::new());
        assert_eq!(
            general.matches("Vec::with_capacity(").count(),
            1,
            "{general}"
        );
        assert!(general.contains("Vec::with_capacity(1),"), "{general}");
        assert!(general.contains("..Default::default()"), "{general}");
        assert!(!general.contains("BYTES"), "{general}");
        let fixed = code(
            Compiler::new()
                .with_dim("N", 2)
                .with_input_value("sr", 16000),
        );
        let reserved: Vec<&str> = fixed
            .lines()
            .filter(|l| l.contains("Vec::with_capacity("))
            .map(str::trim)
            .collect();
        // `y` (2 x 4) and `z` (7 x 2 x 4), in f32.
        assert_eq!(
            reserved,
            [
                "buf_0: Vec::with_capacity(8),",
                "buf_1: Vec::with_capacity(56),"
            ],
            "{fixed}"
        );
        assert!(fixed.contains("pub const BYTES: usize = 256;"), "{fixed}");
    }

    /// An input compiled in is folded where it is read, and `forward` still
    /// takes it, checking its value.
    #[test]
    fn input_values_fold_and_are_checked() {
        let general = code(Compiler::new());
        assert!(general.contains("equal_i64(&sr"), "{general}");
        let code = code(Compiler::new().with_input_value("sr", 16000));
        assert!(!code.contains("equal_i64("), "{code}");
        assert!(code.contains("sr: TensorView<'a, i64>"), "{code}");
        let check = r#"lele::runtime::check_input_value("Model", "sr", &sr.data, &[16000i64]);"#;
        assert!(code.contains(check), "{code}");
        assert!(
            code.contains(r#"check_input_shape("Model", "sr", &sr.shape, &[]);"#),
            "{code}"
        );
    }

    #[test]
    fn what_does_not_fit_the_model_is_an_error() {
        let error = |c: Compiler| c.compile(&graph()).err().map(|e| e.to_string());
        let cases = [
            (
                Compiler::new().with_input_shape("z", &[1]),
                "the model has no input z",
            ),
            (
                Compiler::new().with_input_shape("x", &[2]),
                "input x has 2 dimensions, not 1",
            ),
            (
                Compiler::new().with_input_shape("x", &[2, 5]),
                r#"input x has shape [Sym("N"), Fixed(4)], not [2, 5]"#,
            ),
            (
                Compiler::new()
                    .with_dim("N", 3)
                    .with_input_shape("x", &[2, 4]),
                "input x makes dimension N 2, where it is 3",
            ),
            (
                Compiler::new().with_dim("M", 1),
                "no input of the model has a dimension named M",
            ),
            (
                Compiler::new().with_input_value("x", vec![1, 2]),
                "input x needs a fixed shape to be given a value",
            ),
            (
                Compiler::new()
                    .with_dim("N", 1)
                    .with_input_value("x", vec![1, 2]),
                "input x of shape [1, 4] has 4 elements, not 2",
            ),
            (
                Compiler::new().with_input_value("sr", 0.5),
                "input sr: Floats([0.5]) cannot be a I64 tensor",
            ),
        ];
        for (c, want) in cases {
            assert_eq!(error(c).as_deref(), Some(want));
        }
    }
}
