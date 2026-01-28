use super::{sanitize_name, Pattern};
use crate::model::onnx_proto::NodeProto;

pub fn get_default_patterns() -> Vec<Pattern> {
    vec![
        Pattern {
            name: "LayerNorm".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 9 {
                    return None;
                }
                if nodes[0].op_type != "ReduceMean" {
                    return None;
                }
                if nodes[1].op_type != "Sub" {
                    return None;
                }
                if nodes[2].op_type != "Pow" {
                    return None;
                }
                if nodes[3].op_type != "ReduceMean" {
                    return None;
                }
                if nodes[4].op_type != "Add" {
                    return None;
                }
                if nodes[5].op_type != "Sqrt" {
                    return None;
                }
                if nodes[6].op_type != "Div" {
                    return None;
                }
                if nodes[7].op_type != "Mul" {
                    return None;
                }
                if nodes[8].op_type != "Add" {
                    return None;
                }
                let input = &nodes[0].input[0];
                let _mean = &nodes[0].output[0];
                if nodes[1].input[0] != *input && nodes[1].input[1] != *input {
                    return None;
                }
                let sub = &nodes[1].output[0];
                if nodes[2].input[0] != *sub {
                    return None;
                }
                let pow = &nodes[2].output[0];
                if nodes[3].input[0] != *pow {
                    return None;
                }
                let var = &nodes[3].output[0];
                if nodes[4].input[0] != *var {
                    return None;
                }
                let add_eps = &nodes[4].output[0];
                if nodes[5].input[0] != *add_eps {
                    return None;
                }
                let std = &nodes[5].output[0];
                if nodes[6].input[0] != *sub || nodes[6].input[1] != *std {
                    return None;
                }
                let norm = &nodes[6].output[0];
                if nodes[7].input[0] != *norm {
                    return None;
                }
                let scaled = &nodes[7].output[0];
                if nodes[8].input[0] != *scaled && nodes[8].input[1] != *scaled {
                    return None;
                }
                Some(9)
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let input = sanitize_name(&nodes[0].input[0]);
                let get_weight = |name: &str| -> String {
                    let s = sanitize_name(name);
                    if let Some((o, l, sh)) = weights.get(&s) {
                        format!("self.weight({}, {}, &{:?})", o, l, sh)
                    } else {
                        s
                    }
                };
                let _two = get_weight(&nodes[2].input[1]);
                let eps = get_weight(&nodes[4].input[1]);
                let scale = get_weight(&nodes[7].input[1]);
                let bias_name = if nodes[8].input[0] == nodes[7].output[0] {
                    &nodes[8].input[1]
                } else {
                    &nodes[8].input[0]
                };
                let bias = get_weight(bias_name);
                let output_name = sanitize_name(&nodes[8].output[0]);
                let tab = "    ".repeat(indent);
                let buf_expr = if let Some(alloc) = allocator {
                    if let Some(&idx) = alloc.tensor_to_buffer.get(&nodes[8].output[0]) {
                        format!("&mut ws.buf_{}", idx)
                    } else {
                        writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                        format!("&mut buf_{}", output_name)
                    }
                } else {
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                    format!("&mut buf_{}", output_name)
                };
                writeln!(
                    w,
                    "{}let {} = self.layer_norm(&{}, {}, {}, {}, {}, {});",
                    tab, output_name, input, scale, bias, eps, _two, buf_expr
                )?;
                Ok(())
            }),
        },
        Pattern {
            name: "Quantized Linear".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 6 {
                    return None;
                }
                if nodes[0].op_type != "DynamicQuantizeLinear" {
                    return None;
                }
                if nodes[1].op_type != "Mul" {
                    return None;
                }
                if nodes[2].op_type != "MatMulInteger" {
                    return None;
                }
                if nodes[3].op_type != "Cast" {
                    return None;
                }
                if nodes[4].op_type != "Mul" {
                    return None;
                }
                if nodes[5].op_type != "Add" {
                    return None;
                }
                let q = &nodes[0].output[0];
                let s = &nodes[0].output[1];
                let z = &nodes[0].output[2];
                if q.is_empty() || s.is_empty() || z.is_empty() {
                    return None;
                }
                if nodes[1].input[0] != *s && nodes[1].input.get(1) != Some(s) {
                    return None;
                }
                let combined_scale = &nodes[1].output[0];
                if nodes[2].input[0] != *q {
                    return None;
                }
                if nodes[2].input.len() < 3 || nodes[2].input[2] != *z {
                    return None;
                }
                let mm = &nodes[2].output[0];
                if nodes[3].input[0] != *mm {
                    return None;
                }
                let mm_cast = &nodes[3].output[0];
                if nodes[4].input[0] != *mm_cast && nodes[4].input.get(1) != Some(mm_cast) {
                    return None;
                }
                if nodes[4].input[0] != *combined_scale
                    && nodes[4].input.get(1) != Some(combined_scale)
                {
                    return None;
                }
                let dequant = &nodes[4].output[0];
                if nodes[5].input[0] != *dequant && nodes[5].input.get(1) != Some(dequant) {
                    return None;
                }
                Some(6)
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let input = sanitize_name(&nodes[0].input[0]);
                let get_weight = |name: &str| -> String {
                    let s = sanitize_name(name);
                    if let Some((o, l, sh)) = weights.get(&s) {
                        format!("self.weight({}, {}, &{:?})", o, l, sh)
                    } else {
                        s
                    }
                };
                let weight_int8 = get_weight(&nodes[2].input[1]);
                let s_name = &nodes[0].output[1];
                let ws_name = if nodes[1].input[0] == *s_name {
                    &nodes[1].input[1]
                } else {
                    &nodes[1].input[0]
                };
                let weight_scale = get_weight(ws_name);
                let weight_zero = get_weight(&nodes[2].input[3]);
                let dequant = &nodes[4].output[0];
                let bias_name = if nodes[5].input[0] == *dequant {
                    &nodes[5].input[1]
                } else {
                    &nodes[5].input[0]
                };
                let bias = get_weight(bias_name);
                let output_name = sanitize_name(&nodes[5].output[0]);
                let tab = "    ".repeat(indent);
                let buf_expr = if let Some(alloc) = allocator {
                    if let Some(&idx) = alloc.tensor_to_buffer.get(&nodes[5].output[0]) {
                        format!("&mut ws.buf_{}", idx)
                    } else {
                        writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                        format!("&mut buf_{}", output_name)
                    }
                } else {
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                    format!("&mut buf_{}", output_name)
                };
                writeln!(
                    w,
                    "{}let {} = self.linear_quantized(&{}, {}, {}, {}, {}, {});",
                    tab, output_name, input, weight_int8, weight_scale, weight_zero, bias, buf_expr
                )?;
                Ok(())
            }),
        },
        Pattern {
            name: "Embedding Concat".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 2 {
                    return None;
                }
                if nodes[0].op_type != "ConstantOfShape" {
                    return None;
                }
                if nodes[1].op_type != "Concat" {
                    return None;
                }
                if nodes[0].output.len() != 1 {
                    return None;
                }
                let cos_out = &nodes[0].output[0];
                if nodes[1].input.len() < 2 || &nodes[1].input[1] != cos_out {
                    return None;
                }
                Some(2)
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let cos_node = nodes[0];
                let concat_node = nodes[1];
                let tab = "    ".repeat(indent);
                let weight_name = sanitize_name(&concat_node.input[0]);
                let input_val = cos_node
                    .attribute
                    .iter()
                    .find(|a| a.name == "value")
                    .and_then(|a| a.t.as_ref())
                    .map(|t| {
                        if !t.float_data.is_empty() {
                            t.float_data[0]
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0);
                let shape_input = sanitize_name(&cos_node.input[0]);
                let shape_expr = if let Some((o, l, s)) = weights.get(&shape_input) {
                    format!("&self.weight({}, {}, &{:?})", o, l, s)
                } else {
                    format!("&{}", shape_input)
                };
                let weight_expr = if let Some((o, l, s)) = weights.get(&weight_name) {
                    format!("self.weight({}, {}, &{:?})", o, l, s)
                } else {
                    weight_name
                };
                let output_name = sanitize_name(&concat_node.output[0]);
                let buf_expr = if let Some(alloc) = allocator {
                    if let Some(&idx) = alloc.tensor_to_buffer.get(&concat_node.output[0]) {
                        format!("&mut ws.buf_{}", idx)
                    } else {
                        writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                        format!("&mut buf_{}", output_name)
                    }
                } else {
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                    format!("&mut buf_{}", output_name)
                };
                writeln!(
                    w,
                    "{}let {} = self.embedding_concat({}, {:.1}, {}, {});",
                    tab, output_name, shape_expr, input_val, weight_expr, buf_expr
                )?;
                Ok(())
            }),
        },
        Pattern {
            name: "Conv1d + Relu".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 2 {
                    return None;
                }
                let n0 = nodes[0];
                let n1 = nodes[1];
                if n0.op_type == "Conv"
                    && n1.op_type == "Relu"
                    && n0.output.first() == n1.input.first()
                {
                    return Some(2);
                }
                None
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let conv = nodes[0];
                let tab = "    ".repeat(indent);
                let input = &conv.input[0];
                let weight = &conv.input[1];
                let input_s = sanitize_name(input);
                let weight_s = sanitize_name(weight);
                let input_expr = if let Some((o, l, s)) = weights.get(&input_s) {
                    format!("self.weight({}, {}, &{:?})", o, l, s)
                } else {
                    input_s
                };
                let weight_expr = if let Some((o, l, s)) = weights.get(&weight_s) {
                    format!("self.weight({}, {}, &{:?})", o, l, s)
                } else {
                    weight_s
                };
                let bias = if conv.input.len() > 2 {
                    let b = &conv.input[2];
                    let b_s = sanitize_name(b);
                    if let Some((o, l, s)) = weights.get(&b_s) {
                        format!("Some(&self.weight({}, {}, &{:?}))", o, l, s)
                    } else {
                        format!("Some(&{})", b_s)
                    }
                } else {
                    "None".to_string()
                };
                let mut stride = 1;
                let mut dilation = 1;
                let mut groups = 1;
                let mut padding = 0;
                for attr in &conv.attribute {
                    match attr.name.as_str() {
                        "strides" => {
                            if !attr.ints.is_empty() {
                                stride = attr.ints[0] as usize;
                            }
                        }
                        "dilations" => {
                            if !attr.ints.is_empty() {
                                dilation = attr.ints[0] as usize;
                            }
                        }
                        "group" => groups = attr.i as usize,
                        "pads" => {
                            if !attr.ints.is_empty() {
                                padding = attr.ints[0] as usize;
                            }
                        }
                        _ => {}
                    }
                }
                let output_name = sanitize_name(&nodes[1].output[0]);
                let buf_expr = if let Some(alloc) = allocator {
                    if let Some(&idx) = alloc.tensor_to_buffer.get(&nodes[1].output[0]) {
                        format!("&mut ws.buf_{}", idx)
                    } else {
                        writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                        format!("&mut buf_{}", output_name)
                    }
                } else {
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                    format!("&mut buf_{}", output_name)
                };
                writeln!(
                    w,
                    "{}let {} = self.conv1d_relu({}, {}, {}, {}, {}, {}, {}, {});",
                    tab,
                    output_name,
                    input_expr,
                    weight_expr,
                    bias,
                    stride,
                    dilation,
                    groups,
                    padding,
                    buf_expr
                )?;
                Ok(())
            }),
        },
        Pattern {
            name: "Linear".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 2 {
                    return None;
                }
                if nodes[0].op_type == "MatMul"
                    && nodes[1].op_type == "Add"
                    && (nodes[0].output[0] == nodes[1].input[0]
                        || nodes[0].output[0] == nodes[1].input[1])
                {
                    return Some(2);
                }
                None
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let mm = nodes[0];
                let add = nodes[1];
                let tab = "    ".repeat(indent);
                let input = sanitize_name(&mm.input[0]);
                let getter = |name: &str| -> String {
                    let s = sanitize_name(name);
                    if let Some((o, l, sh)) = weights.get(&s) {
                        format!("self.weight({}, {}, &{:?})", o, l, sh)
                    } else {
                        s
                    }
                };
                let weight = getter(&mm.input[1]);
                let bias_name = if add.input[0] == mm.output[0] {
                    &add.input[1]
                } else {
                    &add.input[0]
                };
                let bias = getter(bias_name);
                let output_name = sanitize_name(&add.output[0]);
                let buf_expr = if let Some(alloc) = allocator {
                    if let Some(&idx) = alloc.tensor_to_buffer.get(&add.output[0]) {
                        format!("&mut ws.buf_{}", idx)
                    } else {
                        writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                        format!("&mut buf_{}", output_name)
                    }
                } else {
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                    format!("&mut buf_{}", output_name)
                };
                writeln!(
                    w,
                    "{}let {} = self.linear(&{}, &{}, &{}, {});",
                    tab, output_name, input, weight, bias, buf_expr
                )?;
                Ok(())
            }),
        },
    ]
}
