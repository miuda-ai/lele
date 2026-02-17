use super::{Pattern, sanitize_name};
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
                    if let Some((o, l, sh, dt)) = weights.get(&s) {
                        match dt {
                            1 => format!("self.weight_f32({}, {}, &{:?})", o, l, sh),
                            3 => format!("self.weight_i8({}, {}, &{:?})", o, l, sh),
                            6 => format!("self.weight_i32_f32({}, {}, &{:?})", o, l, sh),
                            7 => format!("self.weight_i64_f32({}, {}, &{:?})", o, l, sh),
                            10 => format!("self.weight_f16({}, {}, &{:?})", o, l, sh),
                            _ => format!("self.weight_f32({}, {}, &{:?})", o, l, sh),
                        }
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
            name: "Quantized Linear + ReLU".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 7 {
                    return None;
                }
                // Match the quantized linear pattern first (6 nodes)
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
                // Then check for ReLU
                if nodes[6].op_type != "Relu" {
                    return None;
                }
                // Verify the ReLU input is the Add output
                let add_output = &nodes[5].output[0];
                if nodes[6].input[0] != *add_output {
                    return None;
                }
                // Verify the quantized linear pattern connections
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
                Some(7)
            }),
            generator: Box::new(|nodes, weights, _allocator, w, indent| {
                let input = sanitize_name(&nodes[0].input[0]);
                let get_weight = |name: &str| -> String {
                    let s = sanitize_name(name);
                    if let Some((o, l, sh, dt)) = weights.get(&s) {
                        match dt {
                            1 => format!("self.weight_f32({}, {}, &{:?})", o, l, sh),
                            2 => format!("self.weight_u8({}, {}, &{:?})", o, l, sh),
                            3 => format!("self.weight_i8({}, {}, &{:?})", o, l, sh),
                            6 => format!("self.weight_i32({}, {}, &{:?})", o, l, sh),
                            7 => format!("self.weight_i64({}, {}, &{:?})", o, l, sh),
                            10 => format!("self.weight_f16({}, {}, &{:?})", o, l, sh),
                            _ => format!("self.weight_f32({}, {}, &{:?})", o, l, sh),
                        }
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
                let output_name = sanitize_name(&nodes[6].output[0]);
                let tab = "    ".repeat(indent);
                // Always allocate a new buffer for ReLU pattern since allocator doesn't know about patterns
                writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                let buf_expr = format!("&mut buf_{}", output_name);

                // Check if we can use ARM prepared weights path
                let weight_name = sanitize_name(&nodes[2].input[1]);
                if let Some((o, l, sh, dt)) = weights.get(&weight_name) {
                    if (*dt == 3 || *dt == 2) && sh.len() == 2 {
                        let k = sh[0];
                        let n = sh[1];
                        writeln!(w, "{}#[cfg(target_arch = \"aarch64\")]", tab)?;
                        writeln!(
                            w,
                            "{}let {} = self.linear_quantized_relu_arm(&{}, {}, {}, {}, {}, {}, {}, {}, {});",
                            tab,
                            output_name,
                            input,
                            *o,
                            *l,
                            k,
                            n,
                            weight_scale,
                            weight_zero,
                            bias,
                            buf_expr
                        )?;
                        writeln!(w, "{}#[cfg(not(target_arch = \"aarch64\"))]", tab)?;
                        writeln!(
                            w,
                            "{}let {} = self.linear_quantized_relu(&{}, {}, {}, {}, {}, {});",
                            tab,
                            output_name,
                            input,
                            weight_int8,
                            weight_scale,
                            weight_zero,
                            bias,
                            buf_expr
                        )?;
                        return Ok(());
                    }
                }

                writeln!(
                    w,
                    "{}let {} = self.linear_quantized_relu(&{}, {}, {}, {}, {}, {});",
                    tab, output_name, input, weight_int8, weight_scale, weight_zero, bias, buf_expr
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
                    if let Some((o, l, sh, dt)) = weights.get(&s) {
                        match dt {
                            1 => format!("self.weight_f32({}, {}, &{:?})", o, l, sh),
                            2 => format!("self.weight_u8({}, {}, &{:?})", o, l, sh),
                            3 => format!("self.weight_i8({}, {}, &{:?})", o, l, sh),
                            6 => format!("self.weight_i32({}, {}, &{:?})", o, l, sh),
                            7 => format!("self.weight_i64({}, {}, &{:?})", o, l, sh),
                            10 => format!("self.weight_f16({}, {}, &{:?})", o, l, sh),
                            _ => format!("self.weight_f32({}, {}, &{:?})", o, l, sh),
                        }
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

                // Check if we can use ARM prepared weights path
                let weight_name = sanitize_name(&nodes[2].input[1]);
                if let Some((o, l, sh, dt)) = weights.get(&weight_name) {
                    if (*dt == 3 || *dt == 2) && sh.len() == 2 {
                        // ARM-optimized path: use pre-packed weights
                        let k = sh[0];
                        let n = sh[1];
                        writeln!(w, "{}#[cfg(target_arch = \"aarch64\")]", tab)?;
                        writeln!(
                            w,
                            "{}let {} = self.linear_quantized_arm(&{}, {}, {}, {}, {}, {}, {}, {}, {});",
                            tab,
                            output_name,
                            input,
                            *o,
                            *l,
                            k,
                            n,
                            weight_scale,
                            weight_zero,
                            bias,
                            buf_expr
                        )?;
                        writeln!(w, "{}#[cfg(not(target_arch = \"aarch64\"))]", tab)?;
                        writeln!(
                            w,
                            "{}let {} = self.linear_quantized(&{}, {}, {}, {}, {}, {});",
                            tab,
                            output_name,
                            input,
                            weight_int8,
                            weight_scale,
                            weight_zero,
                            bias,
                            buf_expr
                        )?;
                        return Ok(());
                    }
                }

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
                let value_attr = cos_node
                    .attribute
                    .iter()
                    .find(|a| a.name == "value")
                    .and_then(|a| a.t.as_ref());
                let (value_is_int, value_f32, value_i64) = if let Some(t) = value_attr {
                    let dt = t.data_type;
                    if dt == 6 || dt == 7 || !t.int64_data.is_empty() || !t.int32_data.is_empty() {
                        let v = if !t.int64_data.is_empty() {
                            t.int64_data[0]
                        } else if !t.int32_data.is_empty() {
                            t.int32_data[0] as i64
                        } else {
                            0
                        };
                        (true, v as f32, v)
                    } else {
                        let v = if !t.float_data.is_empty() {
                            t.float_data[0]
                        } else {
                            0.0
                        };
                        (false, v, v as i64)
                    }
                } else {
                    (false, 0.0, 0)
                };
                let shape_input = sanitize_name(&cos_node.input[0]);
                let shape_expr = if let Some((o, l, s, dt)) = weights.get(&shape_input) {
                    let loader = match *dt {
                        1 => "weight_f32",
                        2 => "weight_u8",
                        3 => "weight_i8",
                        6 => "weight_i32",
                        7 => "weight_i64",
                        10 => "weight_f16",
                        _ => "weight_f32",
                    };
                    format!("&self.{}({}, {}, &{:?})", loader, o, l, s)
                } else {
                    format!("&{}", shape_input)
                };
                let weight_expr = if let Some((o, l, s, dt)) = weights.get(&weight_name) {
                    let loader = if value_is_int {
                        match *dt {
                            1 => "weight_f32",
                            2 => "weight_u8",
                            3 => "weight_i8",
                            6 => "weight_i32",
                            7 => "weight_i64",
                            10 => "weight_f16",
                            _ => "weight_i64",
                        }
                    } else {
                        match *dt {
                            1 => "weight_f32",
                            2 => "weight_u8",
                            3 => "weight_i8",
                            6 => "weight_i32_f32",
                            7 => "weight_i64_f32",
                            10 => "weight_f16",
                            _ => "weight_f32",
                        }
                    };
                    format!("self.{}({}, {}, &{:?})", loader, o, l, s)
                } else {
                    weight_name
                };
                let output_name = sanitize_name(&concat_node.output[0]);
                let buf_expr = if value_is_int {
                    // Always use a local buffer for i64 to avoid ws.buf<f32> type mismatch
                    writeln!(w, "{}let mut buf_{} = Vec::<i64>::new();", tab, output_name)?;
                    format!("&mut buf_{}", output_name)
                } else if let Some(alloc) = allocator {
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
                if value_is_int {
                    writeln!(
                        w,
                        "{}let {} = self.embedding_concat_i64({}, {}, {}, {});",
                        tab, output_name, shape_expr, value_i64, weight_expr, buf_expr
                    )?;
                } else {
                    writeln!(
                        w,
                        "{}let {} = self.embedding_concat({}, {:.1}, {}, {});",
                        tab, output_name, shape_expr, value_f32, weight_expr, buf_expr
                    )?;
                }
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
                    // Check kernel_shape attribute to distinguish conv1d vs conv2d
                    let kernel_shape = n0
                        .attribute
                        .iter()
                        .find(|a| a.name == "kernel_shape")
                        .map(|a| a.ints.len())
                        .unwrap_or(1);
                    // Only fuse for 1D convolutions; 2D conv+relu handled separately
                    if kernel_shape >= 2 {
                        return None;
                    }
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
                let input_expr = if let Some((o, l, s, dt)) = weights.get(&input_s) {
                    let loader = match *dt {
                        1 => "weight_f32",
                        2 => "weight_u8",
                        3 => "weight_i8",
                        6 => "weight_i32",
                        7 => "weight_i64",
                        10 => "weight_f16",
                        _ => "weight_f32",
                    };
                    format!("self.{}({}, {}, &{:?})", loader, o, l, s)
                } else {
                    input_s
                };
                let weight_expr = if let Some((o, l, s, dt)) = weights.get(&weight_s) {
                    let loader = match *dt {
                        1 => "weight_f32",
                        2 => "weight_u8",
                        3 => "weight_i8",
                        6 => "weight_i32",
                        7 => "weight_i64",
                        10 => "weight_f16",
                        _ => "weight_f32",
                    };
                    format!("self.{}({}, {}, &{:?})", loader, o, l, s)
                } else {
                    weight_s
                };
                let bias = if conv.input.len() > 2 {
                    let b = &conv.input[2];
                    let b_s = sanitize_name(b);
                    if let Some((o, l, s, dt)) = weights.get(&b_s) {
                        let loader = match *dt {
                            1 => "weight_f32",
                            2 => "weight_u8",
                            3 => "weight_i8",
                            6 => "weight_i32",
                            7 => "weight_i64",
                            10 => "weight_f16",
                            _ => "weight_f32",
                        };
                        format!("Some(&self.{}({}, {}, &{:?}))", loader, o, l, s)
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
        // Conv2d + Sigmoid + Mul → Conv2d with fused SiLU
        // Matches: Conv(x) → Sigmoid(conv_out) → Mul(conv_out, sigmoid_out)
        Pattern {
            name: "Conv2d + SiLU".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 3 {
                    return None;
                }
                let n0 = nodes[0];
                let n1 = nodes[1];
                let n2 = nodes[2];
                if n0.op_type != "Conv" || n1.op_type != "Sigmoid" || n2.op_type != "Mul" {
                    return None;
                }
                // Check kernel_shape — only fuse for 2D convolutions
                let kernel_shape_len = n0
                    .attribute
                    .iter()
                    .find(|a| a.name == "kernel_shape")
                    .map(|a| a.ints.len())
                    .unwrap_or(1);
                if kernel_shape_len < 2 {
                    return None;
                }
                let conv_out = &n0.output[0];
                // Sigmoid input must be conv output
                if n1.input.first() != Some(conv_out) {
                    return None;
                }
                let sig_out = &n1.output[0];
                // Mul must take conv_out and sigmoid_out (SiLU = x * sigmoid(x))
                let is_silu = (n2.input.get(0) == Some(conv_out)
                    && n2.input.get(1) == Some(sig_out))
                    || (n2.input.get(0) == Some(sig_out) && n2.input.get(1) == Some(conv_out));
                if !is_silu {
                    return None;
                }
                Some(3)
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let conv = nodes[0];
                let tab = "    ".repeat(indent);
                let input = &conv.input[0];
                let weight = &conv.input[1];
                let input_s = sanitize_name(input);
                let weight_s = sanitize_name(weight);
                let get_expr =
                    |name_s: &str,
                     ws: &std::collections::HashMap<String, (usize, usize, Vec<usize>, i32)>|
                     -> String {
                        if let Some((o, l, s, dt)) = ws.get(name_s) {
                            let loader = match *dt {
                                1 => "weight_f32",
                                2 => "weight_u8",
                                3 => "weight_i8",
                                6 => "weight_i32",
                                7 => "weight_i64",
                                10 => "weight_f16",
                                _ => "weight_f32",
                            };
                            format!("self.{}({}, {}, &{:?})", loader, o, l, s)
                        } else {
                            name_s.to_string()
                        }
                    };
                let input_expr = get_expr(&input_s, weights);
                let weight_expr = get_expr(&weight_s, weights);
                let bias = if conv.input.len() > 2 && !conv.input[2].is_empty() {
                    let b_s = sanitize_name(&conv.input[2]);
                    let b_expr = get_expr(&b_s, weights);
                    format!("Some(&{})", b_expr)
                } else {
                    "None".to_string()
                };
                let mut dilations = vec![1i64, 1];
                let mut group = 1i64;
                let mut pads = vec![0i64, 0, 0, 0];
                let mut strides = vec![1i64, 1];
                for attr in &conv.attribute {
                    match attr.name.as_str() {
                        "dilations" => dilations = attr.ints.clone(),
                        "group" => group = attr.i,
                        "pads" => pads = attr.ints.clone(),
                        "strides" => strides = attr.ints.clone(),
                        _ => {}
                    }
                }
                let output_name = sanitize_name(&nodes[2].output[0]);
                // Use the Sigmoid node's buffer for the conv2d intermediate output
                let conv_out_name = sanitize_name(&nodes[0].output[0]);
                let conv_buf = if let Some(alloc) = allocator {
                    if let Some(&idx) = alloc.tensor_to_buffer.get(&nodes[0].output[0]) {
                        format!("&mut ws.buf_{}", idx)
                    } else {
                        writeln!(
                            w,
                            "{}let mut buf_{} = Vec::<f32>::new();",
                            tab, conv_out_name
                        )?;
                        format!("&mut buf_{}", conv_out_name)
                    }
                } else {
                    writeln!(
                        w,
                        "{}let mut buf_{} = Vec::<f32>::new();",
                        tab, conv_out_name
                    )?;
                    format!("&mut buf_{}", conv_out_name)
                };
                let silu_buf = if let Some(alloc) = allocator {
                    if let Some(&idx) = alloc.tensor_to_buffer.get(&nodes[2].output[0]) {
                        format!("&mut ws.buf_{}", idx)
                    } else {
                        writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                        format!("&mut buf_{}", output_name)
                    }
                } else {
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, output_name)?;
                    format!("&mut buf_{}", output_name)
                };
                // Emit Conv2d, then vectorized SiLU (avoids scalar SiLU in bias loop)
                writeln!(
                    w,
                    "{}let {} = lele::kernels::conv2d(&{}, &{}, {}, &{:?}, {}, &{:?}, &{:?}, {});",
                    tab,
                    conv_out_name,
                    input_expr,
                    weight_expr,
                    bias,
                    dilations,
                    group,
                    pads,
                    strides,
                    conv_buf
                )?;
                writeln!(
                    w,
                    "{}let {} = lele::kernels::silu(&{}, {});",
                    tab, output_name, conv_out_name, silu_buf
                )?;
                Ok(())
            }),
        },
        // Interleaved triple SiLU: Sig(A) → Sig(B) → Sig(C) → Mul(A,sigA) → Mul(B,sigB) → Mul(C,sigC)
        Pattern {
            name: "Triple SiLU".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 6 {
                    return None;
                }
                let (n0, n1, n2, n3, n4, n5) =
                    (nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5]);
                if n0.op_type != "Sigmoid"
                    || n1.op_type != "Sigmoid"
                    || n2.op_type != "Sigmoid"
                    || n3.op_type != "Mul"
                    || n4.op_type != "Mul"
                    || n5.op_type != "Mul"
                {
                    return None;
                }
                // Check SiLU pattern for each pair
                let check_silu = |sig: &NodeProto, mul: &NodeProto| -> bool {
                    let sig_in = &sig.input[0];
                    let sig_out = &sig.output[0];
                    (mul.input.get(0) == Some(sig_in) && mul.input.get(1) == Some(sig_out))
                        || (mul.input.get(0) == Some(sig_out) && mul.input.get(1) == Some(sig_in))
                };
                if check_silu(n0, n3) && check_silu(n1, n4) && check_silu(n2, n5) {
                    Some(6)
                } else {
                    None
                }
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let tab = "    ".repeat(indent);
                let get_input = |node_idx: usize| -> String {
                    let input_s = sanitize_name(&nodes[node_idx].input[0]);
                    if let Some((o, l, s, dt)) = weights.get(&input_s) {
                        let loader = match *dt {
                            1 => "weight_f32",
                            2 => "weight_u8",
                            3 => "weight_i8",
                            6 => "weight_i32",
                            7 => "weight_i64",
                            10 => "weight_f16",
                            _ => "weight_f32",
                        };
                        format!("self.{}({}, {}, &{:?})", loader, o, l, s)
                    } else {
                        input_s
                    }
                };
                let get_buf = |out_key: &str,
                               name: &str,
                               tab: &str,
                               w: &mut dyn std::io::Write,
                               allocator: Option<&super::Allocator>|
                 -> std::io::Result<String> {
                    if let Some(alloc) = allocator {
                        if let Some(&idx) = alloc.tensor_to_buffer.get(out_key) {
                            return Ok(format!("&mut ws.buf_{}", idx));
                        }
                    }
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, name)?;
                    Ok(format!("&mut buf_{}", name))
                };
                // Three SiLU calls: (sig[0]+mul[3]), (sig[1]+mul[4]), (sig[2]+mul[5])
                for (sig_idx, mul_idx) in [(0, 3), (1, 4), (2, 5)] {
                    let input = get_input(sig_idx);
                    let out_name = sanitize_name(&nodes[mul_idx].output[0]);
                    let buf = get_buf(&nodes[mul_idx].output[0], &out_name, &tab, w, allocator)?;
                    writeln!(
                        w,
                        "{}let {} = lele::kernels::silu(&{}, {});",
                        tab, out_name, input, buf
                    )?;
                }
                Ok(())
            }),
        },
        // Interleaved double SiLU: Sigmoid(A) → Sigmoid(B) → Mul(A, sig_A) → Mul(B, sig_B)
        // Common in YOLO architectures where the ONNX exporter batches Sigmoids
        Pattern {
            name: "Double SiLU".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 4 {
                    return None;
                }
                let (n0, n1, n2, n3) = (nodes[0], nodes[1], nodes[2], nodes[3]);
                if n0.op_type != "Sigmoid"
                    || n1.op_type != "Sigmoid"
                    || n2.op_type != "Mul"
                    || n3.op_type != "Mul"
                {
                    return None;
                }
                let sig_a_in = &n0.input[0];
                let sig_a_out = &n0.output[0];
                let sig_b_in = &n1.input[0];
                let sig_b_out = &n1.output[0];
                // Mul(A, sig_A)
                let mul_a_ok = (n2.input.get(0) == Some(sig_a_in)
                    && n2.input.get(1) == Some(sig_a_out))
                    || (n2.input.get(0) == Some(sig_a_out) && n2.input.get(1) == Some(sig_a_in));
                // Mul(B, sig_B)
                let mul_b_ok = (n3.input.get(0) == Some(sig_b_in)
                    && n3.input.get(1) == Some(sig_b_out))
                    || (n3.input.get(0) == Some(sig_b_out) && n3.input.get(1) == Some(sig_b_in));
                if mul_a_ok && mul_b_ok { Some(4) } else { None }
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let tab = "    ".repeat(indent);
                let get_input = |node_idx: usize| -> String {
                    let input_s = sanitize_name(&nodes[node_idx].input[0]);
                    if let Some((o, l, s, dt)) = weights.get(&input_s) {
                        let loader = match *dt {
                            1 => "weight_f32",
                            2 => "weight_u8",
                            3 => "weight_i8",
                            6 => "weight_i32",
                            7 => "weight_i64",
                            10 => "weight_f16",
                            _ => "weight_f32",
                        };
                        format!("self.{}({}, {}, &{:?})", loader, o, l, s)
                    } else {
                        input_s
                    }
                };
                let get_buf = |out_key: &str,
                               name: &str,
                               tab: &str,
                               w: &mut dyn std::io::Write,
                               allocator: Option<&super::Allocator>|
                 -> std::io::Result<String> {
                    if let Some(alloc) = allocator {
                        if let Some(&idx) = alloc.tensor_to_buffer.get(out_key) {
                            return Ok(format!("&mut ws.buf_{}", idx));
                        }
                    }
                    writeln!(w, "{}let mut buf_{} = Vec::<f32>::new();", tab, name)?;
                    Ok(format!("&mut buf_{}", name))
                };
                // SiLU for first pair (nodes[0] Sigmoid + nodes[2] Mul)
                let input_a = get_input(0);
                let out_a = sanitize_name(&nodes[2].output[0]);
                let buf_a = get_buf(&nodes[2].output[0], &out_a, &tab, w, allocator)?;
                writeln!(
                    w,
                    "{}let {} = lele::kernels::silu(&{}, {});",
                    tab, out_a, input_a, buf_a
                )?;
                // SiLU for second pair (nodes[1] Sigmoid + nodes[3] Mul)
                let input_b = get_input(1);
                let out_b = sanitize_name(&nodes[3].output[0]);
                let buf_b = get_buf(&nodes[3].output[0], &out_b, &tab, w, allocator)?;
                writeln!(
                    w,
                    "{}let {} = lele::kernels::silu(&{}, {});",
                    tab, out_b, input_b, buf_b
                )?;
                Ok(())
            }),
        },
        // Sigmoid + Mul → SiLU (standalone, not preceded by Conv)
        // Matches: Sigmoid(x) → Mul(x, sigmoid_out)
        Pattern {
            name: "SiLU".to_string(),
            matcher: Box::new(|nodes: &[&NodeProto]| -> Option<usize> {
                if nodes.len() < 2 {
                    return None;
                }
                let n0 = nodes[0];
                let n1 = nodes[1];
                if n0.op_type != "Sigmoid" || n1.op_type != "Mul" {
                    return None;
                }
                let sig_input = &n0.input[0];
                let sig_out = &n0.output[0];
                // Mul must take original input and sigmoid output
                let is_silu = (n1.input.get(0) == Some(sig_input)
                    && n1.input.get(1) == Some(sig_out))
                    || (n1.input.get(0) == Some(sig_out) && n1.input.get(1) == Some(sig_input));
                if !is_silu {
                    return None;
                }
                Some(2)
            }),
            generator: Box::new(|nodes, weights, allocator, w, indent| {
                let tab = "    ".repeat(indent);
                let input_s = sanitize_name(&nodes[0].input[0]);
                let input_expr = if let Some((o, l, s, dt)) = weights.get(&input_s) {
                    let loader = match *dt {
                        1 => "weight_f32",
                        2 => "weight_u8",
                        3 => "weight_i8",
                        6 => "weight_i32",
                        7 => "weight_i64",
                        10 => "weight_f16",
                        _ => "weight_f32",
                    };
                    format!("self.{}({}, {}, &{:?})", loader, o, l, s)
                } else {
                    input_s
                };
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
                    "{}let {} = lele::kernels::silu(&{}, {});",
                    tab, output_name, input_expr, buf_expr
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
                    if let Some((o, l, sh, dt)) = weights.get(&s) {
                        match dt {
                            1 => format!("self.weight_f32({}, {}, &{:?})", o, l, sh),
                            3 => format!("self.weight_i8({}, {}, &{:?})", o, l, sh),
                            10 => format!("self.weight_f16({}, {}, &{:?})", o, l, sh),
                            _ => format!("self.weight_f32({}, {}, &{:?})", o, l, sh),
                        }
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
