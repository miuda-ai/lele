pub mod onnx_proto {
    include!("onnx_proto_gen.rs");
}
use self::onnx_proto::{GraphProto, ModelProto, TensorProto};
use ::prost::DecodeError;
use ::prost::Message;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("Protobuf decode error")]
    Decode(#[from] DecodeError),
    #[error("Invalid tensor data")]
    InvalidTensorData,
    #[error("Missing graph")]
    MissingGraph,
}
pub struct OnnxModel {
    pub proto: ModelProto,
}
impl OnnxModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ModelError> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let mut proto = ModelProto::decode(&buffer[..])?;
        let model_dir = path.parent().unwrap_or(Path::new("."));
        if let Some(graph) = proto.graph.as_mut() {
            Self::load_external_data(graph, model_dir)?;
        }
        Ok(Self { proto })
    }

    fn load_external_data(graph: &mut GraphProto, model_dir: &Path) -> Result<(), ModelError> {
        let mut data_cache: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();
        for init in &mut graph.initializer {
            if init.data_location == 1 && init.raw_data.is_empty() {
                let location = init.external_data.iter()
                    .find(|e| e.key == "location")
                    .map(|e| e.value.as_str())
                    .unwrap_or("");
                let offset: usize = init.external_data.iter()
                    .find(|e| e.key == "offset")
                    .and_then(|e| e.value.parse().ok())
                    .unwrap_or(0);
                let length: usize = init.external_data.iter()
                    .find(|e| e.key == "length")
                    .and_then(|e| e.value.parse().ok())
                    .unwrap_or(0);
                if location.is_empty() || length == 0 {
                    continue;
                }
                let data_file_path = model_dir.join(location);
                let data_file_key = data_file_path.to_string_lossy().to_string();
                let data = data_cache.entry(data_file_key.clone()).or_insert_with(|| {
                    let mut f = File::open(&data_file_path).ok();
                    match &mut f {
                        Some(f) => {
                            let mut buf = Vec::new();
                            f.read_to_end(&mut buf).ok();
                            buf
                        }
                        None => Vec::new(),
                    }
                });
                if offset + length <= data.len() {
                    init.raw_data = data[offset..offset + length].to_vec();
                }
            }
        }
        Ok(())
    }

    pub fn graph(&self) -> Option<&GraphProto> {
        self.proto.graph.as_ref()
    }
}
pub fn tensor_to_array(tensor: &TensorProto) -> Result<(Vec<f32>, Vec<usize>), ModelError> {
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = if !tensor.float_data.is_empty() {
        tensor.float_data.clone()
    } else if !tensor.int64_data.is_empty() {
        tensor.int64_data.iter().map(|&x| x as f32).collect()
    } else if !tensor.int32_data.is_empty() {
        tensor.int32_data.iter().map(|&x| x as f32).collect()
    } else if !tensor.raw_data.is_empty() {
        match tensor.data_type {
            1 => tensor
                .raw_data
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    f32::from_le_bytes(bytes)
                })
                .collect(),
            2 => tensor.raw_data.iter().map(|&x| x as f32).collect(),
            3 => tensor.raw_data.iter().map(|&x| (x as i8) as f32).collect(),
            6 => tensor
                .raw_data
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    i32::from_le_bytes(bytes) as f32
                })
                .collect(),
            7 => tensor
                .raw_data
                .chunks_exact(8)
                .map(|chunk| {
                    let bytes = [
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ];
                    i64::from_le_bytes(bytes) as f32
                })
                .collect(),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };
    Ok((data, dims))
}

pub fn tensor_to_vec_u8(tensor: &TensorProto) -> Result<(Vec<u8>, Vec<usize>, i32), ModelError> {
    let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
    let data_type = tensor.data_type;

    if !tensor.raw_data.is_empty() {
        return Ok((tensor.raw_data.clone(), dims, data_type));
    }

    // Fallback if raw_data is empty (usually for small constants in some ONNX exporters)
    match data_type {
        1 => {
            // FLOAT
            let mut bytes = Vec::with_capacity(tensor.float_data.len() * 4);
            for &f in &tensor.float_data {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            Ok((bytes, dims, data_type))
        }
        2 => {
            // UINT8
            Ok((
                tensor.int32_data.iter().map(|&x| x as u8).collect(),
                dims,
                data_type,
            ))
        }
        3 => {
            // INT8
            Ok((
                tensor.int32_data.iter().map(|&x| x as u8).collect(),
                dims,
                data_type,
            ))
        }
        6 => {
            // INT32
            let mut bytes = Vec::with_capacity(tensor.int32_data.len() * 4);
            for &i in &tensor.int32_data {
                bytes.extend_from_slice(&i.to_le_bytes());
            }
            Ok((bytes, dims, data_type))
        }
        7 => {
            // INT64
            let mut bytes = Vec::with_capacity(tensor.int64_data.len() * 8);
            for &i in &tensor.int64_data {
                bytes.extend_from_slice(&i.to_le_bytes());
            }
            Ok((bytes, dims, data_type))
        }
        _ => Err(ModelError::InvalidTensorData),
    }
}

pub fn find_constant_node_tensor<'a>(
    graph: &'a GraphProto,
    name_suffix: &str,
) -> Option<&'a TensorProto> {
    for node in &graph.node {
        if node.op_type == "Constant" {
            for output in &node.output {
                if output.ends_with(name_suffix) {
                    for attr in &node.attribute {
                        if attr.name == "value" {
                            return attr.t.as_ref();
                        }
                    }
                }
            }
        }
    }
    None
}
