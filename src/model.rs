pub mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
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
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let proto = ModelProto::decode(&buffer[..])?;
        Ok(Self { proto })
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
