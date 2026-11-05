// Hojo-TTS-Light is a Token-LM TTS with three ONNX sub-models. lele compiles
// the encoder and the decoder. The LLM runs by hand with a KV cache
// (src/llm_cache.rs) on the weights this script takes from its ONNX model by
// the nodes that read them: it writes them to src/llm_weights.bin, as the
// model stores them, and their offsets to src/llm_layout.rs. Neither depends
// on how lele lays out or packs the weights of the models it compiles.

use lele::model::onnx_proto::{GraphProto, NodeProto, TensorProto};
use lele::model::{OnnxModel, tensor_to_vec_u8};
use lele_build::Context;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

const LLM: &str = "onnx/Hojo-TTS-Light-llm.onnx";
const WEIGHTS: &str = "src/llm_weights.bin";
const LAYOUT: &str = "src/llm_layout.rs";

const D_MODEL: i64 = 512;
const D_HEAD: i64 = 128;
const D_FF: i64 = 2304;
const VOCAB: i64 = 17667;
const N_LAYERS: usize = 17;

/// The weight matrices of a layer, `[in, out]`, by the module whose MatMul
/// reads them, in the order of `LAYERS` in the layout.
const PROJECTIONS: [(&str, [i64; 2]); 7] = [
    ("self_attn/q_proj", [D_MODEL, D_MODEL]),
    ("self_attn/k_proj", [D_MODEL, D_HEAD]),
    ("self_attn/v_proj", [D_MODEL, D_HEAD]),
    ("self_attn/o_proj", [D_MODEL, D_MODEL]),
    ("mlp/gate_proj", [D_MODEL, D_FF]),
    ("mlp/up_proj", [D_MODEL, D_FF]),
    ("mlp/down_proj", [D_FF, D_MODEL]),
];

fn main() -> lele_build::Result {
    lele_build::build_models(
        "model.toml",
        "src",
        &[
            ("onnx/Hojo-TTS-Light-encoder.onnx", "Encoder"),
            ("onnx/Hojo-TTS-Light-decoder.onnx", "Decoder"),
        ],
    )?;
    println!("cargo:rerun-if-changed={LLM}");
    if lele_build::should_skip_codegen() {
        return Ok(());
    }
    let modified = |p: &str| Path::new(p).metadata().and_then(|m| m.modified()).ok();
    let inputs = modified(LLM).max(modified("build.rs"));
    let fresh = [WEIGHTS, LAYOUT].iter().all(|p| modified(p) >= inputs);
    if fresh && !lele_build::should_force_regenerate() {
        return Ok(());
    }
    if !Path::new(LLM).exists() {
        let config = lele_build::config::ModelConfig::load("model.toml")?;
        lele_build::fetch_model_files(&config.model, Path::new(""))?;
    }
    println!("cargo:warning=Extracting the LLM weights from {LLM}...");
    extract_llm().context(|| format!("extracting the LLM weights from {LLM}"))
}

fn fail(message: String) -> lele_build::Error {
    std::io::Error::other(message).into()
}

/// Writes the weights the cached LLM reads, and their offsets.
fn extract_llm() -> lele_build::Result {
    let model = OnnxModel::load(LLM)?;
    let graph = Graph::new(model.graph().ok_or_else(|| fail("no graph".into()))?);
    let layer = |n: usize, node: &str| format!("/model/model/layers.{n}/{node}");

    // The exporter stores one RMSNorm weight for every layer and the final
    // norm, and one for every query and key norm; the cached path relies on it.
    let norm = graph.input(&layer(0, "input_layernorm/Mul_1"), 0, &[])?;
    let qk_norm = graph.input(&layer(0, "self_attn/q_norm/Mul_1"), 0, &[])?;
    // Attention scores are scaled by the same constant in every layer.
    let attn_scale = graph.input(&layer(0, "self_attn/Mul_8"), 1, &[])?;
    let mut shared = vec![(norm, "/model/model/norm/Mul_1".to_string(), 0)];
    for n in 0..N_LAYERS {
        for node in ["input_layernorm/Mul_1", "post_attention_layernorm/Mul_1"] {
            shared.push((norm, layer(n, node), 0));
        }
        for node in ["self_attn/q_norm/Mul_1", "self_attn/k_norm/Mul_1"] {
            shared.push((qk_norm, layer(n, node), 0));
        }
        shared.push((attn_scale, layer(n, "self_attn/Mul_8"), 1));
    }
    let contents = |t: &TensorProto| -> lele_build::Result<_> {
        Ok((t.data_type, t.dims.clone(), tensor_to_vec_u8(t)?.0))
    };
    for (tensor, node, i) in shared {
        if contents(tensor)? != contents(graph.input(&node, i, &[])?)? {
            return Err(fail(format!("{node} reads another weight than layer 0")));
        }
    }
    // Rotary embeddings: cos and sin of the positions times the inverse
    // frequencies, scaled by 1.
    for node in ["Mul_1", "Mul_2"] {
        let scale = graph.input(&format!("/model/model/rotary_emb/{node}"), 1, &[])?;
        if tensor_to_vec_u8(scale)?.0 != 1.0f32.to_le_bytes() {
            return Err(fail(format!("rotary_emb/{node} scales by other than 1")));
        }
    }
    let inv_freq = graph.input("/model/model/rotary_emb/MatMul", 0, &["Cast", "Expand"])?;

    let mut out = Out::default();
    let mut layout = String::new();
    let mut constant = |name: &str, offset: usize| {
        writeln!(layout, "pub const {name}: usize = {offset};").unwrap();
    };
    let embed = graph.input("/model/model/embed_tokens/Gather", 0, &[])?;
    constant("EMBED", out.put(embed, &[VOCAB, D_MODEL], false)?);
    constant("NORM", out.put(norm, &[D_MODEL], false)?);
    constant("QK_NORM", out.put(qk_norm, &[D_HEAD], false)?);
    constant("INV_FREQ", out.put(inv_freq, &[1, D_HEAD / 2, 1], true)?);
    constant("ATTN_SCALE", out.put(attn_scale, &[], false)?);
    let lm_head = graph.input("/model/lm_head/MatMul", 1, &[])?;
    constant("LM_HEAD", out.put(lm_head, &[D_MODEL, VOCAB], false)?);
    let mut layers = Vec::new();
    for n in 0..N_LAYERS {
        let mut offsets = Vec::new();
        for (module, dims) in PROJECTIONS {
            let w = graph.input(&layer(n, &format!("{module}/MatMul")), 1, &[])?;
            offsets.push(out.put(w, &dims, false)?.to_string());
        }
        layers.push(format!("[{}]", offsets.join(", ")));
    }
    let regions: Vec<String> = out.f16.iter().map(|(o, l)| format!("({o}, {l})")).collect();
    let layout = format!(
        "// Generated by build.rs from {LLM}: byte offsets of the weights in\n\
         // llm_weights.bin.\n\
         {layout}\
         /// Of each layer: the q, k, v, o, gate, up and down projections.\n\
         pub const LAYERS: [[usize; {}]; {N_LAYERS}] = [{}];\n\
         /// Byte offset and length of every f16 weight.\n\
         pub const F16_REGIONS: [(usize, usize); {}] = [{}];\n",
        PROJECTIONS.len(),
        layers.join(", "),
        regions.len(),
        regions.join(", "),
    );
    std::fs::write(WEIGHTS, &out.data)?;
    std::fs::write(LAYOUT, layout)?;
    Ok(())
}

/// The nodes and constants of a graph, by name.
struct Graph<'a> {
    nodes: HashMap<&'a str, &'a NodeProto>,
    producers: HashMap<&'a str, &'a NodeProto>,
    constants: HashMap<&'a str, &'a TensorProto>,
}

impl<'a> Graph<'a> {
    fn new(g: &'a GraphProto) -> Self {
        let mut constants: HashMap<&str, &TensorProto> =
            g.initializer.iter().map(|t| (t.name.as_str(), t)).collect();
        for n in g.node.iter().filter(|n| n.op_type == "Constant") {
            if let (Some(out), Some(t)) = (n.output.first(), n.attribute[0].t.as_ref()) {
                constants.insert(out, t);
            }
        }
        Self {
            nodes: g.node.iter().map(|n| (n.name.as_str(), n)).collect(),
            producers: g
                .node
                .iter()
                .flat_map(|n| n.output.iter().map(move |o| (o.as_str(), n)))
                .collect(),
            constants,
        }
    }

    /// The constant input `i` of node `node` reads, through `Identity` nodes
    /// and the operators `through`, whose first input it then is.
    fn input(&self, node: &str, i: usize, through: &[&str]) -> lele_build::Result<&'a TensorProto> {
        let n = self
            .nodes
            .get(node)
            .ok_or_else(|| fail(format!("no node {node}")))?;
        let mut name = n
            .input
            .get(i)
            .ok_or_else(|| fail(format!("{node} has no input {i}")))?;
        loop {
            if let Some(t) = self.constants.get(name.as_str()) {
                return Ok(t);
            }
            match self.producers.get(name.as_str()) {
                Some(p) if p.op_type == "Identity" || through.contains(&p.op_type.as_str()) => {
                    name = &p.input[0];
                }
                _ => return Err(fail(format!("input {i} of {node} is not a constant"))),
            }
        }
    }
}

/// The weights file being written.
#[derive(Default)]
struct Out {
    data: Vec<u8>,
    /// Byte offset and length of each f16 weight.
    f16: Vec<(usize, usize)>,
}

impl Out {
    /// Appends `t`, which has to be an f16 tensor of shape `dims` (an f32 one
    /// if `f32`), and returns its offset.
    fn put(&mut self, t: &TensorProto, dims: &[i64], f32: bool) -> lele_build::Result<usize> {
        // TensorProto.DataType FLOAT and FLOAT16.
        let data_type = if f32 { 1 } else { 10 };
        if (t.data_type, &t.dims[..]) != (data_type, dims) {
            return Err(fail(format!(
                "{} has type {} and shape {:?}, not {data_type} and {dims:?}",
                t.name, t.data_type, t.dims
            )));
        }
        let (bytes, _, _) = tensor_to_vec_u8(t)?;
        self.data.resize(self.data.len().next_multiple_of(16), 0);
        let offset = self.data.len();
        self.data.extend_from_slice(&bytes);
        if !f32 {
            self.f16.push((offset, bytes.len()));
        }
        Ok(offset)
    }
}
