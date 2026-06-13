use crate::llm::Llm;

const D_MODEL: usize = 512;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 1;
const D_HEAD: usize = 128;
const D_FF: usize = 2304;
const N_LAYERS: usize = 17;
const VOCAB: usize = 17667;
const LAYER_STRIDE: usize = 8388608;
const LAYER0_Q_OFFSET: usize = 18092288;
const EMBED_OFFSET: usize = 0;
const NORM_WEIGHT_OFFSET: usize = 18091264;
const QK_NORM_OFFSET: usize = 18091008;
const INV_FREQ_OFFSET: usize = 178789744;
const ATTN_SCALE_OFFSET: usize = 178790096;
const LM_HEAD_OFFSET: usize = 160698624;
const LN_EPS: f32 = 1e-6;

fn layer_q_off(n: usize) -> usize { LAYER0_Q_OFFSET + n * LAYER_STRIDE }
fn layer_k_off(n: usize) -> usize { layer_q_off(n) + 524288 }
fn layer_v_off(n: usize) -> usize { layer_k_off(n) + 131072 }
fn layer_o_off(n: usize) -> usize { layer_v_off(n) + 131072 }
fn layer_gate_off(n: usize) -> usize { layer_o_off(n) + 524288 }
fn layer_up_off(n: usize) -> usize { layer_gate_off(n) + 2359296 }
fn layer_down_off(n: usize) -> usize { layer_up_off(n) + 2359296 }

pub struct LlmCache {
    pub k_cache: Vec<Vec<f32>>,
    pub v_cache: Vec<Vec<f32>>,
    inv_freq: Vec<f32>,
    attn_scale: f32,
    seq_len: usize,
}

impl LlmCache {
    pub fn new(llm: &Llm) -> Self {
        let inv_freq: Vec<f32> = llm.weight_f32(INV_FREQ_OFFSET, 256, &[64])
            .data.to_vec();
        let attn_scale = llm.weight_f16(ATTN_SCALE_OFFSET, 2, &[]).data[0];
        Self {
            k_cache: vec![Vec::new(); N_LAYERS],
            v_cache: vec![Vec::new(); N_LAYERS],
            inv_freq,
            attn_scale,
            seq_len: 0,
        }
    }

    pub fn seq_len(&self) -> usize { self.seq_len }
}

impl<'a> Llm<'a> {
    fn rms_norm_inplace(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        let n = x.len();
        let mut ms = 0.0f32;
        for &v in &x[..n] {
            ms += v * v;
        }
        let rms = (ms / n as f32 + eps).sqrt();
        let inv = 1.0 / rms;
        for i in 0..n {
            x[i] = x[i] * inv * weight[i];
        }
    }

    fn matvec(&self, x: &[f32], w: &[f32], m: usize, k: usize) -> Vec<f32> {
        // w is [k, m] = [in_dim, out_dim] row-major (ONNX MatMul convention)
        // out[j] = sum_i x[i] * w[i*m + j]
        let mut out = vec![0.0f32; m];
        for i in 0..k {
            let xi = x[i];
            let wi = &w[i * m..(i + 1) * m];
            for j in 0..m {
                out[j] += xi * wi[j];
            }
        }
        out
    }

    pub fn forward_prompt(&self, cache: &mut LlmCache, input_ids: &[i64]) -> Vec<f32> {
        if input_ids.is_empty() {
            return vec![0.0; VOCAB];
        }

        let seq = input_ids.len();
        cache.seq_len = seq;

        let embed_w = self.weight_f16(EMBED_OFFSET, 17667 * 512 * 2, &[VOCAB, D_MODEL]);
        let mut hidden = vec![0.0f32; seq * D_MODEL];
        for t in 0..seq {
            let row = &embed_w.data[input_ids[t] as usize * D_MODEL..(input_ids[t] as usize + 1) * D_MODEL];
            hidden[t * D_MODEL..(t + 1) * D_MODEL].copy_from_slice(row);
        }

        let norm_w = self.weight_f16(NORM_WEIGHT_OFFSET, 1024, &[D_MODEL]);
        let norm_w = &norm_w.data[..D_MODEL];
        let qk_norm_w = self.weight_f16(QK_NORM_OFFSET, 256, &[D_HEAD]);
        let qk_norm_w = &qk_norm_w.data[..D_HEAD];

        for layer in 0..N_LAYERS {
            hidden = self.forward_layer_full(
                &hidden, seq, layer, norm_w, qk_norm_w,
                &mut cache.k_cache[layer], &mut cache.v_cache[layer],
                &cache.inv_freq, cache.attn_scale,
            );
        }

        self.rms_norm_inplace(&mut hidden[seq * D_MODEL - D_MODEL..seq * D_MODEL], norm_w, LN_EPS);
        let last_row = &hidden[(seq - 1) * D_MODEL..seq * D_MODEL];
        let lm_head = self.weight_f16(LM_HEAD_OFFSET, VOCAB * D_MODEL * 2, &[D_MODEL, VOCAB]);
        let logits = self.matvec(last_row, &lm_head.data, VOCAB, D_MODEL);
        logits
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_full(
        &self,
        hidden: &[f32],
        seq: usize,
        layer: usize,
        norm_w: &[f32],
        qk_norm_w: &[f32],
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
        inv_freq: &[f32],
        attn_scale: f32,
    ) -> Vec<f32> {
        let residual = hidden.to_vec();

        let mut normed = residual.clone();
        for t in 0..seq {
            let row = &mut normed[t * D_MODEL..(t + 1) * D_MODEL];
            self.rms_norm_inplace(row, norm_w, LN_EPS);
        }

        let qw = self.weight_f16(layer_q_off(layer), 524288, &[D_MODEL, D_MODEL]);
        let kw = self.weight_f16(layer_k_off(layer), 131072, &[D_MODEL, D_HEAD]);
        let vw = self.weight_f16(layer_v_off(layer), 131072, &[D_MODEL, D_HEAD]);
        let ow = self.weight_f16(layer_o_off(layer), 524288, &[D_MODEL, D_MODEL]);

        let mut q_all = vec![0.0f32; seq * N_HEADS * D_HEAD];
        let mut k_all = vec![0.0f32; seq * D_HEAD];
        let mut v_all = vec![0.0f32; seq * D_HEAD];

        for t in 0..seq {
            let x = &normed[t * D_MODEL..(t + 1) * D_MODEL];
            let q = self.matvec(x, &qw.data, D_MODEL, D_MODEL);
            let k = self.matvec(x, &kw.data, D_HEAD, D_MODEL);
            let v = self.matvec(x, &vw.data, D_HEAD, D_MODEL);

            let mut q = q;
            for h in 0..N_HEADS {
                let hr = &mut q[h * D_HEAD..(h + 1) * D_HEAD];
                self.rms_norm_inplace(hr, qk_norm_w, LN_EPS);
            }
            let mut k_normed = k;
            self.rms_norm_inplace(&mut k_normed, qk_norm_w, LN_EPS);

            let pos = t as f32;
            let mut cos = [0.0f32; D_HEAD];
            let mut sin = [0.0f32; D_HEAD];
            for i in 0..(D_HEAD / 2) {
                let freq = pos * inv_freq[i];
            let (s, c) = freq.sin_cos();
                cos[i] = c;
                cos[i + D_HEAD / 2] = c;
                sin[i] = s;
                sin[i + D_HEAD / 2] = s;
            }

            for h in 0..N_HEADS {
                let qh = &mut q_all[t * N_HEADS * D_HEAD + h * D_HEAD..t * N_HEADS * D_HEAD + (h + 1) * D_HEAD];
                for d in 0..D_HEAD {
                    let qh_orig = q[h * D_HEAD + d];
                    let qh_rot = if d < D_HEAD / 2 {
                        -q[h * D_HEAD + d + D_HEAD / 2]
                    } else {
                        q[h * D_HEAD + d - D_HEAD / 2]
                    };
                    qh[d] = qh_orig * cos[d] + qh_rot * sin[d];
                }
            }
            for d in 0..D_HEAD {
                let kh_orig = k_normed[d];
                let kh_rot = if d < D_HEAD / 2 {
                    -k_normed[d + D_HEAD / 2]
                } else {
                    k_normed[d - D_HEAD / 2]
                };
                k_all[t * D_HEAD + d] = kh_orig * cos[d] + kh_rot * sin[d];
            }
            v_all[t * D_HEAD..(t + 1) * D_HEAD].copy_from_slice(&v);
        }

        *k_cache = k_all.clone();
        *v_cache = v_all.clone();

        let mut attn_out = vec![0.0f32; seq * D_MODEL];
        for t in 0..seq {
            for h in 0..N_HEADS {
                let qv = &q_all[t * N_HEADS * D_HEAD + h * D_HEAD..t * N_HEADS * D_HEAD + (h + 1) * D_HEAD];
                let mut scores = vec![0.0f32; t + 1];
                for tt in 0..=t {
                    let kv = &k_all[tt * D_HEAD..(tt + 1) * D_HEAD];
                    let mut s = 0.0f32;
                    for d in 0..D_HEAD {
                        s += qv[d] * kv[d];
                    }
                    scores[tt] = s * attn_scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                for d in 0..D_HEAD {
                    let mut acc = 0.0f32;
                    for tt in 0..=t {
                        acc += scores[tt] * v_all[tt * D_HEAD + d];
                    }
                    attn_out[t * D_MODEL + h * D_HEAD + d] = acc;
                }
            }
        }

        let mut out = vec![0.0f32; seq * D_MODEL];
        for t in 0..seq {
            let ao = &attn_out[t * D_MODEL..(t + 1) * D_MODEL];
            let proj = self.matvec(ao, &ow.data, D_MODEL, D_MODEL);
            for i in 0..D_MODEL {
                out[t * D_MODEL + i] = proj[i] + residual[t * D_MODEL + i];
            }
        }

        let residual2 = out.clone();
        for t in 0..seq {
            let row = &mut out[t * D_MODEL..(t + 1) * D_MODEL];
            self.rms_norm_inplace(row, norm_w, LN_EPS);
        }

        let gw = self.weight_f16(layer_gate_off(layer), 2359296, &[D_MODEL, D_FF]);
        let uw = self.weight_f16(layer_up_off(layer), 2359296, &[D_MODEL, D_FF]);
        let dw = self.weight_f16(layer_down_off(layer), 2359296, &[D_FF, D_MODEL]);

        for t in 0..seq {
            let x = &out[t * D_MODEL..(t + 1) * D_MODEL];
            let gate = self.matvec(x, &gw.data, D_FF, D_MODEL);
            let up = self.matvec(x, &uw.data, D_FF, D_MODEL);
            let mut mid = vec![0.0f32; D_FF];
            for i in 0..D_FF {
                let sig = 1.0 / (1.0 + (-gate[i]).exp());
                mid[i] = gate[i] * sig * up[i];
            }
            let down = self.matvec(&mid, &dw.data, D_MODEL, D_FF);
            for i in 0..D_MODEL {
                out[t * D_MODEL + i] = down[i] + residual2[t * D_MODEL + i];
            }
        }

        out
    }

    pub fn forward_token(&self, cache: &mut LlmCache, token_id: i64) -> Vec<f32> {
        let pos = cache.seq_len;
        cache.seq_len += 1;

        let embed_w = self.weight_f16(EMBED_OFFSET, 17667 * 512 * 2, &[VOCAB, D_MODEL]);
        let mut hidden: Vec<f32> = embed_w.data[token_id as usize * D_MODEL..(token_id as usize + 1) * D_MODEL].to_vec();

        let norm_w = self.weight_f16(NORM_WEIGHT_OFFSET, 1024, &[D_MODEL]);
        let norm_w = &norm_w.data[..D_MODEL];
        let qk_norm_w = self.weight_f16(QK_NORM_OFFSET, 256, &[D_HEAD]);
        let qk_norm_w = &qk_norm_w.data[..D_HEAD];

        let pos_f = pos as f32;
        let mut cos = [0.0f32; D_HEAD];
        let mut sin = [0.0f32; D_HEAD];
        for i in 0..(D_HEAD / 2) {
            let freq = pos_f * cache.inv_freq[i];
            let (s, c) = freq.sin_cos();
            cos[i] = c;
            cos[i + D_HEAD / 2] = c;
            sin[i] = s;
            sin[i + D_HEAD / 2] = s;
        }

        for layer in 0..N_LAYERS {
            hidden = self.forward_layer_single(
                &hidden, pos, layer, norm_w, qk_norm_w,
                &mut cache.k_cache[layer], &mut cache.v_cache[layer],
                &cos, &sin, cache.attn_scale,
            );
        }

        self.rms_norm_inplace(&mut hidden, norm_w, LN_EPS);
        let lm_head = self.weight_f16(LM_HEAD_OFFSET, VOCAB * D_MODEL * 2, &[D_MODEL, VOCAB]);
        let logits = self.matvec(&hidden, &lm_head.data, VOCAB, D_MODEL);
        logits
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_single(
        &self,
        hidden: &[f32],
        pos: usize,
        layer: usize,
        norm_w: &[f32],
        qk_norm_w: &[f32],
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
        cos: &[f32; D_HEAD],
        sin: &[f32; D_HEAD],
        attn_scale: f32,
    ) -> Vec<f32> {
        let residual = hidden.to_vec();

        let mut normed = hidden.to_vec();
        self.rms_norm_inplace(&mut normed, norm_w, LN_EPS);

        let qw = self.weight_f16(layer_q_off(layer), 524288, &[D_MODEL, D_MODEL]);
        let kw = self.weight_f16(layer_k_off(layer), 131072, &[D_MODEL, D_HEAD]);
        let vw = self.weight_f16(layer_v_off(layer), 131072, &[D_MODEL, D_HEAD]);
        let ow = self.weight_f16(layer_o_off(layer), 524288, &[D_MODEL, D_MODEL]);

        let mut q = self.matvec(&normed, &qw.data, D_MODEL, D_MODEL);
        let mut k = self.matvec(&normed, &kw.data, D_HEAD, D_MODEL);
        let v = self.matvec(&normed, &vw.data, D_HEAD, D_MODEL);

        for h in 0..N_HEADS {
            let hr = &mut q[h * D_HEAD..(h + 1) * D_HEAD];
            self.rms_norm_inplace(hr, qk_norm_w, LN_EPS);
        }
        self.rms_norm_inplace(&mut k, qk_norm_w, LN_EPS);

        for h in 0..N_HEADS {
            let base = h * D_HEAD;
            for d in 0..(D_HEAD / 2) {
                let idx0 = base + d;
                let idx1 = base + d + D_HEAD / 2;
                let q0 = q[idx0];
                let q1 = q[idx1];
                q[idx0] = q0 * cos[d] - q1 * sin[d];
                q[idx1] = q1 * cos[d] + q0 * sin[d];
            }
        }
        for d in 0..(D_HEAD / 2) {
            let k0 = k[d];
            let k1 = k[d + D_HEAD / 2];
            k[d] = k0 * cos[d] - k1 * sin[d];
            k[d + D_HEAD / 2] = k1 * cos[d] + k0 * sin[d];
        }
        k_cache.extend_from_slice(&k);
        v_cache.extend_from_slice(&v);

        let cached_len = pos + 1;
        let mut attn = vec![0.0f32; D_MODEL];
        for h in 0..N_HEADS {
            let qh = &q[h * D_HEAD..(h + 1) * D_HEAD];
            let mut scores = vec![0.0f32; cached_len];
            for tt in 0..cached_len {
                let kv = &k_cache[tt * D_HEAD..(tt + 1) * D_HEAD];
                let mut s = 0.0f32;
                for d in 0..D_HEAD {
                    s += qh[d] * kv[d];
                }
                scores[tt] = s * attn_scale;
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            for d in 0..D_HEAD {
                let mut acc = 0.0f32;
                for tt in 0..cached_len {
                    acc += scores[tt] * v_cache[tt * D_HEAD + d];
                }
                attn[h * D_HEAD + d] = acc;
            }
        }

        let proj = self.matvec(&attn, &ow.data, D_MODEL, D_MODEL);
        let mut out = vec![0.0f32; D_MODEL];
        for i in 0..D_MODEL {
            out[i] = proj[i] + residual[i];
        }

        let residual2 = out.clone();
        self.rms_norm_inplace(&mut out, norm_w, LN_EPS);

        let gw = self.weight_f16(layer_gate_off(layer), 2359296, &[D_MODEL, D_FF]);
        let uw = self.weight_f16(layer_up_off(layer), 2359296, &[D_MODEL, D_FF]);
        let dw = self.weight_f16(layer_down_off(layer), 2359296, &[D_FF, D_MODEL]);

        let gate = self.matvec(&out, &gw.data, D_FF, D_MODEL);
        let up = self.matvec(&out, &uw.data, D_FF, D_MODEL);
        let mut mid = vec![0.0f32; D_FF];
        for i in 0..D_FF {
            let sig = 1.0 / (1.0 + (-gate[i]).exp());
            mid[i] = gate[i] * sig * up[i];
        }
        let down = self.matvec(&mid, &dw.data, D_MODEL, D_FF);
        for i in 0..D_MODEL {
            out[i] = down[i] + residual2[i];
        }

        out
    }
}
