//! Delayed-streams LM of `kyutai/tts-1.6b-en_fr`, batch size 1, one step at a time.
//!
//! Layout follows the Python checkpoint (`dsm_tts_*.safetensors`):
//! main transformer (16 layers, self-attn with RoPE + cross-attn on speaker
//! embeddings + gated MLP), demuxed text embedding, 32 audio embeddings, and a
//! depformer (4 layers) whose linear weights are shared across codebook steps
//! according to `depformer_weights_per_step_schedule`.
use crate::config::Config;
use anyhow::{anyhow, bail, Result};
use candle::quantized::{GgmlDType, QMatMul, QTensor};
use candle::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{ops, Linear, VarBuilder};

const EPS: f64 = 1e-5;

/// Linear layer without bias: dense, or a quantized matmul (weights are
/// quantized on the CPU and uploaded once, activations stay F32).
enum Lin {
    Dense(Linear),
    Quant(QMatMul),
}

/// Where and how to quantize the large matrices (`None` keeps them dense).
#[derive(Clone)]
pub struct QuantPlan {
    pub dtype: GgmlDType,
    pub device: Device,
}

impl Lin {
    fn new(w: Tensor, q: Option<&QuantPlan>) -> Result<Self> {
        match q {
            None => Ok(Self::Dense(Linear::new(w, None))),
            Some(plan) => {
                let w = w.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?;
                let qt = QTensor::quantize_onto(&w, plan.dtype, &plan.device)?;
                Ok(Self::Quant(QMatMul::from_qtensor(qt)?))
            }
        }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Dense(l) => l.forward(x)?,
            Self::Quant(q) => q.forward(x)?,
        })
    }
}

/// `-1` is the "zero token": its embedding is exactly zero.
pub const ZERO_TOKEN: i64 = -1;

struct RmsNorm {
    alpha: Tensor,        // [d], f32 (reference path)
    alpha_native: Tensor, // [d], checkpoint dtype (fused bf16 path)
}

impl RmsNorm {
    fn load(vb: &VarBuilder, dim: usize) -> Result<Self> {
        let raw = vb.get((1, 1, dim), "alpha")?.reshape(dim)?;
        let alpha = raw.to_dtype(DType::F32)?;
        Ok(Self { alpha, alpha_native: raw })
    }
    /// Computed in F32 whatever the activation dtype: the sum of squares of a
    /// residual stream with large-magnitude channels overflows F16 (|x| > 256).
    /// On Vulkan with bf16 activations, one bf16-native dispatch (f32 math inside).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(out) = x.rms_norm_fused(&self.alpha_native, EPS)? {
            return Ok(out);
        }
        let dt = x.dtype();
        let x32 = x.to_dtype(DType::F32)?.contiguous()?;
        Ok(ops::rms_norm(&x32, &self.alpha, EPS as f32)?.to_dtype(dt)?)
    }
}

/// `(a + b, lin(RMSNorm(a + b)))` for one row, as residual add, RMSNorm, then the projection.
fn add_norm_linear(a: &Tensor, b: &Tensor, norm: &RmsNorm, lin: &Lin) -> Result<(Tensor, Tensor)> {
    let (s, n) = add_norm(a, b, norm)?;
    Ok((s, lin.forward(&n)?))
}

/// `lin(silu(y[:h]) * y[h:])` for one row: one fused dispatch (gate read inside the GEMV) on Vulkan,
/// else the fused silu-gate kernel followed by the projection.
fn gated_linear(y: &Tensor, lin: &Lin) -> Result<Tensor> {
    if let Lin::Dense(l) = lin {
        if let Some(out) = y.gated_linear_fused(l.weight())? {
            return Ok(out);
        }
    }
    let h = y.dim(D::Minus1)? / 2;
    match y.silu_gate_fused(h)? {
        Some(g) => Ok(lin.forward(&g)?),
        None => {
            let a = y.narrow(D::Minus1, 0, h)?;
            let b = y.narrow(D::Minus1, h, h)?;
            Ok(lin.forward(&(ops::silu(&a)? * b)?)?)
        }
    }
}

/// `x + h` and `norm(x + h)`: one fused bf16 dispatch on Vulkan, the composition elsewhere.
fn add_norm(x: &Tensor, h: &Tensor, norm: &RmsNorm) -> Result<(Tensor, Tensor)> {
    if let Some(pair) = x.add_rms_norm_fused(h, &norm.alpha_native, EPS)? {
        return Ok(pair);
    }
    let s = (x + h)?;
    let n = norm.forward(&s)?;
    Ok((s, n))
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn load(vb: &VarBuilder, dim: usize) -> Result<Self> {
        Ok(Self { weight: vb.get(dim, "weight")?, bias: vb.get(dim, "bias")? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // bf16 on Vulkan: centre in place, then one fused bf16 RMSNorm (mean(xc^2) is the
        // variance), then the bias. Falls through to the reference path when declined.
        if x.dtype() == DType::BF16 && self.weight.dtype() == DType::BF16 {
            let xc = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
            if let Some(y) = xc.rms_norm_fused(&self.weight, EPS)? {
                return Ok(y.broadcast_add(&self.bias)?);
            }
        }
        let dt = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let y = xc.broadcast_div(&(var + EPS)?.sqrt()?)?;
        let y = y
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_add(&self.bias.to_dtype(DType::F32)?)?;
        Ok(y.to_dtype(dt)?)
    }
}

/// Gated feed-forward: `linear_out(act(a) * b)` with `[a, b] = linear_in(x)`.
struct Gating {
    /// One projection to `[a | b]`; the gate and value halves are read by the fused
    /// gated-activation kernel on Vulkan, with no split or narrow copy per step.
    linear_in: Lin,
    linear_out: Lin,
    hidden: usize,
}

impl Gating {
    fn new(linear_in: Tensor, linear_out: Tensor, hidden: usize, q: Option<&QuantPlan>) -> Result<Self> {
        Ok(Self { linear_in: Lin::new(linear_in, q)?, linear_out: Lin::new(linear_out, q)?, hidden })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.linear_in.forward(x)?;
        // silu(a) * b with [a | b] = y: one fused bf16 dispatch on Vulkan, the composition elsewhere.
        let h = match y.silu_gate_fused(self.hidden)? {
            Some(h) => h,
            None => {
                let a = y.narrow(D::Minus1, 0, self.hidden)?;
                let b = y.narrow(D::Minus1, self.hidden, self.hidden)?;
                (ops::silu(&a)? * b)?
            }
        };
        Ok(self.linear_out.forward(&h)?)
    }
}

/// Interleaved rotary embedding tables for positions `0..max_pos`.
struct Rope {
    cos: Tensor, // [max_pos, d/2]
    sin: Tensor,
}

impl Rope {
    fn new(head_dim: usize, max_pos: usize, max_period: f64, dev: &Device, dt: DType) -> Result<Self> {
        let half = head_dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (i as f64 * (-max_period.ln() * 2.0 / head_dim as f64)).exp() as f32)
            .collect();
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for t in 0..max_pos {
            for f in &freqs {
                let a = f * t as f32;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (max_pos, half), dev)?.to_dtype(dt)?,
            sin: Tensor::from_vec(sin, (max_pos, half), dev)?.to_dtype(dt)?,
        })
    }
    /// `q`, `k`: `[1, h, 1, d]` at absolute position `pos`.
    fn apply(&self, q: &Tensor, k: &Tensor, pos: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, pos, 1)?;
        let sin = self.sin.narrow(0, pos, 1)?;
        Ok((
            candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

/// Single-token attention over a growing K/V cache.
/// Self-attention K/V cache. On Vulkan with bf16, each token's rows are written in place into
/// preallocated `(1, heads, max_len, d)` buffers by one fused dispatch and read back as strided
/// views. Otherwise, and once the window is full, the reference cat-based sliding window runs.
struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    /// Rows written in the preallocated buffers since the last reset.
    len: usize,
    max_len: usize,
    /// Reference path: the current sliding window.
    window: Option<(Tensor, Tensor)>,
    /// Decided by the first append and kept across resets: `Some(true)` = fused buffers in use.
    fused: Option<bool>,
}

impl KvCache {
    fn new(max_len: usize) -> Self {
        Self { k: None, v: None, len: 0, max_len, window: None, fused: None }
    }
    fn reset(&mut self) {
        self.len = 0;
        self.window = None;
    }
    /// Fused rotary + append for one token from the packed `[q | k | v]` row. `None` when the fused path
    /// does not apply; the caller then runs the reference split, rope and append.
    fn append_rope(&mut self, qkv: &Tensor, rope: &Rope, pos: usize, heads: usize, head_dim: usize) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        if self.window.is_some() || self.fused == Some(false) || self.len >= self.max_len {
            return Ok(None);
        }
        if self.k.is_none() {
            self.k = Some(Tensor::zeros((1, heads, self.max_len, head_dim), qkv.dtype(), qkv.device())?);
            self.v = Some(Tensor::zeros((1, heads, self.max_len, head_dim), qkv.dtype(), qkv.device())?);
        }
        let q = match (&self.k, &self.v) {
            (Some(bk), Some(bv)) => qkv.rope_kv_append_fused(&rope.cos, &rope.sin, pos, heads, head_dim, bk, bv, self.len, true)?,
            _ => None,
        };
        let Some(q) = q else {
            return Ok(None);
        };
        self.fused = Some(true);
        self.len += 1;
        match (&self.k, &self.v) {
            (Some(bk), Some(bv)) => Ok(Some((q, bk.narrow(2, 0, self.len)?, bv.narrow(2, 0, self.len)?))),
            _ => bail!("kv cache: buffers missing after a fused rope append"),
        }
    }
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.window.is_none() && self.fused != Some(false) && self.len < self.max_len {
            if self.k.is_none() {
                let (_, h, _, d) = k.dims4()?;
                self.k = Some(Tensor::zeros((1, h, self.max_len, d), k.dtype(), k.device())?);
                self.v = Some(Tensor::zeros((1, h, self.max_len, d), v.dtype(), v.device())?);
            }
            let written = match (&self.k, &self.v) {
                (Some(bk), Some(bv)) => k.kv_append_fused(v, bk, bv, self.len)?,
                _ => false,
            };
            if written {
                self.fused = Some(true);
                self.len += 1;
                return match (&self.k, &self.v) {
                    (Some(bk), Some(bv)) => Ok((bk.narrow(2, 0, self.len)?, bv.narrow(2, 0, self.len)?)),
                    _ => bail!("kv cache: buffers missing after a fused append"),
                };
            }
            if self.len > 0 {
                bail!("kv cache: fused append declined mid-sequence");
            }
            self.fused = Some(false);
            self.k = None;
            self.v = None;
        }
        if self.window.is_none() && self.fused == Some(true) {
            // Buffer full: the sliding window continues on the reference path from here.
            match (&self.k, &self.v) {
                (Some(bk), Some(bv)) => self.window = Some((bk.contiguous()?, bv.contiguous()?)),
                _ => bail!("kv cache: buffers missing at window seed"),
            }
        }
        let (mut nk, mut nv) = match &self.window {
            Some((pk, pv)) => (Tensor::cat(&[pk, k], 2)?, Tensor::cat(&[pv, v], 2)?),
            None => (k.clone(), v.clone()),
        };
        let len = nk.dim(2)?;
        if len > self.max_len {
            nk = nk.narrow(2, len - self.max_len, self.max_len)?.contiguous()?;
            nv = nv.narrow(2, len - self.max_len, self.max_len)?.contiguous()?;
        }
        self.len = nk.dim(2)?;
        self.window = Some((nk.clone(), nv.clone()));
        Ok((nk, nv))
    }
}

fn attend(q: &Tensor, k: &Tensor, v: &Tensor, head_dim: usize) -> Result<Tensor> {
    // q: [1,h,1,d], k/v: [1,h,T,d]
    // One fused dispatch on Vulkan (bf16); the composition below is the reference path.
    if let Some(out) = q.decode_attention_fused(k, v, 1.0 / (head_dim as f32).sqrt())? {
        return Ok(out);
    }
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    let probs = ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
    Ok(probs.matmul(&v.contiguous()?)?)
}

fn split_heads(x: &Tensor, heads: usize, head_dim: usize) -> Result<Tensor> {
    // [1,1,h*d] -> [1,h,1,d]
    Ok(x.reshape((1, 1, heads, head_dim))?.transpose(1, 2)?.contiguous()?)
}

fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (_, h, t, d) = x.dims4()?;
    Ok(x.transpose(1, 2)?.contiguous()?.reshape((1, t, h * d))?)
}

struct MainLayer {
    norm1: RmsNorm,
    norm_cross: LayerNorm,
    norm2: RmsNorm,
    in_proj: Lin, // fused qkv
    out_proj: Lin,
    cross_q: Lin,
    cross_out: Lin,
    cross_kv: Lin, // [2*dim, dim], used when the voice changes
    gating: Gating,
    cache: KvCache,
    cross_k: Option<Tensor>,
    cross_v: Option<Tensor>,
}

struct DepLayer {
    norm1: RmsNorm,
    norm2: RmsNorm,
    in_proj: Vec<Lin>,
    out_proj: Vec<Lin>,
    gating: Vec<Gating>,
    cache: KvCache,
}

/// `text_emb` / `depformer_text_emb`: two text streams multiplexed in one id
/// (`(second + 1) * card + main`), demuxed through `out1` / `out2`.
struct DemuxEmbedding {
    weight: Tensor,
    out1: Linear,
    out2: Linear,
    card: usize,
}

impl DemuxEmbedding {
    fn forward(&self, token: i64, dev: &Device, dt: DType) -> Result<Tensor> {
        let card = self.card as i64;
        let left = token.rem_euclid(card);
        let right = token.div_euclid(card) - 1;
        let l = self.weight.i(left as usize)?.unsqueeze(0)?;
        let mut y = self.out1.forward(&l)?;
        if right >= 0 {
            let r = self.weight.i(right as usize)?.unsqueeze(0)?;
            y = (y + self.out2.forward(&r)?)?;
        }
        let _ = (dev, dt);
        Ok(y.reshape((1, 1, ()))?)
    }
}

pub struct Lm {
    pub cfg: Config,
    dev: Device,
    dt: DType,
    emb_all: Tensor, // [(n_q) * (card+1) + 1, dim], last row is zeros
    text_emb: DemuxEmbedding,
    out_norm: RmsNorm,
    text_linear: Lin,
    layers: Vec<MainLayer>,
    rope: Rope,
    sum_cond: Tensor, // [1,1,dim]
    dep_in: Vec<Lin>,
    dep_emb: Vec<(Tensor, Linear)>, // low-rank embeddings, index = cb-1
    dep_text_emb: DemuxEmbedding,
    dep_layers: Vec<DepLayer>,
    dep_linears: Vec<Lin>,
    offset: usize,
}

fn lin(vb: &VarBuilder, out: usize, inn: usize, name: &str) -> Result<Linear> {
    Ok(Linear::new(vb.get((out, inn), name)?, None))
}

/// Big matrix: loaded from `src` (CPU, F32 when quantizing) and optionally quantized.
fn big(src: &VarBuilder, out: usize, inn: usize, name: &str, q: Option<&QuantPlan>) -> Result<Lin> {
    Lin::new(src.get((out, inn), name)?, q)
}

impl Lm {
    pub fn load(
        cfg: &Config,
        vb: VarBuilder,
        weights_src: &VarBuilder,
        quant: Option<&QuantPlan>,
        dev: &Device,
        dt: DType,
        cond: &CondInputs,
    ) -> Result<Self> {
        // `weights_src` supplies the large matrices (a CPU F32 builder when quantizing).
        // KYUTAI_Q_SKIP=cross,heads,main,dep keeps those groups dense (sensitivity analysis).
        let skip = std::env::var("KYUTAI_Q_SKIP").unwrap_or_default();
        let q_for = |group: &str| if skip.split(',').any(|g| g == group) { None } else { quant };
        let (q_main, q_cross, q_heads, q_dep) = (q_for("main"), q_for("cross"), q_for("heads"), q_for("dep"));
        let dim = cfg.dim;
        let hd = dim / cfg.num_heads;
        let n_rows = cfg.card + 1;

        // Audio embeddings, concatenated so one gather fetches all 32 rows per step.
        let mut parts = Vec::new();
        for k in 0..cfg.n_q {
            parts.push(vb.pp(format!("emb.{k}")).get((n_rows, dim), "weight")?);
        }
        parts.push(Tensor::zeros((1, dim), dt, dev)?);
        let emb_all = Tensor::cat(&parts, 0)?;
        drop(parts);

        let te = vb.pp("text_emb");
        let text_emb = DemuxEmbedding {
            weight: te.get((cfg.text_card + 1, dim), "weight")?,
            out1: lin(&te, dim, dim, "out1.weight")?,
            out2: lin(&te, dim, dim, "out2.weight")?,
            card: cfg.text_card + 1,
        };

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            let lv = vb.pp(format!("transformer.layers.{i}"));
            let sv = weights_src.pp(format!("transformer.layers.{i}"));
            let ca_w = sv.get((3 * dim, dim), "cross_attention.in_proj_weight")?;
            layers.push(MainLayer {
                norm1: RmsNorm::load(&lv.pp("norm1"), dim)?,
                norm_cross: LayerNorm::load(&lv.pp("norm_cross"), dim)?,
                norm2: RmsNorm::load(&lv.pp("norm2"), dim)?,
                in_proj: big(&sv, 3 * dim, dim, "self_attn.in_proj_weight", q_main)?,
                out_proj: big(&sv, dim, dim, "self_attn.out_proj.weight", q_main)?,
                cross_q: Lin::new(ca_w.narrow(0, 0, dim)?.contiguous()?, q_cross)?,
                cross_out: big(&sv, dim, dim, "cross_attention.out_proj.weight", q_cross)?,
                cross_kv: Lin::new(ca_w.narrow(0, dim, 2 * dim)?.contiguous()?, q_cross)?,
                gating: Gating::new(
                    sv.get((2 * cfg.hidden_ff, dim), "gating.linear_in.weight")?,
                    sv.get((dim, cfg.hidden_ff), "gating.linear_out.weight")?,
                    cfg.hidden_ff,
                    q_main,
                )?,
                cache: KvCache::new(cfg.context),
                cross_k: None,
                cross_v: None,
            });
        }

        let ddim = cfg.dep_dim;
        let nw = cfg.num_weight_sets();
        let mut dep_in = Vec::new();
        for i in 0..nw {
            dep_in.push(big(&weights_src.pp(format!("depformer_in.{i}")), ddim, dim, "weight", q_dep)?);
        }
        let mut dep_emb = Vec::new();
        for i in 0..cfg.dep_q - 1 {
            let ev = vb.pp(format!("depformer_emb.{i}"));
            dep_emb.push((
                ev.get((n_rows, cfg.dep_low_rank), "weight")?,
                lin(&ev, ddim, cfg.dep_low_rank, "low_rank.weight")?,
            ));
        }
        let dtv = vb.pp("depformer_text_emb");
        let dep_text_emb = DemuxEmbedding {
            weight: dtv.get((cfg.text_card + 1, cfg.dep_low_rank), "weight")?,
            out1: lin(&dtv, ddim, cfg.dep_low_rank, "out1.weight")?,
            out2: lin(&dtv, ddim, cfg.dep_low_rank, "out2.weight")?,
            card: cfg.text_card + 1,
        };
        let mut dep_layers = Vec::new();
        for i in 0..cfg.dep_layers {
            let lv = vb.pp(format!("depformer.layers.{i}"));
            let sv = weights_src.pp(format!("depformer.layers.{i}"));
            let in_all = sv.get((nw * 3 * ddim, ddim), "self_attn.in_proj_weight")?;
            let out_all = sv.get((nw * ddim, ddim), "self_attn.out_proj.weight")?;
            let mut in_proj = Vec::new();
            let mut out_proj = Vec::new();
            let mut gating = Vec::new();
            for w in 0..nw {
                in_proj.push(Lin::new(in_all.narrow(0, w * 3 * ddim, 3 * ddim)?.contiguous()?, q_dep)?);
                out_proj.push(Lin::new(out_all.narrow(0, w * ddim, ddim)?.contiguous()?, q_dep)?);
                let gv = sv.pp(format!("gating.{w}"));
                gating.push(Gating::new(
                    gv.get((2 * cfg.dep_hidden_ff, ddim), "linear_in.weight")?,
                    gv.get((ddim, cfg.dep_hidden_ff), "linear_out.weight")?,
                    cfg.dep_hidden_ff,
                    q_dep,
                )?);
            }
            dep_layers.push(DepLayer {
                norm1: RmsNorm::load(&lv.pp("norm1"), ddim)?,
                norm2: RmsNorm::load(&lv.pp("norm2"), ddim)?,
                in_proj,
                out_proj,
                gating,
                cache: KvCache::new(cfg.dep_q),
            });
        }
        let mut dep_linears = Vec::new();
        for i in 0..cfg.dep_q {
            dep_linears.push(big(&weights_src.pp(format!("linears.{i}")), cfg.card, ddim, "weight", q_heads)?);
        }

        let mut lm = Self {
            cfg: cfg.clone(),
            dev: dev.clone(),
            dt,
            emb_all,
            text_emb,
            out_norm: RmsNorm::load(&vb.pp("out_norm"), dim)?,
            text_linear: big(weights_src, cfg.text_card, dim, "text_linear.weight", q_heads)?,
            layers,
            rope: Rope::new(hd, 8192, cfg.max_period, dev, dt)?,
            sum_cond: cond.sum_cond.to_dtype(dt)?,
            dep_in,
            dep_emb,
            dep_text_emb,
            dep_layers,
            dep_linears,
            offset: 0,
        };
        lm.set_cross_attention_source(&cond.cross_src.to_dtype(dt)?)?;
        let _ = hd;
        Ok(lm)
    }

    /// Swap the voice / CFG conditioning (does not reload weights).
    pub fn set_conditions(&mut self, cond: &CondInputs) -> Result<()> {
        self.sum_cond = cond.sum_cond.to_dtype(self.dt)?;
        self.set_cross_attention_source(&cond.cross_src.to_dtype(self.dt)?)
    }

    fn set_cross_attention_source(&mut self, src: &Tensor) -> Result<()> {
        let h = self.cfg.num_heads;
        let hd = self.cfg.dim / h;
        let t = src.dim(1)?;
        for l in &mut self.layers {
            let kv = l.cross_kv.forward(src)?; // [1,T,2*dim]
            let k = kv.narrow(D::Minus1, 0, self.cfg.dim)?;
            let v = kv.narrow(D::Minus1, self.cfg.dim, self.cfg.dim)?;
            l.cross_k = Some(k.reshape((1, t, h, hd))?.transpose(1, 2)?.contiguous()?);
            l.cross_v = Some(v.reshape((1, t, h, hd))?.transpose(1, 2)?.contiguous()?);
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        for l in &mut self.layers {
            l.cache.reset();
        }
    }

    /// One main-transformer step. `audio` holds `n_q` tokens (`ZERO_TOKEN` for
    /// an all-zero embedding, `card` for the initial token); `text` is the (muxed)
    /// text token. Returns `(text_logits [text_card], transformer_out [1,1,dim])`.
    pub fn forward_text(&mut self, text: i64, audio: &[i64]) -> Result<(Tensor, Tensor)> {
        let cfg = &self.cfg;
        let n_rows = (cfg.card + 1) as i64;
        let zero_row = (cfg.n_q as i64) * n_rows;
        let idx: Vec<u32> = audio
            .iter()
            .enumerate()
            .map(|(k, &t)| if t < 0 { zero_row as u32 } else { (k as i64 * n_rows + t) as u32 })
            .collect();
        let idx = Tensor::from_vec(idx, cfg.n_q, &self.dev)?;
        let audio_emb = self.emb_all.index_select(&idx, 0)?.sum(0)?.reshape((1, 1, cfg.dim))?;
        let mut x = (audio_emb + self.text_emb.forward(text, &self.dev, self.dt)?)?;
        x = x.broadcast_add(&self.sum_cond)?;

        let (h, hd) = (cfg.num_heads, cfg.dim / cfg.num_heads);
        // Each residual add is fused with the projection that consumes its RMSNorm (one dispatch on
        // Vulkan). The MLP output is carried as `pending` and added at the next layer's norm1.
        let mut pending = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
        for li in 0..self.layers.len() {
            let l = &mut self.layers[li];
            // self-attention. The main in_proj stays on the composition: its fused GEMV differs from
            // the reference by one bf16 ULP on some in-model rows, which changes the audio.
            let (xr, xn) = add_norm(&x, &pending, &l.norm1)?;
            x = xr;
            let qkv = l.in_proj.forward(&xn)?;
            // Rotary + KV append in one dispatch on Vulkan; the reference split/rope/append otherwise.
            let (q, kc, vc) = match l.cache.append_rope(&qkv, &self.rope, self.offset, h, hd)? {
                Some(t) => t,
                None => {
                    let q = split_heads(&qkv.narrow(D::Minus1, 0, cfg.dim)?, h, hd)?;
                    let k = split_heads(&qkv.narrow(D::Minus1, cfg.dim, cfg.dim)?, h, hd)?;
                    let v = split_heads(&qkv.narrow(D::Minus1, 2 * cfg.dim, cfg.dim)?, h, hd)?;
                    let (q, k) = self.rope.apply(&q, &k, self.offset)?;
                    let (kc, vc) = l.cache.append(&k, &v)?;
                    (q, kc, vc)
                }
            };
            let a = merge_heads(&attend(&q, &kc, &vc, hd)?)?;
            x = (x + l.out_proj.forward(&a)?)?;
            // cross-attention on speaker embeddings
            let xc = l.norm_cross.forward(&x)?;
            let q = split_heads(&l.cross_q.forward(&xc)?, h, hd)?;
            let ck = l.cross_k.as_ref().ok_or_else(|| anyhow!("cross-attention source not set"))?;
            let cv = l.cross_v.as_ref().ok_or_else(|| anyhow!("cross-attention source not set"))?;
            let a = merge_heads(&attend(&q, ck, cv, hd)?)?;
            let co = l.cross_out.forward(&a)?;
            // feed-forward: residual + norm2 + linear_in, then the gated linear_out
            let (xr2, y) = add_norm_linear(&x, &co, &l.norm2, &l.gating.linear_in)?;
            x = xr2;
            pending = gated_linear(&y, &l.gating.linear_out)?;
        }
        x = (x + pending)?;
        self.offset += 1;
        let out = self.out_norm.forward(&x)?;
        let logits = self.text_linear.forward(&out)?.reshape(cfg.text_card)?;
        Ok((logits, out))
    }

    /// Start a depformer pass for the current timestep.
    pub fn depformer_begin(&mut self) {
        for l in &mut self.dep_layers {
            l.cache.reset();
        }
    }

    /// Logits for codebook `cb`. `prev` is the muxed text token for `cb == 0`,
    /// otherwise the token sampled for codebook `cb - 1`.
    pub fn depformer_step(&mut self, cb: usize, prev: i64, transformer_out: &Tensor) -> Result<Tensor> {
        let cfg = &self.cfg;
        let w = cfg.dep_schedule[cb];
        let (h, hd) = (cfg.dep_heads, cfg.dep_dim / cfg.dep_heads);
        let e = || candle::Device::encode_count();
        let mut mark = e();
        let mut tick = |slot: usize, mark: &mut u64| {
            let now = e();
            SECTION_DISPATCHES[slot].fetch_add(now - *mark, std::sync::atomic::Ordering::Relaxed);
            *mark = now;
        };
        let mut x = self.dep_in[w].forward(transformer_out)?;
        let tok = if cb == 0 {
            self.dep_text_emb.forward(prev, &self.dev, self.dt)?
        } else if prev < 0 {
            Tensor::zeros((1, 1, cfg.dep_dim), self.dt, &self.dev)?
        } else {
            let (emb, low) = &self.dep_emb[cb - 1];
            low.forward(&emb.i(prev as usize)?.unsqueeze(0)?)?.reshape((1, 1, ()))?
        };
        x = (x + tok)?;
        tick(0, &mut mark);
        let n_dep = self.dep_layers.len();
        // The MLP output is carried as `pending` and added inside the next projection's dispatch.
        let mut pending = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
        for li in 0..n_dep {
            let l = &mut self.dep_layers[li];
            tick(1, &mut mark);
            let (xr, qkv) = add_norm_linear(&x, &pending, &l.norm1, &l.in_proj[w])?;
            x = xr;
            tick(2, &mut mark);
            let d = cfg.dep_dim;
            let q = split_heads(&qkv.narrow(D::Minus1, 0, d)?, h, hd)?;
            let k = split_heads(&qkv.narrow(D::Minus1, d, d)?, h, hd)?;
            let v = split_heads(&qkv.narrow(D::Minus1, 2 * d, d)?, h, hd)?;
            tick(3, &mut mark);
            let (kc, vc) = l.cache.append(&k, &v)?;
            tick(4, &mut mark);
            let a = merge_heads(&attend(&q, &kc, &vc, hd)?)?;
            tick(5, &mut mark);
            let o = l.out_proj[w].forward(&a)?;
            tick(6, &mut mark);
            let (xr2, y) = add_norm_linear(&x, &o, &l.norm2, &l.gating[w].linear_in)?;
            x = xr2;
            tick(7, &mut mark);
            pending = gated_linear(&y, &l.gating[w].linear_out)?;
            tick(8, &mut mark);
        }
        x = (x + pending)?;
        let out = self.dep_linears[cb].forward(&x)?.reshape(cfg.card)?;
        tick(9, &mut mark);
        Ok(out)
    }
}

/// GPU dispatches per depformer section (diagnostic): 0 head, 1 norm1, 2 qkv,
/// 3 head split, 4 cache append, 5 attention, 6 out proj + residual, 7 norm2,
/// 8 gated FFN + residual, 9 output head.
pub static SECTION_DISPATCHES: [std::sync::atomic::AtomicU64; 10] = [const { std::sync::atomic::AtomicU64::new(0) }; 10];

/// Conditioning tensors computed once per (voice, cfg) request.
pub struct CondInputs {
    /// `[1,1,dim]`: sum of the `control` and `cfg` LUT conditioners.
    pub sum_cond: Tensor,
    /// `[1, n_speakers * T, dim]`: speaker embeddings (+ sinusoidal positions).
    pub cross_src: Tensor,
}

fn sin_embedding(len: usize, dim: usize, dev: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let mut out = vec![0f32; len * dim];
    for p in 0..len {
        for i in 0..half {
            let phase = p as f64 / 10000f64.powf(i as f64 / (half - 1) as f64);
            out[p * dim + i] = phase.cos() as f32;
            out[p * dim + half + i] = phase.sin() as f32;
        }
    }
    Ok(Tensor::from_vec(out, (1, len, dim), dev)?)
}

/// Build the conditioning inputs from the checkpoint's `condition_provider.*`
/// weights, a voice embedding (`speaker_wavs`, `[1, 512, T]`) and a CFG value.
pub fn build_conditions(
    cfg: &Config,
    vb: &VarBuilder,
    voice: &Tensor,
    cfg_value: &str,
    dev: &Device,
) -> Result<CondInputs> {
    let dim = cfg.dim;
    let cp = vb.pp("condition_provider.conditioners");
    // LUT conditioners: embed(idx) -> output_proj.
    let lut = |name: &str, n_bins: usize, emb_dim: usize, idx: u32| -> Result<Tensor> {
        let c = cp.pp(name);
        let e = c.get((n_bins + 1, emb_dim), "embed.weight")?;
        let e = e.index_select(&Tensor::new(&[idx], dev)?, 0)?; // [1, emb_dim]
        Ok(Linear::new(c.get((dim, emb_dim), "output_proj.weight")?, None)
            .forward(&e)?
            .reshape((1, 1, dim))?)
    };
    let cfg_idx = cfg
        .cfg_values
        .iter()
        .position(|v| v == cfg_value)
        .ok_or_else(|| anyhow!("unsupported cfg value {cfg_value}; valid: {:?}", cfg.cfg_values))? as u32;
    let sum_cond = (lut("control", 1, dim, 0)? + lut("cfg", cfg.cfg_values.len(), 16, cfg_idx)?)?;

    // Speaker embeddings -> output_proj; unused speaker slots use learnt_padding.
    let sp = cp.pp("speaker_wavs");
    let proj = Linear::new(sp.get((dim, cfg.speaker_dim), "output_proj.weight")?, None);
    let voice = voice.to_device(dev)?.to_dtype(sp.dtype())?; // [1,512,T]
    let t = voice.dim(2)?;
    let spk = proj.forward(&voice.transpose(1, 2)?.contiguous()?)?; // [1,T,dim]
    let pad = sp.get((1, 1, dim), "learnt_padding")?;
    let mut parts = vec![spk];
    for _ in 1..cfg.max_speakers {
        parts.push(pad.broadcast_as((1, t, dim))?.contiguous()?);
    }
    let cross = Tensor::cat(&parts, 1)?;
    let pos = sin_embedding(cross.dim(1)?, dim, dev)?.to_dtype(cross.dtype())?;
    Ok(CondInputs { sum_cond, cross_src: (cross + pos)? })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
        let d = (a - b).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let n = b.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        (d / n).sqrt()
    }

    #[test]
    fn q8_linear_matches_dense_within_quantization_error() {
        let dev = Device::Cpu;
        for (out, inn) in [(2048usize, 2048usize), (11264, 2048), (2048, 5632), (2048, 1024)] {
            // Weights/activations with LLM-like scale: N(0, 0.02) weights, N(0,1) inputs.
            let w = (Tensor::randn(0f32, 1f32, (out, inn), &dev).unwrap() * 0.02).unwrap();
            let x = Tensor::randn(0f32, 1f32, (1, 1, inn), &dev).unwrap();
            let plan = QuantPlan { dtype: GgmlDType::Q8_0, device: dev.clone() };
            let q = Lin::new(w.clone(), Some(&plan)).unwrap();
            let d = Lin::new(w, None).unwrap();
            let e = rel_err(&q.forward(&x).unwrap(), &d.forward(&x).unwrap());
            assert!(e < 0.02, "({out},{inn}) relative error {e}");
        }
    }
}
