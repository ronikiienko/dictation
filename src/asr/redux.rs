//! Parakeet Redux (moondream/parakeet-redux: Parakeet TDT 0.6B with a ternary encoder) -> our weights file.
//! Download: `model.safetensors` (Hugging Face tensor names), `tokenizer.json`, `config.json`. Same architecture,
//! tokenizer (checked: identical 8192 pieces), pause head and frontend as Parakeet Ultra, so the metadata is Ultra's
//! (transcribe.cpp GGUF keys): sizes from config.json, pieces from tokenizer.json, frontend/pause/segmenter settings
//! as constants (Redux ships none; values as in Ultra's GGUF).
//! Storage: the 264 ternary modules (encoder linears + pointwise convs) as "t2" (2-bit codes + f32 scale per group of
//! 128; CPU only), or with `ternary_q8` as Ultra's kinds (q8 / f16, exact) so the GPU runs it (experiments only:
//! same memory and speed as Ultra there). Linears Ultra keeps as q8 (subsampler out, joint, predictor): Q8_0.
//! Everything else f32. LSTM: one bias = bias_ih + bias_hh (our decoder adds a single bias).
use super::gguf::Value;
use super::weights::{q8_bytes, write, Blob};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Values per ternary scale (config.json `ternary_group_size`; the "t2" kind assumes it).
pub const GROUP: usize = 128;

struct St {
    data: Vec<u8>,
    start: usize,
    header: serde_json::Map<String, Json>,
}

fn bad(m: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m)
}

impl St {
    fn open(path: &Path) -> io::Result<St> {
        let data = std::fs::read(path)?;
        let n = u64::from_le_bytes(data.get(..8).ok_or_else(|| bad("safetensors: too short".into()))?.try_into().unwrap()) as usize;
        let header = serde_json::from_slice(data.get(8..8 + n).ok_or_else(|| bad("safetensors: bad header".into()))?)?;
        Ok(St { data, start: 8 + n, header })
    }
    fn get(&self, name: &str) -> io::Result<(&str, Vec<usize>, &[u8])> {
        let t = self.header.get(name).ok_or_else(|| bad(format!("{name} not in safetensors")))?;
        let o = t["data_offsets"].as_array().ok_or_else(|| bad(format!("{name}: no offsets")))?;
        let (a, b) = (o[0].as_u64().unwrap_or(0) as usize, o[1].as_u64().unwrap_or(0) as usize);
        let shape = t["shape"].as_array().into_iter().flatten().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
        let bytes = self.data.get(self.start + a..self.start + b).ok_or_else(|| bad(format!("{name}: out of range")))?;
        Ok((t["dtype"].as_str().unwrap_or(""), shape, bytes))
    }
    fn f32(&self, name: &str) -> io::Result<(Vec<f32>, Vec<usize>)> {
        let (dt, shape, b) = self.get(name)?;
        let v = match dt {
            "F32" => b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect(),
            "F16" => b.chunks_exact(2).map(|x| half::f16::from_le_bytes([x[0], x[1]]).to_f32()).collect(),
            d => return Err(bad(format!("{name}: dtype {d}"))),
        };
        Ok((v, shape))
    }
    /// Ternary module: codes -1/0/+1 [rows][cols] (stored as base-3 digits, 5 per byte, lowest first), f32 scales
    /// [rows][cols / GROUP].
    fn ternary(&self, module: &str) -> io::Result<(Vec<i8>, Vec<f32>, usize, usize)> {
        let (_, qs, q) = self.get(&format!("{module}.qweight"))?;
        let (s, ss) = self.f32(&format!("{module}.scales"))?;
        let (rows, row_bytes, groups) = (qs[0], qs[1], ss[1]);
        let cols = groups * GROUP;
        if row_bytes != cols.div_ceil(5) {
            return Err(bad(format!("{module}: packed row length {row_bytes} for {cols} columns")));
        }
        let mut codes = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row = &q[r * row_bytes..(r + 1) * row_bytes];
            for i in 0..cols {
                codes.push(((row[i / 5] as u32 / 3u32.pow((i % 5) as u32)) % 3) as i8 - 1);
            }
        }
        Ok((codes, s, rows, cols))
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Q8_0 like GGUF: per block of 32, d = amax / 127 (stored f16), q = round(x / d).
fn quantize(v: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let mut q = Vec::with_capacity(v.len());
    let mut d = Vec::with_capacity(v.len() / 32);
    for b in v.chunks_exact(32) {
        let amax = b.iter().fold(0f32, |m, x| m.max(x.abs()));
        let s = half::f16::from_f32(amax / 127.0).to_f32();
        d.push(s);
        q.extend(b.iter().map(|x| if s == 0.0 { 0 } else { (x / s).round().clamp(-127.0, 127.0) as i8 }));
    }
    q8_bytes(q, &d, rows, cols)
}

/// "t2" bytes: codes + 1 (0..2), 4 per byte, lowest bits first, row-major; then the f32 scales.
fn t2_bytes(codes: &[i8], scales: &[f32]) -> Vec<u8> {
    let mut b: Vec<u8> = codes.chunks(4).map(|c| c.iter().enumerate().fold(0u8, |a, (j, &v)| a | (((v + 1) as u8) << (2 * j)))).collect();
    b.extend(f32_bytes(scales));
    b
}

/// Metadata in Ultra's keys (see the module comment).
fn meta(dir: &Path) -> io::Result<HashMap<String, Value>> {
    let cfg: Json = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let tok: Json = serde_json::from_slice(&std::fs::read(dir.join("tokenizer.json"))?)?;
    let num = |v: &Json, k: &str| v[k].as_i64().ok_or_else(|| bad(format!("config.json: no {k}")));
    let enc = &cfg["encoder_config"];
    let vocab = tok["model"]["vocab"].as_object().ok_or_else(|| bad("tokenizer.json: no vocab".into()))?;
    let mut pieces = vec![String::new(); vocab.len()];
    for (p, id) in vocab {
        let i = id.as_u64().unwrap_or(u64::MAX) as usize;
        *pieces.get_mut(i).ok_or_else(|| bad(format!("tokenizer.json: id {i}")))? = p.clone();
    }
    let blank = num(&cfg, "blank_token_id")?;
    if blank as usize != pieces.len() || num(&cfg, "vocab_size")? != blank + 1 {
        return Err(bad("tokenizer/config: blank is not the last class".into()));
    }
    pieces.push("<blank>".into());
    // 1 normal, 2 unknown (<unk> = 0), 3 control (<blank>), as in Ultra's GGUF.
    let types: Vec<Value> = (0..pieces.len()).map(|i| Value::Int(if i == 0 { 2 } else if i == pieces.len() - 1 { 3 } else { 1 })).collect();
    // GGUF floats are f32: keep the same rounding as Ultra's file.
    let f = |x: f32| Value::Float(x as f64);
    let s = |x: &str| Value::Str(x.into());
    let i = Value::Int;
    let mut m: HashMap<String, Value> = [
        ("general.architecture", s("parakeet")),
        ("general.name", s("Parakeet Redux")),
        ("general.repo_url", s("https://huggingface.co/moondream/parakeet-redux")),
        ("general.license", s("cc-by-4.0")),
        ("stt.variant", s("tdt-0.6b-redux")),
        ("stt.frontend.sample_rate", i(16000)),
        ("stt.frontend.n_fft", i(512)),
        ("stt.frontend.win_length", i(400)),
        ("stt.frontend.hop_length", i(160)),
        ("stt.frontend.num_mels", i(num(enc, "num_mel_bins")?)),
        ("stt.frontend.f_min", f(0.0)),
        ("stt.frontend.f_max", f(8000.0)),
        ("stt.frontend.pre_emphasis", f(0.97)),
        ("stt.parakeet.encoder.n_layers", i(num(enc, "num_hidden_layers")?)),
        ("stt.parakeet.encoder.subsampling_factor", i(num(enc, "subsampling_factor")?)),
        ("stt.parakeet.joint.hidden", i(num(&cfg, "decoder_hidden_size")?)),
        ("stt.parakeet.predictor.hidden", i(num(&cfg, "decoder_hidden_size")?)),
        ("stt.parakeet.predictor.n_layers", i(num(&cfg, "num_decoder_layers")?)),
        ("stt.parakeet.predictor.vocab", i(num(&cfg, "vocab_size")?)),
        ("stt.parakeet.tdt.max_symbols", i(num(&cfg, "max_symbols_per_step")?)),
        ("stt.parakeet.tdt.symbol_budget", s("global")),
        ("stt.parakeet.vad.speech_threshold", f(0.5)),
        ("stt.parakeet.vad.min_speech_seconds", f(0.1)),
        ("stt.parakeet.vad.min_gap_seconds", f(0.1)),
        ("stt.parakeet.segmenter.max_segment_seconds", f(30.0)),
        ("stt.parakeet.segmenter.min_segment_seconds", f(1.0)),
        ("stt.parakeet.segmenter.min_pause_seconds", f(0.2)),
        ("stt.parakeet.segmenter.scan_block_seconds", f(120.0)),
        ("tokenizer.ggml.model", s("bpe")),
        ("tokenizer.ggml.unknown_token_id", i(0)),
        ("tokenizer.ggml.blank_token_id", i(blank)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let durations = cfg["durations"].as_array().ok_or_else(|| bad("config.json: no durations".into()))?;
    m.insert("stt.parakeet.tdt.durations".into(), Value::Arr(durations.iter().map(|d| i(d.as_i64().unwrap_or(0))).collect()));
    m.insert("tokenizer.ggml.tokens".into(), Value::Arr(pieces.into_iter().map(Value::Str).collect()));
    m.insert("tokenizer.ggml.token_type".into(), Value::Arr(types));
    Ok(m)
}

/// `dir` with model.safetensors, tokenizer.json, config.json -> our weights file `dst`.
pub fn convert(dir: &Path, dst: &Path, ternary_q8: bool) -> io::Result<()> {
    let st = St::open(&dir.join("model.safetensors"))?;
    let meta = meta(dir)?;
    let layers = match meta["stt.parakeet.encoder.n_layers"] { Value::Int(n) => n as usize, _ => unreachable!() };
    let lstm = match meta["stt.parakeet.predictor.n_layers"] { Value::Int(n) => n as usize, _ => unreachable!() };
    let mut out: Vec<Blob> = Vec::new();
    let f32t = |out: &mut Vec<Blob>, ours: &str, theirs: &str| -> io::Result<()> {
        let (v, shape) = st.f32(theirs)?;
        out.push((ours.into(), "f32", shape, f32_bytes(&v)));
        Ok(())
    };
    let q8 = |out: &mut Vec<Blob>, ours: &str, theirs: &str| -> io::Result<()> {
        let (v, shape) = st.f32(theirs)?;
        out.push((ours.into(), "q8", shape.clone(), quantize(&v, shape[0], shape[1])));
        Ok(())
    };
    // `conv`: a pointwise conv, shape [rows, cols, 1] (Ultra stores those f16).
    let ternary = |out: &mut Vec<Blob>, ours: &str, module: &str, conv: bool| -> io::Result<()> {
        let (codes, s, rows, cols) = st.ternary(module)?;
        let shape = if conv { vec![rows, cols, 1] } else { vec![rows, cols] };
        if !ternary_q8 {
            out.push((ours.into(), "t2", shape, t2_bytes(&codes, &s)));
        } else if conv {
            let b = codes
                .iter()
                .enumerate()
                .flat_map(|(i, &c)| half::f16::from_f32(c as f32 * s[(i / cols) * (cols / GROUP) + (i % cols) / GROUP]).to_le_bytes())
                .collect();
            out.push((ours.into(), "f16", shape, b));
        } else {
            let d: Vec<f32> = s.iter().flat_map(|&x| [x; GROUP / 32]).collect(); // a group = four blocks of 32
            out.push((ours.into(), "q8", shape, q8_bytes(codes, &d, rows, cols)));
        }
        Ok(())
    };

    for l in 0..layers {
        let (o, t) = (format!("enc.blocks.{l}"), format!("encoder.layers.{l}"));
        for (ours, theirs) in [
            ("ff1.linear1", "feed_forward1.linear1"),
            ("ff1.linear2", "feed_forward1.linear2"),
            ("ff2.linear1", "feed_forward2.linear1"),
            ("ff2.linear2", "feed_forward2.linear2"),
            ("attn.linear_q", "self_attn.q_proj"),
            ("attn.linear_k", "self_attn.k_proj"),
            ("attn.linear_v", "self_attn.v_proj"),
            ("attn.linear_out", "self_attn.o_proj"),
            ("attn.linear_pos", "self_attn.relative_k_proj"),
        ] {
            ternary(&mut out, &format!("{o}.{ours}.weight"), &format!("{t}.{theirs}"), false)?;
        }
        ternary(&mut out, &format!("{o}.conv.pointwise1.weight"), &format!("{t}.conv.pointwise_conv1"), true)?;
        ternary(&mut out, &format!("{o}.conv.pointwise2.weight"), &format!("{t}.conv.pointwise_conv2"), true)?;
        f32t(&mut out, &format!("{o}.attn.pos_bias_u"), &format!("{t}.self_attn.bias_u"))?;
        f32t(&mut out, &format!("{o}.attn.pos_bias_v"), &format!("{t}.self_attn.bias_v"))?;
        f32t(&mut out, &format!("{o}.conv.depthwise.weight"), &format!("{t}.conv.depthwise_conv.weight"))?;
        for p in ["weight", "bias", "running_mean", "running_var"] {
            f32t(&mut out, &format!("{o}.conv.bn.{p}"), &format!("{t}.conv.norm.{p}"))?;
        }
        for (ours, theirs) in [
            ("norm_ff1", "norm_feed_forward1"),
            ("norm_attn", "norm_self_att"),
            ("norm_conv", "norm_conv"),
            ("norm_ff2", "norm_feed_forward2"),
            ("norm_out", "norm_out"),
        ] {
            for p in ["weight", "bias"] {
                f32t(&mut out, &format!("{o}.{ours}.{p}"), &format!("{t}.{theirs}.{p}"))?;
            }
        }
    }
    for i in [0, 2, 3, 5, 6] {
        for p in ["weight", "bias"] {
            f32t(&mut out, &format!("enc.pre_encode.conv.{i}.{p}"), &format!("encoder.subsampling.layers.{i}.{p}"))?;
        }
    }
    q8(&mut out, "enc.pre_encode.out.weight", "encoder.subsampling.linear.weight")?;
    f32t(&mut out, "enc.pre_encode.out.bias", "encoder.subsampling.linear.bias")?;
    for (ours, theirs) in [("joint.enc", "encoder_projector"), ("joint.pred", "decoder.decoder_projector"), ("joint.out", "joint.head")] {
        q8(&mut out, &format!("{ours}.weight"), &format!("{theirs}.weight"))?;
        f32t(&mut out, &format!("{ours}.bias"), &format!("{theirs}.bias"))?;
    }
    q8(&mut out, "pred.embed.weight", "decoder.embedding.weight")?;
    for l in 0..lstm {
        q8(&mut out, &format!("pred.lstm.{l}.Wx"), &format!("decoder.lstm.weight_ih_l{l}"))?;
        q8(&mut out, &format!("pred.lstm.{l}.Wh"), &format!("decoder.lstm.weight_hh_l{l}"))?;
        let (bi, s) = st.f32(&format!("decoder.lstm.bias_ih_l{l}"))?;
        let (bh, _) = st.f32(&format!("decoder.lstm.bias_hh_l{l}"))?;
        let b: Vec<f32> = bi.iter().zip(&bh).map(|(a, b)| a + b).collect();
        out.push((format!("pred.lstm.{l}.bias"), "f32", s, f32_bytes(&b)));
    }
    for p in ["proj.weight", "proj.bias", "ctx.weight", "ctx.bias", "out.bias"] {
        f32t(&mut out, &format!("vad.{p}"), &format!("vad_head.{p}"))?;
    }
    let (v, _) = st.f32("vad_head.out.weight")?; // [1, 128, 1] -> ours [128, 1]
    out.push(("vad.out.weight".into(), "f32", vec![v.len(), 1], f32_bytes(&v)));
    out.sort_by(|a, b| a.0.cmp(&b.0));
    write(dst, &meta, out)
}
