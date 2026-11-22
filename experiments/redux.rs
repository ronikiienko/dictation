//! Parakeet Redux (moondream/parakeet-redux: ternary encoder, safetensors with Hugging Face names) -> our weights file,
//! to compare its accuracy with Ultra on the same runtime. Same architecture and tokenizer as Ultra, so the metadata is
//! Ultra's (names changed). Storage kinds follow Ultra's file, so the same GPU kernels run:
//!   ternary linears -> q8, exact: int8 code -1/0/+1, each group's f16 scale (128 values) copied to its four 32-blocks;
//!   ternary pointwise convs -> f16, exact (scale x -1/0/+1);
//!   f16 linears that Ultra keeps as q8 (subsampler out, joint, predictor) -> Q8_0 (amax/127 per block of 32, as GGUF);
//!   everything else -> f32. LSTM: one bias = bias_ih + bias_hh (our decoder adds a single bias).
//! Not the fast path: the encoder runs as int8 like Ultra, only the accuracy is Redux's.
//! usage: cargo run --release --example redux -- <redux folder: model.safetensors> <ultra model.weights> <out.weights>
use dictation::asr::{q8_bytes, write, Blob, Value, Weights};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::path::Path;

struct St {
    data: Vec<u8>,
    start: usize,
    header: serde_json::Map<String, Json>,
}

impl St {
    fn open(path: &Path) -> St {
        let data = std::fs::read(path).unwrap();
        let n = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let header = serde_json::from_slice(&data[8..8 + n]).unwrap();
        St { data, start: 8 + n, header }
    }
    fn get(&self, name: &str) -> (&str, Vec<usize>, &[u8]) {
        let t = self.header.get(name).unwrap_or_else(|| panic!("{name} not in safetensors"));
        let o = t["data_offsets"].as_array().unwrap();
        let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
        let shape = t["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        (t["dtype"].as_str().unwrap(), shape, &self.data[self.start + a..self.start + b])
    }
    fn f32(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let (dt, shape, b) = self.get(name);
        let v = match dt {
            "F32" => b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect(),
            "F16" => b.chunks_exact(2).map(|x| half::f16::from_le_bytes([x[0], x[1]]).to_f32()).collect(),
            d => panic!("{name}: dtype {d}"),
        };
        (v, shape)
    }
    /// Ternary linear: (codes -1/0/+1 [out][in], f16 scales per group of 128 [out][in/128]).
    fn ternary(&self, module: &str) -> (Vec<i8>, Vec<f32>, usize, usize) {
        let (_, qs, q) = self.get(&format!("{module}.qweight"));
        let (s, ss) = self.f32(&format!("{module}.scales"));
        let (rows, row_bytes, groups) = (qs[0], qs[1], ss[1]);
        let cols = groups * 128;
        assert_eq!(row_bytes, cols.div_ceil(5), "{module}: packed row length");
        let mut codes = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row = &q[r * row_bytes..(r + 1) * row_bytes];
            for i in 0..cols {
                let digit = (row[i / 5] as u32 / 3u32.pow((i % 5) as u32)) % 3;
                codes.push(digit as i8 - 1);
            }
        }
        (codes, s, rows, cols)
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

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let st = St::open(&Path::new(&a[0]).join("model.safetensors"));
    let ultra = Weights::open(Path::new(&a[1])).unwrap();
    let mut meta: HashMap<String, Value> = ultra.meta.clone();
    meta.insert("general.name".into(), Value::Str("Parakeet Redux".into()));
    meta.insert("general.repo_url".into(), Value::Str("https://huggingface.co/moondream/parakeet-redux".into()));
    meta.insert("stt.variant".into(), Value::Str("tdt-0.6b-redux (converted from ternary)".into()));

    let mut out: Vec<Blob> = Vec::new();
    let f32t = |out: &mut Vec<Blob>, ours: &str, theirs: &str| {
        let (v, shape) = st.f32(theirs);
        out.push((ours.into(), "f32", shape, f32_bytes(&v)));
    };
    let q8 = |out: &mut Vec<Blob>, ours: &str, theirs: &str| {
        let (v, shape) = st.f32(theirs);
        assert_eq!(shape.len(), 2);
        out.push((ours.into(), "q8", shape.clone(), quantize(&v, shape[0], shape[1])));
    };
    let ternary_q8 = |out: &mut Vec<Blob>, ours: &str, module: &str| {
        let (codes, s, rows, cols) = st.ternary(module);
        let d: Vec<f32> = s.iter().flat_map(|&x| [x; 4]).collect(); // group of 128 = four blocks of 32
        out.push((ours.into(), "q8", vec![rows, cols], q8_bytes(codes, &d, rows, cols)));
    };
    let ternary_f16 = |out: &mut Vec<Blob>, ours: &str, module: &str| {
        let (codes, s, rows, cols) = st.ternary(module);
        let b = codes
            .iter()
            .enumerate()
            .flat_map(|(i, &c)| half::f16::from_f32(c as f32 * s[(i / cols) * (cols / 128) + (i % cols) / 128]).to_le_bytes())
            .collect();
        out.push((ours.into(), "f16", vec![rows, cols, 1], b));
    };

    for l in 0..ultra.int("stt.parakeet.encoder.n_layers") as usize {
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
            ternary_q8(&mut out, &format!("{o}.{ours}.weight"), &format!("{t}.{theirs}"));
        }
        ternary_f16(&mut out, &format!("{o}.conv.pointwise1.weight"), &format!("{t}.conv.pointwise_conv1"));
        ternary_f16(&mut out, &format!("{o}.conv.pointwise2.weight"), &format!("{t}.conv.pointwise_conv2"));
        f32t(&mut out, &format!("{o}.attn.pos_bias_u"), &format!("{t}.self_attn.bias_u"));
        f32t(&mut out, &format!("{o}.attn.pos_bias_v"), &format!("{t}.self_attn.bias_v"));
        f32t(&mut out, &format!("{o}.conv.depthwise.weight"), &format!("{t}.conv.depthwise_conv.weight"));
        for p in ["weight", "bias", "running_mean", "running_var"] {
            f32t(&mut out, &format!("{o}.conv.bn.{p}"), &format!("{t}.conv.norm.{p}"));
        }
        for (ours, theirs) in [
            ("norm_ff1", "norm_feed_forward1"),
            ("norm_attn", "norm_self_att"),
            ("norm_conv", "norm_conv"),
            ("norm_ff2", "norm_feed_forward2"),
            ("norm_out", "norm_out"),
        ] {
            for p in ["weight", "bias"] {
                f32t(&mut out, &format!("{o}.{ours}.{p}"), &format!("{t}.{theirs}.{p}"));
            }
        }
    }
    for i in [0, 2, 3, 5, 6] {
        for p in ["weight", "bias"] {
            f32t(&mut out, &format!("enc.pre_encode.conv.{i}.{p}"), &format!("encoder.subsampling.layers.{i}.{p}"));
        }
    }
    q8(&mut out, "enc.pre_encode.out.weight", "encoder.subsampling.linear.weight");
    f32t(&mut out, "enc.pre_encode.out.bias", "encoder.subsampling.linear.bias");
    for (ours, theirs) in [("joint.enc", "encoder_projector"), ("joint.pred", "decoder.decoder_projector"), ("joint.out", "joint.head")] {
        q8(&mut out, &format!("{ours}.weight"), &format!("{theirs}.weight"));
        f32t(&mut out, &format!("{ours}.bias"), &format!("{theirs}.bias"));
    }
    q8(&mut out, "pred.embed.weight", "decoder.embedding.weight");
    for l in 0..ultra.int("stt.parakeet.predictor.n_layers") as usize {
        q8(&mut out, &format!("pred.lstm.{l}.Wx"), &format!("decoder.lstm.weight_ih_l{l}"));
        q8(&mut out, &format!("pred.lstm.{l}.Wh"), &format!("decoder.lstm.weight_hh_l{l}"));
        let (bi, s) = st.f32(&format!("decoder.lstm.bias_ih_l{l}"));
        let (bh, _) = st.f32(&format!("decoder.lstm.bias_hh_l{l}"));
        let b: Vec<f32> = bi.iter().zip(&bh).map(|(a, b)| a + b).collect();
        out.push((format!("pred.lstm.{l}.bias"), "f32", s, f32_bytes(&b)));
    }
    for p in ["proj.weight", "proj.bias", "ctx.weight", "ctx.bias", "out.bias"] {
        f32t(&mut out, &format!("vad.{p}"), &format!("vad_head.{p}"));
    }
    let (v, _) = st.f32("vad_head.out.weight"); // [1, 128, 1] -> ours [128, 1]
    out.push(("vad.out.weight".into(), "f32", vec![v.len(), 1], f32_bytes(&v)));

    // Same tensors, shapes and kinds as Ultra's file, else the runtime would run something else.
    let theirs: HashMap<&str, (&str, &Vec<usize>)> = out.iter().map(|(n, k, s, _)| (n.as_str(), (*k, s))).collect();
    let mut bad = 0;
    for n in ultra.names() {
        let (k, s) = (ultra.kind(&n), ultra.shape(&n));
        match theirs.get(n.as_str()) {
            Some((k2, s2)) if *k2 == k && **s2 == s => {}
            other => {
                eprintln!("mismatch {n}: ultra {k} {s:?}, redux {other:?}");
                bad += 1;
            }
        }
    }
    assert!(bad == 0 && theirs.len() == ultra.names().len(), "{bad} mismatches, {} vs {} tensors", theirs.len(), ultra.names().len());
    out.sort_by(|a, b| a.0.cmp(&b.0));
    write(Path::new(&a[2]), &meta, out).unwrap();
    eprintln!("written {}", a[2]);
}
