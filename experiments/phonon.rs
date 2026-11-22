//! Phonon-2 (FermionResearch, English-only Parakeet TDT 0.6B v3 derivative) -> our weights file, for comparisons.
//! Input: cstr/phonon2-GGUF `phonon2-q8_0.gguf` (CrispASR tensor names). Same architecture and tokenizer as Ultra
//! (checked here), so the metadata is Ultra's. Phonon-2 has no pause head: Ultra's is copied (it reads the
//! subsampler output, which is Phonon-2's own here, so long-take cuts can differ from Ultra's).
//! Kinds as in Ultra's file: q8 linears (Q8_0 copied as is), f16 pointwise convs, f32 rest; LSTM bias = b_ih + b_hh.
//! usage: cargo run --release --example phonon -- <phonon2-q8_0.gguf> <ultra model.weights> <out>
use dictation::asr::{q8_bytes, write, Blob, Gguf, Value, Weights};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let g = Gguf::open(Path::new(&a[0])).unwrap();
    let ultra = Weights::open(Path::new(&a[1])).unwrap();
    let toks = |v: &Value| match v {
        Value::Arr(x) => x.iter().map(|t| if let Value::Str(s) = t { s.clone() } else { String::new() }).collect::<Vec<_>>(),
        _ => vec![],
    };
    let mut theirs = toks(&g.meta["tokenizer.ggml.tokens"]);
    theirs.push("<blank>".into());
    assert_eq!(theirs, ultra.strs("tokenizer.ggml.tokens"), "tokenizer differs from Ultra's");
    for (k, v) in [("n_layers", 24), ("d_model", 1024), ("ff_dim", 4096), ("conv_kernel", 9), ("vocab_size", 8192)] {
        assert!(matches!(g.meta[&format!("parakeet.{k}")], Value::Int(x) if x == v), "{k}");
    }
    let mut meta = ultra.meta.clone();
    meta.insert("general.name".into(), Value::Str("Phonon-2".into()));
    meta.insert("general.repo_url".into(), Value::Str("https://huggingface.co/FermionResearch/Phonon-2".into()));
    meta.insert("stt.variant".into(), Value::Str("tdt-0.6b-phonon2".into()));

    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut out: Vec<Blob> = Vec::new();
    let f32t = |out: &mut Vec<Blob>, ours: &str, t: &str, shape: Option<Vec<usize>>| {
        let (v, s) = g.f32(t);
        out.push((ours.into(), "f32", shape.unwrap_or(s), f32b(&v)));
    };
    let q8 = |out: &mut Vec<Blob>, ours: &str, t: &str| {
        let (q, d, s) = match g.q8_0(t) {
            Some(x) => x,
            None => {
                // f16 in the GGUF (LSTM): Q8_0 like GGUF, per block of 32, d = amax / 127.
                let (v, s) = g.f32(t);
                let mut q = Vec::new();
                let mut d = Vec::new();
                for b in v.chunks_exact(32) {
                    let m = half::f16::from_f32(b.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0).to_f32();
                    d.push(m);
                    q.extend(b.iter().map(|x| if m == 0.0 { 0 } else { (x / m).round().clamp(-127.0, 127.0) as i8 }));
                }
                (q, d, s)
            }
        };
        out.push((ours.into(), "q8", s.clone(), q8_bytes(q, &d, s[0], s[1])));
    };
    for l in 0..24 {
        let (o, t) = (format!("enc.blocks.{l}"), format!("encoder.layers.{l}"));
        for (ours, th) in [
            ("ff1.linear1", "ff1.linear1"),
            ("ff1.linear2", "ff1.linear2"),
            ("ff2.linear1", "ff2.linear1"),
            ("ff2.linear2", "ff2.linear2"),
            ("attn.linear_q", "attn.q"),
            ("attn.linear_k", "attn.k"),
            ("attn.linear_v", "attn.v"),
            ("attn.linear_out", "attn.out"),
            ("attn.linear_pos", "attn.pos"),
        ] {
            q8(&mut out, &format!("{o}.{ours}.weight"), &format!("{t}.{th}.weight"));
        }
        for (ours, th) in [("pointwise1", "pw1"), ("pointwise2", "pw2")] {
            let (v, s) = g.f32(&format!("{t}.conv.{th}.weight"));
            let b = v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect();
            out.push((format!("{o}.conv.{ours}.weight"), "f16", vec![s[0], s[1], 1], b));
        }
        f32t(&mut out, &format!("{o}.attn.pos_bias_u"), &format!("{t}.attn.pos_bias_u"), None);
        f32t(&mut out, &format!("{o}.attn.pos_bias_v"), &format!("{t}.attn.pos_bias_v"), None);
        f32t(&mut out, &format!("{o}.conv.depthwise.weight"), &format!("{t}.conv.dw.weight"), None);
        for p in ["weight", "bias", "running_mean", "running_var"] {
            f32t(&mut out, &format!("{o}.conv.bn.{p}"), &format!("{t}.conv.bn.{p}"), None);
        }
        for n in ["norm_ff1", "norm_attn", "norm_conv", "norm_ff2", "norm_out"] {
            for p in ["weight", "bias"] {
                f32t(&mut out, &format!("{o}.{n}.{p}"), &format!("{t}.{n}.{p}"), None);
            }
        }
    }
    for i in [0, 2, 3, 5, 6] {
        for p in ["weight", "bias"] {
            f32t(&mut out, &format!("enc.pre_encode.conv.{i}.{p}"), &format!("encoder.pre.conv.{i}.{p}"), None);
        }
    }
    q8(&mut out, "enc.pre_encode.out.weight", "encoder.pre.out.weight");
    f32t(&mut out, "enc.pre_encode.out.bias", "encoder.pre.out.bias", None);
    for j in ["enc", "pred", "out"] {
        q8(&mut out, &format!("joint.{j}.weight"), &format!("joint.{j}.weight"));
        f32t(&mut out, &format!("joint.{j}.bias"), &format!("joint.{j}.bias"), None);
    }
    q8(&mut out, "pred.embed.weight", "decoder.embed.weight");
    for l in 0..2 {
        q8(&mut out, &format!("pred.lstm.{l}.Wx"), &format!("decoder.lstm.{l}.w_ih"));
        q8(&mut out, &format!("pred.lstm.{l}.Wh"), &format!("decoder.lstm.{l}.w_hh"));
        let (bi, s) = g.f32(&format!("decoder.lstm.{l}.b_ih"));
        let (bh, _) = g.f32(&format!("decoder.lstm.{l}.b_hh"));
        let b: Vec<f32> = bi.iter().zip(&bh).map(|(x, y)| x + y).collect();
        out.push((format!("pred.lstm.{l}.bias"), "f32", s, f32b(&b)));
    }
    for n in ultra.names().into_iter().filter(|n| n.starts_with("vad.")) {
        let (v, s) = ultra.f32(&n);
        out.push((n, "f32", s, f32b(&v)));
    }
    // Every name and shape as in Ultra's file.
    let mut ours: Vec<(String, Vec<usize>)> = out.iter().map(|b| (b.0.clone(), b.2.clone())).collect();
    let mut want: Vec<(String, Vec<usize>)> = ultra.names().into_iter().map(|n| (n.clone(), ultra.shape(&n))).collect();
    ours.sort();
    want.sort();
    assert_eq!(ours, want, "names/shapes differ from Ultra's file");
    out.sort_by(|a, b| a.0.cmp(&b.0));
    write(Path::new(&a[2]), &meta, out).unwrap();
    eprintln!("written {}", a[2]);
}
