//! Builds the Neural Engine model (compiled CoreML, [`MODEL`]) from moondream/parakeet-ultra's
//! `model.safetensors` (fp16, as published): the architecture ships with the app (assets/parakeet-ultra-ane, made
//! from our export by experiments/ane/export.py + recipe.py; NOTES.md "Neural Engine"), only `weights/weight.bin` is written here,
//! following the recipe's segments:
//! - literal bytes (file and blob headers, architecture-only constants: index ranges, zero biases), from consts.bin;
//! - an encoder tensor as fp16 (published fp16: copied; the few f32 ones, norms etc.: rounded to nearest);
//! - the depthwise conv with its batch norm folded in (CoreML folds it at export; same f32 formula, same bytes);
//! - `relative_k_proj`'s sin or cos half (attention's position term without the shift, export.py): copied;
//! - the sin/cos tables of that term (f64, rounded to f32, then fp16: the same bytes as CoreML's).
//! One plain model, one 30 s window (why: ane.rs); ane.rs loads it.
use super::shapes::{D, DK, H};
use super::redux::St;
use half::f16;
use std::io::{self, Write};
use std::path::Path;

/// The built model's folder name in the model folder. A new name for a new template, so an older build is not taken
/// for this one (models.rs: not installed, downloaded and built again). Was enc-30s.mlmodelc (with the shift).
pub const MODEL: &str = "enc-30s-noshift.mlmodelc";
const DIR: &str = "parakeet-ultra-ane";
const MIL: &[u8] = include_bytes!("../../assets/parakeet-ultra-ane/model.mil");
const COREMLDATA: &[u8] = include_bytes!("../../assets/parakeet-ultra-ane/coremldata.bin");
const METADATA: &[u8] = include_bytes!("../../assets/parakeet-ultra-ane/metadata.json");
const ANALYTICS: &[u8] = include_bytes!("../../assets/parakeet-ultra-ane/analytics/coremldata.bin");
const CONSTS: &[u8] = include_bytes!("../../assets/parakeet-ultra-ane/consts.bin");
const RECIPE: &str = include_str!("../../assets/parakeet-ultra-ane/recipe.json");
const BN_EPS: f32 = 1e-5;

fn bad(m: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m)
}

fn f16_bytes(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(|x| f16::from_f32(x).to_le_bytes()).collect()
}

/// Encoder tensor `name` as fp16 bytes.
fn tensor(st: &St, name: &str) -> io::Result<Vec<u8>> {
    let (dt, _, b) = st.get(name)?;
    match dt {
        "F16" => Ok(b.to_vec()),
        "F32" => Ok(f16_bytes(st.f32(name)?.0)),
        d => Err(bad(format!("{name}: dtype {d}"))),
    }
}

/// Depthwise conv weight [C][1][K] (or bias [C]) of layer `i` with the batch norm folded in:
/// s = gamma / sqrt(var + eps); weight * s, beta - mean * s (f32, as numpy/CoreML).
fn bn(st: &St, i: usize, bias: bool) -> io::Result<Vec<u8>> {
    let p = format!("encoder.layers.{i}.conv.norm.");
    let (gamma, var) = (st.f32(&format!("{p}weight"))?.0, st.f32(&format!("{p}running_var"))?.0);
    let s: Vec<f32> = gamma.iter().zip(&var).map(|(g, v)| g / (v + BN_EPS).sqrt()).collect();
    if bias {
        let (beta, mean) = (st.f32(&format!("{p}bias"))?.0, st.f32(&format!("{p}running_mean"))?.0);
        return Ok(f16_bytes(beta.iter().zip(&mean).zip(&s).map(|((b, m), s)| b - m * s)));
    }
    let (w, shape) = st.f32(&format!("encoder.layers.{i}.conv.depthwise_conv.weight"))?;
    let k = shape.iter().skip(1).product::<usize>();
    Ok(f16_bytes(w.iter().enumerate().map(|(j, x)| x * s[j / k])))
}

/// relative_k_proj (no bias, [D out][D in], inputs = (sin, cos) interleaved) of layer `i`: its sin (`half` 0, even
/// inputs) or cos (1, odd) columns, laid out as [1][H][DK][D/2] (heads, head dims, frequencies).
fn rkp(st: &St, i: usize, half: usize) -> io::Result<Vec<u8>> {
    let (w, shape) = st.f32(&format!("encoder.layers.{i}.self_attn.relative_k_proj.weight"))?;
    if shape != [D, D] || half > 1 {
        return Err(bad(format!("relative_k_proj {shape:?}, half {half}")));
    }
    Ok(f16_bytes((0..H * DK).flat_map(|o| (0..D / 2).map(move |k| (o, k))).map(|(o, k)| w[o * D + 2 * k + half])))
}

/// sin or cos of position j times frequency k (export.py `sincos`): [t][D/2], or [D/2][t] when `transposed`.
fn sincos(cos: bool, t: usize, transposed: bool) -> Vec<u8> {
    let v = |j: usize, k: usize| {
        let a = j as f64 / 10000f64.powf((2 * k) as f64 / D as f64);
        (if cos { a.cos() } else { a.sin() }) as f32
    };
    if transposed {
        f16_bytes((0..D / 2).flat_map(|k| (0..t).map(move |j| v(j, k))))
    } else {
        f16_bytes((0..t).flat_map(|j| (0..D / 2).map(move |k| v(j, k))))
    }
}

/// `safetensors` (moondream/parakeet-ultra's model.safetensors) -> `<dir>/`[`MODEL`], written next to it first and
/// renamed when complete; anything else in `dir` (an older build) is removed first. `progress(done 0..1)`.
pub fn build(safetensors: &Path, dir: &Path, mut progress: impl FnMut(f64)) -> io::Result<()> {
    let recipe: serde_json::Value = serde_json::from_str(RECIPE).map_err(|e| bad(format!("{DIR}/recipe.json: {e}")))?;
    let segs = recipe["segments"].as_array().ok_or_else(|| bad("recipe: no segments".into()))?;
    let size = recipe["weight_bin_size"].as_u64().ok_or_else(|| bad("recipe: no size".into()))?;
    let st = St::open(safetensors)?;
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir)?;
    let done = dir.join(MODEL);
    let tmp = done.with_extension("part");
    std::fs::create_dir_all(tmp.join("weights"))?;
    std::fs::create_dir_all(tmp.join("analytics"))?;
    std::fs::write(tmp.join("model.mil"), MIL)?;
    std::fs::write(tmp.join("coremldata.bin"), COREMLDATA)?;
    std::fs::write(tmp.join("metadata.json"), METADATA)?;
    std::fs::write(tmp.join("analytics/coremldata.bin"), ANALYTICS)?;
    let mut f = io::BufWriter::with_capacity(1 << 22, std::fs::File::create(tmp.join("weights/weight.bin"))?);
    let mut written = 0u64;
    for (n, s) in segs.iter().enumerate() {
        let a = s.as_array().ok_or_else(|| bad("recipe: bad segment".into()))?;
        let num = |k: usize| a.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| bad(format!("recipe: segment {n}")));
        let bytes = match a.first().and_then(|v| v.as_str()) {
            Some("bytes") => CONSTS.get(num(1)?..num(1)? + num(2)?).ok_or_else(|| bad("recipe: consts range".into()))?.to_vec(),
            Some("tensor") => tensor(&st, a.get(1).and_then(|v| v.as_str()).ok_or_else(|| bad("recipe: tensor name".into()))?)?,
            Some("bn_w") => bn(&st, num(1)?, false)?,
            Some("bn_b") => bn(&st, num(1)?, true)?,
            Some("rkp") => rkp(&st, num(1)?, num(2)?)?,
            Some("sincos") => {
                let cos = match a.get(1).and_then(|v| v.as_str()) {
                    Some("sin") => false,
                    Some("cos") => true,
                    f => return Err(bad(format!("recipe: sincos {f:?}"))),
                };
                sincos(cos, num(2)?, num(3)? == 1)
            }
            k => return Err(bad(format!("recipe: segment kind {k:?}"))),
        };
        f.write_all(&bytes)?;
        written += bytes.len() as u64;
        progress((n + 1) as f64 / segs.len() as f64);
    }
    f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    if written != size {
        return Err(bad(format!("weight.bin: {written} bytes written, recipe says {size}")));
    }
    std::fs::rename(&tmp, &done)
}
