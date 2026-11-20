//! Our model file: the GGUF converted once into tensors that are ready to upload, so loading is mmap + upload with no
//! CPU work. Layout:
//!   "PQWEIGHT" | u32 format version | u64 header length | header (JSON) | zero padding | data
//! Header: {"format", "layout", "meta": GGUF metadata, "tensors": {name: {"kind", "shape", "offset", "len"}}}.
//! Data starts at a 4096-byte boundary, every tensor at a 64-byte boundary (offsets are relative to the data start).
//! Kinds: "f32", "f16" (as stored in the GGUF), "q8" (Q8_0 in burn's quantized byte layout: the int8 values, then one
//! f16 scale per block of 32, aligned to 4; 2-D [rows, cols] with blocks along cols). The q8 layout belongs to burn
//! 0.22: `LAYOUT` names it, and a file with another layout must be converted again.
use crate::gguf::{Gguf, Value};
use burn::tensor::quantization::{QuantScheme, QuantValue, ScaleDtype};
use burn::tensor::{AllocationProperty, Bytes, DType, TensorData};
use serde_json::{json, Map, Value as Json};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"PQWEIGHT";
const FORMAT: u32 = 1;
/// Changes whenever the stored byte layout would (burn's quantized layout included).
pub const LAYOUT: &str = "burn-0.22 q8s block32 f16-scales";
const DATA_ALIGN: usize = 4096;
const TENSOR_ALIGN: usize = 64;

pub fn q8_scheme() -> QuantScheme {
    QuantScheme::default().with_value(QuantValue::Q8S).per_block([32], ScaleDtype::F16)
}

struct Entry {
    kind: String,
    shape: Vec<usize>,
    offset: usize,
    len: usize,
}

pub struct Weights {
    data: bytes::Bytes, // the mmap from the data start on
    tensors: HashMap<String, Entry>,
    pub meta: HashMap<String, Value>,
}

fn to_json(v: &Value) -> Json {
    match v {
        Value::Int(i) => json!(i),
        Value::Float(f) => json!({ "f": f }), // tagged: 2.0 must stay a float
        Value::Bool(b) => json!(b),
        Value::Str(s) => json!(s),
        Value::Arr(a) => Json::Array(a.iter().map(to_json).collect()),
    }
}
fn from_json(v: &Json) -> Value {
    match v {
        Json::Number(n) => Value::Int(n.as_i64().expect("integer")),
        Json::Object(o) => Value::Float(o["f"].as_f64().expect("float")),
        Json::Bool(b) => Value::Bool(*b),
        Json::String(s) => Value::Str(s.clone()),
        Json::Array(a) => Value::Arr(a.iter().map(from_json).collect()),
        Json::Null => panic!("null in metadata"),
    }
}

/// burn's bytes for a Q8_0 tensor (int8 values + f16 scales).
fn q8_bytes(q: Vec<i8>, d: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let data = TensorData::quantized(q, [rows, cols], q8_scheme(), d, None);
    data.as_bytes().to_vec()
}

/// GGUF -> our file. Every Q8_0 tensor is checked: its stored bytes must read back as the GGUF's values and scales.
pub fn convert(src: &Path, dst: &Path) -> io::Result<()> {
    let g = Gguf::open(src)?;
    let mut names: Vec<&String> = g.tensors.keys().collect();
    names.sort();
    let mut tensors = Map::new();
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut offset = 0usize;
    for name in names {
        let (kind, shape, blob) = if let Some((q, d, shape)) = g.q8_0(name) {
            let cols = *shape.last().unwrap();
            let rows = q.len() / cols;
            let blob = q8_bytes(q.clone(), &d, rows, cols);
            let (q2, d2) = split_q8(&blob, rows * cols);
            assert!(q2 == q.as_slice() && d2 == d, "{name}: burn's q8 layout is not the expected one");
            ("q8", shape, blob)
        } else if let Some((v, shape)) = g.f16(name) {
            ("f16", shape, v.iter().flat_map(|x| x.to_le_bytes()).collect())
        } else {
            let (v, shape) = g.f32(name);
            ("f32", shape, v.iter().flat_map(|x| x.to_le_bytes()).collect())
        };
        offset = offset.next_multiple_of(TENSOR_ALIGN);
        tensors.insert(name.clone(), json!({ "kind": kind, "shape": shape, "offset": offset, "len": blob.len() }));
        offset += blob.len();
        blobs.push(blob);
    }
    let meta: Map<String, Json> = g.meta.iter().map(|(k, v)| (k.clone(), to_json(v))).collect();
    let header = serde_json::to_vec(&json!({ "format": FORMAT, "layout": LAYOUT, "meta": meta, "tensors": tensors }))?;

    let tmp = dst.with_extension("part");
    let mut f = io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(MAGIC)?;
    f.write_all(&FORMAT.to_le_bytes())?;
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(&header)?;
    let mut pos = 8 + 4 + 8 + header.len();
    let start = pos.next_multiple_of(DATA_ALIGN);
    f.write_all(&vec![0u8; start - pos])?;
    pos = 0;
    for blob in &blobs {
        let pad = pos.next_multiple_of(TENSOR_ALIGN) - pos;
        f.write_all(&vec![0u8; pad])?;
        f.write_all(blob)?;
        pos += pad + blob.len();
    }
    f.into_inner()?.sync_all()?;
    std::fs::rename(tmp, dst)
}

/// burn q8 bytes -> (int8 values, f32 scales). Values first, scales are the last n/32 f16s.
fn split_q8(b: &[u8], n: usize) -> (&[i8], Vec<f32>) {
    let nb = n / 32;
    let q = bytemuck::cast_slice(&b[..n]);
    let s = &b[b.len() - nb * 2..];
    let d = s.chunks_exact(2).map(|x| half::f16::from_le_bytes([x[0], x[1]]).to_f32()).collect();
    (q, d)
}

impl Weights {
    pub fn open(path: &Path) -> io::Result<Weights> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the file is ours and not modified while mapped (conversion writes a new file and renames it).
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {m}", path.display()));
        if map.len() < 20 || &map[..8] != MAGIC {
            return Err(bad("not a weights file"));
        }
        let format = u32::from_le_bytes(map[8..12].try_into().unwrap());
        let hlen = u64::from_le_bytes(map[12..20].try_into().unwrap()) as usize;
        let header: Json = serde_json::from_slice(&map[20..20 + hlen]).map_err(|e| bad(&e.to_string()))?;
        if format != FORMAT || header["layout"] != LAYOUT {
            return Err(bad(&format!("format {format} / layout {}: convert the model again", header["layout"])));
        }
        let start = (20 + hlen).next_multiple_of(DATA_ALIGN);
        let meta = header["meta"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), from_json(v))).collect();
        let tensors = header["tensors"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                let u = |f: &str| v[f].as_u64().unwrap() as usize;
                let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                (k.clone(), Entry { kind: v["kind"].as_str().unwrap().to_string(), shape, offset: u("offset"), len: u("len") })
            })
            .collect();
        let data = bytes::Bytes::from_owner(map).slice(start..);
        Ok(Weights { data, tensors, meta })
    }

    fn entry(&self, name: &str) -> &Entry {
        self.tensors.get(name).unwrap_or_else(|| panic!("tensor {name} not in the weights file"))
    }
    fn raw(&self, e: &Entry) -> &[u8] {
        &self.data[e.offset..e.offset + e.len]
    }

    /// A linear weight as stored, 2-D [out, in], without copying (burn reads it from the mapped file when
    /// uploading).
    pub fn lin_data(&self, name: &str) -> TensorData {
        let e = self.entry(name);
        let shape = [e.shape[0], e.shape[1..].iter().product()];
        let bytes = Bytes::from_shared(self.data.slice(e.offset..e.offset + e.len), AllocationProperty::File);
        let dtype = match e.kind.as_str() {
            "f32" => DType::F32,
            "f16" => DType::F16,
            "q8" => {
                assert_eq!(e.shape.len(), 2, "{name}: q8 must be 2-D");
                DType::QFloat(q8_scheme())
            }
            k => panic!("{name}: unknown kind {k}"),
        };
        TensorData::from_bytes(bytes, shape, dtype)
    }
    pub fn kind(&self, name: &str) -> &str {
        &self.entry(name).kind
    }
    /// Tensor as f32 (q8 dequantized), shape in PyTorch order.
    pub fn f32(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let e = self.entry(name);
        let b = self.raw(e);
        let n: usize = e.shape.iter().product();
        let v = match e.kind.as_str() {
            "f32" => b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect(),
            "f16" => b.chunks_exact(2).map(|x| half::f16::from_le_bytes([x[0], x[1]]).to_f32()).collect(),
            "q8" => {
                let (q, d) = split_q8(b, n);
                q.iter().enumerate().map(|(i, &v)| v as f32 * d[i / 32]).collect()
            }
            k => panic!("{name}: unknown kind {k}"),
        };
        (v, e.shape.clone())
    }

    /// Q8_0 tensor as int8 values + one f32 scale per block of 32 (CPU decoder). None if not q8.
    pub fn q8_0(&self, name: &str) -> Option<(Vec<i8>, Vec<f32>, Vec<usize>)> {
        let e = self.entry(name);
        if e.kind != "q8" {
            return None;
        }
        let (q, d) = split_q8(self.raw(e), e.shape.iter().product());
        Some((q.to_vec(), d, e.shape.clone()))
    }

    pub fn int(&self, key: &str) -> i64 {
        match self.meta.get(key) {
            Some(Value::Int(v)) => *v,
            v => panic!("{key}: expected int, got {v:?}"),
        }
    }
    pub fn float(&self, key: &str) -> f64 {
        match self.meta.get(key) {
            Some(Value::Float(v)) => *v,
            Some(Value::Int(v)) => *v as f64,
            v => panic!("{key}: expected float, got {v:?}"),
        }
    }
    pub fn strs(&self, key: &str) -> Vec<String> {
        match self.meta.get(key) {
            Some(Value::Arr(a)) => a.iter().map(|v| match v { Value::Str(s) => s.clone(), _ => panic!("{key}: not strings") }).collect(),
            v => panic!("{key}: expected array, got {v:?}"),
        }
    }
    pub fn ints(&self, key: &str) -> Vec<i64> {
        match self.meta.get(key) {
            Some(Value::Arr(a)) => a.iter().map(|v| match v { Value::Int(i) => *i, _ => panic!("{key}: not ints") }).collect(),
            v => panic!("{key}: expected array, got {v:?}"),
        }
    }
}
