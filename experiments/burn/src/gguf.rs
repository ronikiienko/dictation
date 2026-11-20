//! Minimal GGUF reader: metadata values are parsed, tensors are dequantized to f32 on request.
//! Supports the types in the Parakeet Ultra Q8_0 file: F32, F16, Q8_0.
use std::collections::HashMap;
use std::io::{self, Read};

pub struct TensorInfo {
    /// ggml order: ne[0] is the fastest-varying (row) dimension.
    pub ne: Vec<usize>,
    pub ty: u32,
    offset: usize,
}

pub struct Gguf {
    data: Vec<u8>,
    data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
    pub meta: HashMap<String, Value>,
}

/// A metadata value (only the kinds Parakeet's GGUF uses are kept; others are skipped as `Other`).
#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

struct Cur<'a>(&'a [u8], usize);
impl Cur<'_> {
    fn bytes(&mut self, n: usize) -> &[u8] {
        let s = &self.0[self.1..self.1 + n];
        self.1 += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.bytes(8).try_into().unwrap())
    }
    fn string(&mut self) -> String {
        let n = self.u64() as usize;
        String::from_utf8_lossy(self.bytes(n)).into_owned()
    }
    fn value(&mut self, t: u32) -> Value {
        match t {
            0 => Value::Int(self.bytes(1)[0] as i64),
            1 => Value::Int(self.bytes(1)[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.bytes(2).try_into().unwrap()) as i64),
            3 => Value::Int(i16::from_le_bytes(self.bytes(2).try_into().unwrap()) as i64),
            4 => Value::Int(self.u32() as i64),
            5 => Value::Int(self.u32() as i32 as i64),
            6 => Value::Float(f32::from_le_bytes(self.bytes(4).try_into().unwrap()) as f64),
            7 => Value::Bool(self.bytes(1)[0] != 0),
            8 => Value::Str(self.string()),
            9 => {
                let et = self.u32();
                let n = self.u64();
                Value::Arr((0..n).map(|_| self.value(et)).collect())
            }
            10 => Value::Int(self.u64() as i64),
            11 => Value::Int(self.u64() as i64),
            12 => Value::Float(f64::from_le_bytes(self.bytes(8).try_into().unwrap())),
            _ => panic!("unknown gguf value type {t}"),
        }
    }
}

impl Gguf {
    pub fn open(path: &std::path::Path) -> io::Result<Gguf> {
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;
        let mut c = Cur(&data, 0);
        assert_eq!(c.bytes(4), b"GGUF");
        let _version = c.u32();
        let n_tensors = c.u64();
        let n_kv = c.u64();
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let key = c.string();
            let t = c.u32();
            meta.insert(key, c.value(t));
        }
        let alignment = match meta.get("general.alignment") {
            Some(Value::Int(a)) => *a as usize,
            _ => 32,
        };
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = c.string();
            let nd = c.u32();
            let ne = (0..nd).map(|_| c.u64() as usize).collect();
            let ty = c.u32();
            let offset = c.u64() as usize;
            tensors.insert(name, TensorInfo { ne, ty, offset });
        }
        let data_start = c.1.div_ceil(alignment) * alignment;
        Ok(Gguf { data, data_start, tensors, meta })
    }

    pub fn int(&self, key: &str) -> i64 {
        match self.meta.get(key) {
            Some(Value::Int(v)) => *v,
            v => panic!("gguf {key}: expected int, got {v:?}"),
        }
    }
    pub fn float(&self, key: &str) -> f64 {
        match self.meta.get(key) {
            Some(Value::Float(v)) => *v,
            Some(Value::Int(v)) => *v as f64,
            v => panic!("gguf {key}: expected float, got {v:?}"),
        }
    }
    pub fn strs(&self, key: &str) -> Vec<String> {
        match self.meta.get(key) {
            Some(Value::Arr(a)) => a.iter().map(|v| match v { Value::Str(s) => s.clone(), _ => panic!("{key}: not strings") }).collect(),
            v => panic!("gguf {key}: expected array, got {v:?}"),
        }
    }
    pub fn ints(&self, key: &str) -> Vec<i64> {
        match self.meta.get(key) {
            Some(Value::Arr(a)) => a.iter().map(|v| match v { Value::Int(i) => *i, _ => panic!("{key}: not ints") }).collect(),
            v => panic!("gguf {key}: expected array, got {v:?}"),
        }
    }

    pub fn info(&self, name: &str) -> &TensorInfo {
        self.tensors.get(name).unwrap_or_else(|| panic!("tensor {name} not in gguf"))
    }

    /// Tensor as f32, row-major with shape = reversed ne (PyTorch order).
    pub fn f32(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let t = self.info(name);
        let n: usize = t.ne.iter().product();
        let b = &self.data[self.data_start + t.offset..];
        let v = match t.ty {
            0 => b[..n * 4].chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect(),
            1 => b[..n * 2].chunks_exact(2).map(|x| half::f16::from_le_bytes(x.try_into().unwrap()).to_f32()).collect(),
            8 => {
                // Q8_0: blocks of 32 values = f16 scale + 32 x i8, along ne[0].
                let mut v = Vec::with_capacity(n);
                for blk in b[..n / 32 * 34].chunks_exact(34) {
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    v.extend(blk[2..].iter().map(|&q| q as i8 as f32 * d));
                }
                v
            }
            ty => panic!("{name}: unsupported ggml type {ty}"),
        };
        (v, t.ne.iter().rev().copied().collect())
    }

    /// Q8_0 tensor as raw int8 values + one f32 scale per block of 32 (blocks run along ne[0]),
    /// shape = reversed ne. None if the tensor isn't Q8_0.
    pub fn q8_0(&self, name: &str) -> Option<(Vec<i8>, Vec<f32>, Vec<usize>)> {
        let t = self.info(name);
        if t.ty != 8 {
            return None;
        }
        let n: usize = t.ne.iter().product();
        let b = &self.data[self.data_start + t.offset..][..n / 32 * 34];
        let mut q = Vec::with_capacity(n);
        let mut d = Vec::with_capacity(n / 32);
        for blk in b.chunks_exact(34) {
            d.push(half::f16::from_le_bytes([blk[0], blk[1]]).to_f32());
            q.extend(blk[2..].iter().map(|&v| v as i8));
        }
        Some((q, d, t.ne.iter().rev().copied().collect()))
    }
}
