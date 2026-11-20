//! Minimal GGUF reader: metadata is skipped except what we need, tensors are dequantized to f32 on request.
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
    /// Skip a metadata value of type `t`; returns u32/u64 values (for general.alignment).
    fn skip_value(&mut self, t: u32) -> Option<u64> {
        match t {
            0 | 1 | 7 => {
                self.bytes(1);
                None
            }
            2 | 3 => {
                self.bytes(2);
                None
            }
            4 | 5 => Some(self.u32() as u64),
            6 => {
                self.bytes(4);
                None
            }
            10 | 11 => Some(self.u64()),
            12 => {
                self.bytes(8);
                None
            }
            8 => {
                self.string();
                None
            }
            9 => {
                let et = self.u32();
                let n = self.u64();
                for _ in 0..n {
                    self.skip_value(et);
                }
                None
            }
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
        let mut alignment = 32usize;
        for _ in 0..n_kv {
            let key = c.string();
            let t = c.u32();
            let v = c.skip_value(t);
            if key == "general.alignment" {
                alignment = v.unwrap() as usize;
            }
        }
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
        Ok(Gguf { data, data_start, tensors })
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
