//! Parakeet Ultra's encoder (subsampler + conformer blocks) on the Apple Neural Engine, through CoreML. macOS only.
//! The model: our export of moondream/parakeet-ultra (fp16), built by the app from the download (ane_build.rs): one
//! compiled `.mlmodelc` with a function per window (shared weights, listed in its metadata.json; a folder of plain
//! one-window models works too). NOTES.md "Neural Engine". Each window: input `mel` [1, FR, 128] (frame-major, zero past the valid frames) + `length` [1]
//! (valid mel frames, the model masks the rest), output `encoder` [1, FR/8, 1024]. A piece goes to the smallest
//! window that holds it (padding is exact). The first load of a window compiles it for the Neural Engine (~30 s, once:
//! macOS caches it), later loads take ~0.1 s each.
// `dataPointer` (deprecated in favour of block-based access) is the plain way to fill and read the arrays.
#![allow(deprecated)]
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration,
    MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use std::path::Path;

const C: usize = 128; // mel bins
const D: usize = 1024; // encoder width

struct Window {
    model: Retained<MLModel>,
    /// Mel frames it takes.
    frames: usize,
    /// Its input, reused: [1, frames, 128] f32.
    mel: Retained<MLMultiArray>,
}

pub struct Encoder {
    windows: Vec<Window>,
    length: Retained<MLMultiArray>,
}

fn err(what: &str, e: Retained<NSError>) -> String {
    format!("{what}: {}", e.localizedDescription())
}

fn shape(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
    let n: Vec<Retained<NSNumber>> = dims.iter().map(|&d| NSNumber::new_isize(d as isize)).collect();
    NSArray::from_retained_slice(&n)
}

fn ints(a: &NSArray<NSNumber>) -> Vec<usize> {
    a.iter().map(|n| n.integerValue() as usize).collect()
}

/// Function names of a multifunction model (its metadata.json `[{"functions": [{"name": …}]}]`), empty otherwise.
fn functions(model: &Path) -> Vec<String> {
    let Ok(s) = std::fs::read_to_string(model.join("metadata.json")) else { return Vec::new() };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else { return Vec::new() };
    v.get(0).and_then(|m| m.get("functions")).and_then(|f| f.as_array()).map_or_else(Vec::new, |f| {
        f.iter().filter_map(|f| f.get("name").and_then(|n| n.as_str()).map(String::from)).collect()
    })
}

impl Encoder {
    /// Every `*.mlmodelc` in `dir` (one per window), on the Neural Engine (CPU for what it can't run).
    /// `progress(done 0..1)` after each window (loading is where macOS compiles it, the first time ~30 s).
    pub fn load(dir: &Path, mut progress: impl FnMut(f64)) -> Result<Encoder, String> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("Neural Engine model folder {}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "mlmodelc"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("no Neural Engine model (*.mlmodelc) in {}", dir.display()));
        }
        // SAFETY (all blocks here): plain CoreML calls with valid, retained objects; arrays are written only within
        // the shape/strides CoreML reports.
        unsafe {
            let cfg = MLModelConfiguration::new();
            cfg.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
            let mut windows = Vec::new();
            // (model folder, function name or None for a plain model)
            let mut todo = Vec::new();
            for p in &paths {
                let names = functions(p);
                if names.is_empty() {
                    todo.push((p.clone(), None));
                }
                todo.extend(names.into_iter().map(|f| (p.clone(), Some(f))));
            }
            for (p, function) in &todo {
                let url = NSURL::from_file_path(p).ok_or_else(|| format!("bad path {}", p.display()))?;
                cfg.setFunctionName(function.as_deref().map(NSString::from_str).as_deref());
                let model = MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg)
                    .map_err(|e| err(&format!("loading {} {}", p.display(), function.as_deref().unwrap_or("")), e))?;
                let inputs = model.modelDescription().inputDescriptionsByName();
                let mel_desc = inputs.objectForKey(&NSString::from_str("mel")).ok_or("model has no `mel` input")?;
                let s = ints(&mel_desc.multiArrayConstraint().ok_or("`mel` is not an array")?.shape());
                if s.len() != 3 || s[2] != C {
                    return Err(format!("{}: `mel` shape {s:?}, expected [1, frames, {C}]", p.display()));
                }
                let mel = MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &shape(&s), MLMultiArrayDataType::Float32)
                    .map_err(|e| err("input array", e))?;
                windows.push(Window { model, frames: s[1], mel });
                progress(windows.len() as f64 / todo.len() as f64);
            }
            windows.sort_by_key(|w| w.frames);
            let length = MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &shape(&[1]), MLMultiArrayDataType::Int32)
                .map_err(|e| err("input array", e))?;
            Ok(Encoder { windows, length })
        }
    }

    /// Mel frames of the largest window (longest piece it takes).
    pub fn max_frames(&self) -> usize {
        self.windows.last().map_or(0, |w| w.frames)
    }

    /// Run every window once: the first run after a load is slow (~0.1-0.4 s per window).
    pub fn warm_up(&self) -> Result<(), String> {
        for w in &self.windows {
            self.encode(&vec![0.0; w.frames * C], w.frames)?;
        }
        Ok(())
    }

    /// mel [valid][128] (only the real frames, unpadded) -> encoder output [n][1024] f32 and n (valid encoder frames).
    pub fn encode(&self, mel: &[f32], valid: usize) -> Result<(Vec<f32>, usize), String> {
        let w = self.windows.iter().find(|w| w.frames >= valid)
            .ok_or_else(|| format!("piece of {valid} mel frames, longest window {}", self.max_frames()))?;
        assert!(mel.len() >= valid * C);
        unsafe {
            let p = w.mel.dataPointer().as_ptr() as *mut f32;
            let st = ints(&w.mel.strides());
            if st[2] != 1 {
                return Err(format!("`mel` strides {st:?}"));
            }
            for t in 0..w.frames {
                // SAFETY: frame t, bins 0..128 of an array of shape [1, frames, 128] with these strides.
                let row = std::slice::from_raw_parts_mut(p.add(t * st[1]), C);
                if t < valid {
                    row.copy_from_slice(&mel[t * C..(t + 1) * C]);
                } else {
                    row.fill(0.0);
                }
            }
            *(self.length.dataPointer().as_ptr() as *mut i32) = valid as i32;
            let (k1, k2) = (NSString::from_str("mel"), NSString::from_str("length"));
            let (v1, v2) = (MLFeatureValue::featureValueWithMultiArray(&w.mel), MLFeatureValue::featureValueWithMultiArray(&self.length));
            let dict = NSDictionary::<NSString, AnyObject>::from_slices(&[&*k1, &*k2], &[v1.as_ref(), v2.as_ref()]);
            let input = MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dict)
                .map_err(|e| err("input", e))?;
            let out = w.model.predictionFromFeatures_error(ProtocolObject::from_ref(&*input)).map_err(|e| err("Neural Engine", e))?;
            let e = out.featureValueForName(&NSString::from_str("encoder")).and_then(|v| v.multiArrayValue())
                .ok_or("model has no `encoder` output")?;
            // Valid encoder frames: three stride-2 convs (kernel 3, padding 1).
            let mut n = valid;
            for _ in 0..3 {
                n = (n - 1) / 2 + 1;
            }
            let (sh, st) = (ints(&e.shape()), ints(&e.strides()));
            if sh.len() != 3 || sh[2] != D {
                return Err(format!("`encoder` shape {sh:?}"));
            }
            n = n.min(sh[1]);
            let mut v = vec![0f32; n * D];
            let q = e.dataPointer().as_ptr();
            let dt = e.dataType();
            for t in 0..n {
                for c in 0..D {
                    let i = t * st[1] + c * st[2];
                    v[t * D + c] = if dt == MLMultiArrayDataType::Float16 {
                        half::f16::from_bits(*(q as *const u16).add(i)).to_f32()
                    } else if dt == MLMultiArrayDataType::Float32 {
                        *(q as *const f32).add(i)
                    } else {
                        return Err(format!("`encoder` data type {}", dt.0));
                    };
                }
            }
            Ok((v, n))
        }
    }
}
