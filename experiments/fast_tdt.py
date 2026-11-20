"""Faster greedy TDT decoding for onnx-asr Parakeet models (same result as onnx-asr's loop).

onnx-asr runs the whole decoder_joint model once per decoding step (~500 steps for 2 min of audio): the LSTM
prediction network, both joint projections and the 8198-wide output layer, each time with a GPU round trip.

Here the same weights are rebuilt into one "step" graph that, per call:
  - runs the LSTM on the last emitted token,
  - applies the joint to a window of the next K encoder frames at once (encoder side projected once up front),
  - returns only argmax token / argmax duration per frame (no 8198-wide copies back to the CPU).
The CPU walks the window with onnx-asr's exact rules and calls again only when a token is emitted (the LSTM
input changed) or the window is used up.

usage:  import fast_tdt; fast_tdt.install(model)   # model = onnx_asr.load_model(...)
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

WINDOW = 8


def _weights(path):
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    nodes = m.graph.node
    lstms = [n for n in nodes if n.op_type == "LSTM"]
    gather = next(n for n in nodes if n.op_type == "Gather")
    mm = [n for n in nodes if n.op_type == "MatMul"]  # joint.enc, joint.pred, joint_net output (graph order)
    def mm_bias(n):  # the Add right after a MatMul holds its bias
        add = next(a for a in nodes if a.op_type == "Add" and n.output[0] in a.input)
        return init[next(i for i in add.input if i in init)]
    return dict(
        embed=init[gather.input[0]],
        lstm=[(n, [init[i] for i in n.input[1:4]]) for n in lstms],
        enc_w=init[mm[0].input[1]], enc_b=mm_bias(mm[0]),
        pred_w=init[mm[1].input[1]], pred_b=mm_bias(mm[1]),
        out_w=init[mm[2].input[1]], out_b=mm_bias(mm[2]),
    )


def _step_graph(w, vocab, half=False):
    """token[1,1] i32, h/c[L,1,H], enc_win[K,640] -> td[K,2] i64 (argmax token, argmax duration), h_out, c_out."""
    t = lambda name, a: numpy_helper.from_array(np.ascontiguousarray(a) if np.ndim(a) else np.asarray(a), name)
    H = w["lstm"][0][1][1].shape[-1]
    L = len(w["lstm"])
    inits = [t("embed", w["embed"]), t("pred_w", w["pred_w"]), t("pred_b", w["pred_b"]),
             t("out_w", w["out_w"]), t("out_b", w["out_b"]),
             t("sq1", np.array([1], np.int64)), t("flat", np.array([1, -1], np.int64)),
             t("tok_s", np.array([0], np.int64)), t("tok_e", np.array([vocab], np.int64)),
             t("dur_e", np.array([1 << 30], np.int64)), t("ax", np.array([1], np.int64))]
    # LSTM written out as plain ops (ONNX gate order i, o, f, c; default sigmoid/tanh/tanh, no peepholes):
    # the cuDNN LSTM kernel has ~0.25 ms fixed cost per call, and plain ops can be captured in a CUDA graph.
    g = [helper.make_node("Cast", ["token"], ["tok64"], to=TensorProto.INT64),
         helper.make_node("Gather", ["embed", "tok64"], ["emb"]),          # [1, 1, 640]
         helper.make_node("Reshape", ["emb", "flat"], ["x0"])]             # [1, 640]
    inits.append(t("split4", np.array([H] * 4, np.int64)))
    x, hs, cs = "x0", [], []
    for i, (node, (W, R, B)) in enumerate(w["lstm"]):
        WR = np.concatenate([W[0], R[0]], axis=1).T                       # [in + H, 4H]
        inits += [t(f"WR{i}", WR), t(f"b{i}", B[0, :4 * H] + B[0, 4 * H:]), t(f"i{i}", np.array(i, np.int64))]
        g += [helper.make_node("Gather", ["h", f"i{i}"], [f"h{i}"], axis=0),   # [1, H]
              helper.make_node("Gather", ["c", f"i{i}"], [f"c{i}"], axis=0),
              helper.make_node("Concat", [x, f"h{i}"], [f"xh{i}"], axis=1),
              helper.make_node("MatMul", [f"xh{i}", f"WR{i}"], [f"gm{i}"]),
              helper.make_node("Add", [f"gm{i}", f"b{i}"], [f"g{i}"]),
              helper.make_node("Split", [f"g{i}", "split4"], [f"gi{i}", f"go{i}", f"gf{i}", f"gc{i}"], axis=1),
              helper.make_node("Sigmoid", [f"gi{i}"], [f"si{i}"]),
              helper.make_node("Sigmoid", [f"go{i}"], [f"so{i}"]),
              helper.make_node("Sigmoid", [f"gf{i}"], [f"sf{i}"]),
              helper.make_node("Tanh", [f"gc{i}"], [f"tc{i}"]),
              helper.make_node("Mul", [f"sf{i}", f"c{i}"], [f"fc{i}"]),
              helper.make_node("Mul", [f"si{i}", f"tc{i}"], [f"ic{i}"]),
              helper.make_node("Add", [f"fc{i}", f"ic{i}"], [f"cn{i}"]),     # new c [1, H]
              helper.make_node("Tanh", [f"cn{i}"], [f"tcn{i}"]),
              helper.make_node("Mul", [f"so{i}", f"tcn{i}"], [f"hn{i}"]),    # new h [1, H]
              helper.make_node("Unsqueeze", [f"hn{i}", "z0"], [f"ho{i}"]),
              helper.make_node("Unsqueeze", [f"cn{i}", "z0"], [f"co{i}"])]
        x = f"hn{i}"
        hs.append(f"ho{i}"); cs.append(f"co{i}")
    inits.append(t("z0", np.array([0], np.int64)))
    g += [helper.make_node("Concat", hs, ["h_out"], axis=0),
          helper.make_node("Concat", cs, ["c_out"], axis=0),
          helper.make_node("Reshape", [x, "flat"], ["p"]),                 # [1, H]
          helper.make_node("MatMul", ["p", "pred_w"], ["pm"]),
          helper.make_node("Add", ["pm", "pred_b"], ["pp"]),
          helper.make_node("Add", ["enc_win", "pp"], ["j"]),               # [K, 640]
          helper.make_node("Relu", ["j"], ["jr"]),
          helper.make_node("MatMul", ["jr", "out_w"], ["om"]),
          helper.make_node("Add", ["om", "out_b"], ["logits"]),            # [K, vocab + durations]
          helper.make_node("Slice", ["logits", "tok_s", "tok_e", "ax"], ["lt"]),
          helper.make_node("Slice", ["logits", "tok_e", "dur_e", "ax"], ["ld"]),
          helper.make_node("ArgMax", ["lt"], ["tok"], axis=1, keepdims=1),
          helper.make_node("ArgMax", ["ld"], ["dur"], axis=1, keepdims=1),
          helper.make_node("Concat", ["tok", "dur"], ["td"], axis=1)]       # [K, 2]: one small copy back
    if half:  # fp16 weights for the matrix multiplies (half the bytes read per call); everything else stays fp32
        by_name = {i.name: i for i in inits}
        g2 = []
        for n in g:
            if n.op_type == "MatMul":
                wt = by_name[n.input[1]]
                wt.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(wt).astype(np.float16), wt.name))
                g2 += [helper.make_node("Cast", [n.input[0]], [n.input[0] + "_h"], to=TensorProto.FLOAT16),
                       helper.make_node("MatMul", [n.input[0] + "_h", n.input[1]], [n.output[0] + "_h"]),
                       helper.make_node("Cast", [n.output[0] + "_h"], [n.output[0]], to=TensorProto.FLOAT)]
            else:
                g2.append(n)
        g = g2
    f = TensorProto.FLOAT
    graph = helper.make_graph(
        g, "tdt_step",
        [helper.make_tensor_value_info("token", TensorProto.INT32, [1, 1]),
         helper.make_tensor_value_info("h", f, [L, 1, H]), helper.make_tensor_value_info("c", f, [L, 1, H]),
         helper.make_tensor_value_info("enc_win", f, ["K", w["enc_w"].shape[1]])],
        [helper.make_tensor_value_info("td", TensorProto.INT64, ["K", 2]),
         helper.make_tensor_value_info("h_out", f, [L, 1, H]),
         helper.make_tensor_value_info("c_out", f, [L, 1, H])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString(), L, H


def _find(obj, depth=0):
    """the inner onnx-asr model object (the one holding _decoder_joint) inside its adapters"""
    if hasattr(obj, "_decoder_joint"):
        return obj
    if depth < 4 and hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            if hasattr(v, "__dict__") and not isinstance(v, type) and (r := _find(v, depth + 1)) is not None:
                return r
    return None


class _CudaStep:
    """step graph on CUDA with fixed device buffers and captured CUDA graphs (no per-op launch cost).
    Two bindings share the token/window inputs and swap the LSTM state buffers (A->B, B->A), so committing a new
    state is just switching binding; the state never leaves the GPU."""

    def __init__(self, blob, L, H, window, enc_dim):
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.s = ort.InferenceSession(blob, so, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1",
                                                                       "arena_extend_strategy": "kSameAsRequested"})])
        self.s._model_bytes = None  # ORT's Python wrapper keeps the ~45 MB graph bytes only to rebuild on fallback
        dev = lambda a: ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0)
        self.token = dev(np.zeros((1, 1), np.int32))
        self.win = dev(np.zeros((window, enc_dim), np.float32))
        st = [(dev(np.zeros((L, 1, H), np.float32)), dev(np.zeros((L, 1, H), np.float32))) for _ in range(2)]
        self.td = [dev(np.zeros((window, 2), np.int64)) for _ in range(2)]
        self.zero = np.zeros((L, 1, H), np.float32)
        self.st, self.bind, self.ro = st, [], []
        for k in range(2):
            b = self.s.io_binding()
            b.bind_ortvalue_input("token", self.token); b.bind_ortvalue_input("enc_win", self.win)
            b.bind_ortvalue_input("h", st[k][0]); b.bind_ortvalue_input("c", st[k][1])
            b.bind_ortvalue_output("h_out", st[1 - k][0]); b.bind_ortvalue_output("c_out", st[1 - k][1])
            b.bind_ortvalue_output("td", self.td[k])
            ro = ort.RunOptions(); ro.add_run_config_entry("gpu_graph_id", str(k + 1))
            self.bind.append(b); self.ro.append(ro)
        self.cur = 0

    def reset(self):
        self.cur = 0
        self.st[0][0].update_inplace(self.zero); self.st[0][1].update_inplace(self.zero)

    def run(self, token, win):
        self.token.update_inplace(np.array([[token]], np.int32))
        self.win.update_inplace(win)
        self.s.run_with_iobinding(self.bind[self.cur], self.ro[self.cur])
        return self.td[self.cur].numpy()

    def commit(self):  # new state = this call's output
        self.cur = 1 - self.cur


class _HostStep:
    """same graph, plain session.run (CPU, or any provider); state round-trips through numpy."""

    def __init__(self, blob, L, H, providers):
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.s = ort.InferenceSession(blob, so, providers=list(providers))
        self.s._model_bytes = None  # see _CudaStep
        self.L, self.H = L, H

    def reset(self):
        self.h = np.zeros((self.L, 1, self.H), np.float32); self.c = self.h.copy()

    def run(self, token, win):
        td, self.h2, self.c2 = self.s.run(None, {"token": np.array([[token]], np.int32), "h": self.h, "c": self.c,
                                                 "enc_win": win})
        return td

    def commit(self):
        self.h, self.c = self.h2, self.c2


def install(asr, decoder_path, providers=("CUDAExecutionProvider", "CPUExecutionProvider"), window=WINDOW,
            cuda_graph=True, half=False):
    """Replace the greedy TDT loop of an onnx-asr Parakeet TDT model (in place). Returns the model.
    decoder_path: the model's decoder_joint-model*.onnx (weights are read from it)."""
    m = _find(asr)
    path = str(decoder_path)
    w = _weights(path)
    blob, L, H = _step_graph(w, m._vocab_size, half)
    enc_dim = w["enc_w"].shape[1]
    if cuda_graph and "CUDAExecutionProvider" in providers:
        step = _CudaStep(blob, L, H, window, enc_dim)
    else:
        step = _HostStep(blob, L, H, providers)
    enc_w, enc_b = w["enc_w"], w["enc_b"]
    blank, max_per_step, orig = m._blank_idx, m._max_tokens_per_step, m._decoding

    def decoding(encoder_out, encoder_out_lens, /, **kwargs):
        if kwargs.get("need_logprobs"):
            yield from orig(encoder_out, encoder_out_lens, **kwargs)
            return
        for enc, n in zip(encoder_out, encoder_out_lens, strict=True):
            n = int(n)
            proj = np.zeros((n + window, enc_w.shape[1]), np.float32)
            proj[:n] = enc[:n] @ enc_w + enc_b
            step.reset()
            last = blank
            tokens, timestamps = [], []
            t = emitted = 0
            while t < n:
                start = t
                td = step.run(last, proj[t:t + window])
                while t < n and t - start < window:
                    token, d = int(td[t - start, 0]), int(td[t - start, 1])
                    if token != blank:
                        step.commit(); last = token
                        tokens.append(token); timestamps.append(t)
                        emitted += 1
                    if d > 0:
                        t += d; emitted = 0
                    elif token == blank or emitted == max_per_step:
                        t += 1; emitted = 0
                    if token != blank:
                        break  # LSTM input changed -> need a new call
            yield tokens, timestamps, None

    m._decoding = decoding
    # onnx-asr's own decoder_joint session is no longer used for decoding; free it (its GPU copy is ~140 MB) and
    # keep a CPU session that is only created if something still asks for it (need_logprobs fallback).
    m._decoder_joint = _LazyCpuSession(path)
    return asr


class _LazyCpuSession:
    """Stands in for an InferenceSession; loads the model on the CPU on first use."""

    def __init__(self, path):
        self._path, self._s = path, None

    def __getattr__(self, name):
        if self._s is None:
            so = ort.SessionOptions(); so.log_severity_level = 3
            self._s = ort.InferenceSession(self._path, so, providers=["CPUExecutionProvider"])
        return getattr(self._s, name)


def pad_encoder(asr, seconds, sample_rate=16000):
    """Pad encoder input features to a fixed length (`seconds` of audio) and pass the real length, so the encoder
    always sees one shape (one warm-up; no +70-90 ms first-run cost per new length). Padding is masked by the
    encoder via `length`. Inputs longer than `seconds` pass through unpadded. Returns the frame count."""
    m = _find(asr)
    feats, _ = m._preprocessor(np.zeros((1, int(seconds * sample_rate)), np.float32),
                               np.array([int(seconds * sample_rate)], np.int64))
    T = feats.shape[-1]
    orig = m._encode

    def encode(features, features_lens):
        if features.shape[-1] < T:
            pad = np.zeros((*features.shape[:-1], T - features.shape[-1]), features.dtype)
            features = np.concatenate([features, pad], axis=-1)
        enc, lens = orig(features, features_lens)
        return enc, lens

    m._encode = encode
    return T
