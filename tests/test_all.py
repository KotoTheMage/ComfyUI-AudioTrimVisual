"""Standalone tests for ComfyUI-AudioTrimVisual.

Run:  python3 tests/test_all.py
No pytest required; exits non-zero on failure.

Part 1: unit tests for audio_utils.py (pure trimming math).
Part 2: integration tests for nodes.py run() using a stubbed folder_paths
        module, a generated WAV file, and a torch stub for the AUDIO socket
        round-trip (real ComfyUI always ships real torch; this sandbox may
        not have it, so a minimal fake is injected in that case).
"""

import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- stubbing
class _FolderPathsStub:
    """Minimal folder_paths stand-in so nodes.py imports outside ComfyUI."""

    def __init__(self, input_dir, output_dir, temp_dir):
        self.input_dir = str(input_dir)
        self.output_dir = str(output_dir)
        self.temp_dir = str(temp_dir)

    def get_filename_list(self, kind):
        raise KeyError(kind)  # force the os.listdir fallback path

    def get_input_directory(self):
        return self.input_dir

    def get_output_directory(self):
        return self.output_dir

    def get_temp_directory(self):
        return self.temp_dir

    def get_annotated_filepath(self, name):
        return os.path.join(self.input_dir, str(name))

    def exists_annotated_filepath(self, name):
        return os.path.isfile(self.get_annotated_filepath(name))

    def get_save_image_path(self, prefix, output_dir):
        prefix = str(prefix).replace("\\", "/")
        parts = prefix.split("/")
        subfolder = "/".join(parts[:-1])
        base = parts[-1]
        full = os.path.join(output_dir, subfolder) if subfolder else output_dir
        # extension-agnostic counter, like ComfyUI's get_save_image_path
        counter = 1
        if os.path.isdir(full):
            counter += sum(1 for f in os.listdir(full) if f.startswith(base + "_"))
        return full, base, counter, subfolder, prefix


class _FakeTensor:
    """Tiny stand-in for torch.Tensor: just the methods nodes.py uses."""

    def __init__(self, arr):
        self._arr = np.ascontiguousarray(arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeTorch:
    """Tiny stand-in for the torch module (from_numpy only)."""

    @staticmethod
    def from_numpy(arr):
        return _FakeTensor(np.asarray(arr))


def load_torch_module():
    """Use real torch when available, else inject the fake into sys.modules."""
    try:
        import torch

        return torch
    except Exception:
        fake = _FakeTorch()
        sys.modules["torch"] = fake
        return fake


def unwrap(waveform):
    """torch/fake AUDIO waveform tensor [B, C, N] -> float32 numpy."""
    return waveform.detach().cpu().numpy().astype(np.float32)


def make_sine_wav(path, seconds=1.5, sr=44100, freq=440.0, amp=0.5, stereo=True):
    import soundfile as sf

    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64)
    wave = (amp * np.sin(2 * math.pi * freq * t / sr)).astype(np.float32)
    data = np.stack([wave, wave], axis=1) if stereo else wave.reshape(-1, 1)
    sf.write(str(path), data, sr, subtype="PCM_16", format="WAV")
    return n


# ------------------------------------------------------------- assertions
CHECKS = []


def check(name, fn):
    try:
        fn()
        CHECKS.append((name, True, ""))
        print(f"  PASS  {name}")
    except AssertionError as e:
        CHECKS.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:  # noqa: BLE001
        CHECKS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


def approx(a, b, tol):
    return abs(a - b) <= tol


# ----------------------------------------------------------------- part 1
def run_unit_tests(u):
    sr = 44100
    n = 2 * sr

    def t_trim_basic():
        s, e = u.compute_trim_indices(n, sr, 0.5, 1.25)
        assert (s, e) == (22050, 55125), (s, e)
    check("compute_trim_indices basic", t_trim_basic)

    def t_trim_eof():
        s, e = u.compute_trim_indices(n, sr, 0.0, 0.0)
        assert (s, e) == (0, n), (s, e)
    check("compute_trim_indices end<=0 means EOF", t_trim_eof)

    def t_trim_clamp_start():
        s, e = u.compute_trim_indices(n, sr, 5.0, 6.0)
        assert s == n - 1 and e == n, (s, e)
    check("compute_trim_indices clamps start beyond EOF", t_trim_clamp_start)

    def t_trim_inverted():
        s, e = u.compute_trim_indices(n, sr, 1.0, 0.5)
        assert e > s, (s, e)
        assert e - s <= round(0.001 * sr) + 1, (s, e)
    check("compute_trim_indices fixes inverted range", t_trim_inverted)

    def t_trim_ms_rounding():
        s, e = u.compute_trim_indices(n, sr, 0.1234, 0.9876)
        assert s == round(0.1234 * sr) and e == round(0.9876 * sr), (s, e)
    check("compute_trim_indices ms rounding", t_trim_ms_rounding)

    # flat ±0.9 signal with one planted exact zero at a known index:
    # every non-zero sample has |amplitude| 0.9, so any quiet sample the
    # snapper finds must be the planted zero
    zero_idx = 30000
    mono = np.where(np.arange(n) % 2 == 0, 0.9, -0.9).astype(np.float32)
    mono[zero_idx] = 0.0

    def t_quietest():
        i = u.find_quietest_sample(mono, zero_idx + 500, int(0.02 * sr))
        assert i == zero_idx, (i, zero_idx)
    check("find_quietest_sample finds planted zero", t_quietest)

    def t_quietest_window():
        far = zero_idx + int(0.05 * sr)  # 2205 frames: outside the 20 ms window
        w = int(0.02 * sr)
        i = u.find_quietest_sample(mono, far, w)
        assert abs(i - far) <= w, (i, far, w)  # result stays inside the window
        assert i != zero_idx, (i, zero_idx)    # ... and the zero is too far
    check("find_quietest_sample respects window", t_quietest_window)

    def t_snap_enabled():
        s, e = u.snap_indices(mono, zero_idx + 700, zero_idx + 3000, sr, True)
        assert s == zero_idx, (s, e)
    check("snap_indices snaps start to planted zero", t_snap_enabled)

    def t_snap_disabled():
        s, e = u.snap_indices(mono, 12345, 54321, sr, False)
        assert (s, e) == (12345, 54321), (s, e)
    check("snap_indices disabled keeps exact values", t_snap_disabled)

    def t_snap_collision():
        # selection shorter than the snap window: raw values must survive
        s, e = u.snap_indices(mono, 10000, 10001, sr, True)
        assert e > s, (s, e)
    check("snap_indices collision guard", t_snap_collision)

    def t_snap_idempotent():
        for raw in [(29000, 31000), (zero_idx + 700, zero_idx + 3000),
                    (zero_idx, zero_idx + 4000)]:
            s1, e1 = u.snap_indices(mono, raw[0], raw[1], sr, True)
            s2, e2 = u.snap_indices(mono, s1, e1, sr, True)
            assert (s1, e1) == (s2, e2), ((s1, e1), (s2, e2))
    check("snap_indices is idempotent (JS/Python agreement)", t_snap_idempotent)

    data = np.tile(np.linspace(-0.4, 0.4, 1000, dtype=np.float32), (2, 1))

    def t_gain_up():
        out, clipped = u.apply_gain(data, 6.0206)
        assert approx(out[0, 10], data[0, 10] * 2.0, 1e-6), (out[0, 10], data[0, 10])
        assert clipped == 0.0
    check("apply_gain +6 dB doubles amplitude", t_gain_up)

    def t_gain_down():
        out, clipped = u.apply_gain(data, -6.0206)
        assert approx(out[0, 10], data[0, 10] * 0.5, 1e-6)
        assert clipped == 0.0
    check("apply_gain -6 dB halves amplitude", t_gain_down)

    def t_gain_clip():
        loud = np.full((1, 100), 0.9, dtype=np.float32)
        out, clipped = u.apply_gain(loud, 6.0206)  # x2 -> all samples clip
        assert clipped == 1.0, clipped
        assert np.all(np.abs(out) <= 1.0)
    check("apply_gain clips and reports fraction", t_gain_clip)

    def t_gain_zero():
        out, clipped = u.apply_gain(data, 0.0)
        assert np.array_equal(out, data.astype(np.float32)) and clipped == 0.0
    check("apply_gain 0 dB is a no-op", t_gain_zero)

    def t_mixdown():
        m = u.mono_mixdown(data)
        assert m.shape == (1000,) and approx(m[5], data[:, 5].mean(), 1e-6)
    check("mono_mixdown averages channels", t_mixdown)

    def t_duration():
        s_txt, ms = u.format_duration(0, 44100, sr)
        assert s_txt == "1.000" and ms == 1000, (s_txt, ms)
    check("format_duration seconds + ms", t_duration)

    def t_duration_round():
        _, ms = u.format_duration(0, 44211, sr)  # 1.00281... s
        assert ms == 1003, ms
    check("format_duration rounds to nearest ms", t_duration_round)

    def t_fp_deterministic():
        rng = np.random.default_rng(7)
        wf = rng.standard_normal((2, 2, 1000)).astype(np.float32)
        a = u.audio_fingerprint(wf, 44100)
        assert a == u.audio_fingerprint(wf.copy(), 44100)
        assert len(a) == 32 and int(a, 16) >= 0  # 32-char hex digest
    check("audio_fingerprint deterministic for identical audio",
          t_fp_deterministic)

    def t_fp_distinguishes():
        rng = np.random.default_rng(7)
        wf = rng.standard_normal((1, 1, 1000)).astype(np.float32)
        wf2 = wf.copy()
        wf2[0, 0, 500] += 0.001
        assert u.audio_fingerprint(wf, 44100) != u.audio_fingerprint(wf2, 44100)
        assert u.audio_fingerprint(wf, 44100) != u.audio_fingerprint(wf, 22050)
    check("audio_fingerprint distinguishes content and sample rate",
          t_fp_distinguishes)


# ----------------------------------------------------------------- part 2
def run_integration_tests(pkg, stub, tmp, torch_mod):
    import soundfile as sf

    input_dir = tmp / "input"
    output_dir = tmp / "output"
    temp_dir = tmp / "temp"
    for d in (input_dir, output_dir, temp_dir):
        d.mkdir(exist_ok=True)
    stub.input_dir = str(input_dir)
    stub.output_dir = str(output_dir)
    stub.temp_dir = str(temp_dir)

    node_cls = pkg.NODE_CLASS_MAPPINGS["AudioTrimVisual"]
    node = node_cls()

    src = input_dir / "take1.wav"
    make_sine_wav(src, seconds=1.5, sr=44100)

    def audio_input(waveform_bcn=None):
        """Build the core AUDIO dict the node receives through its socket."""
        if waveform_bcn is None:
            data, _sr = sf.read(str(src), always_2d=True)  # [frames, channels]
            waveform_bcn = np.ascontiguousarray(data.T, dtype=np.float32)[None]
        return {"waveform": torch_mod.from_numpy(waveform_bcn),
                "sample_rate": 44100}

    def t_run_basic():
        res = node.run(audio_input(), 0.5, 1.0, 0.0, False)
        out = res["result"][0]
        assert set(out.keys()) == {"waveform", "sample_rate"}, out.keys()
        assert int(out["sample_rate"]) == 44100
        arr = unwrap(out["waveform"])
        assert arr.shape == (1, 2, 22050), arr.shape  # exactly 0.500 s
        text = res["ui"]["text"][0]
        assert "trimmed 0.500s (500 ms)" in text.replace(" ", " "), text
        assert "500 ms" in text
        # 0 dB gain keeps the 0.5 amplitude of the source
        assert approx(float(np.max(np.abs(arr))), 0.5, 0.01), np.max(np.abs(arr))
    check("run(): socket in/out, trim 0.5-1.0 s -> 500 ms", t_run_basic)

    def t_no_output_files():
        node.run(audio_input(), 0.0, 0.5, 0.0, False)
        stray = [str(p) for p in output_dir.rglob("*") if p.is_file()]
        assert not stray, f"node wrote output files: {stray[:3]}"
    check("run(): writes nothing into the output folder", t_no_output_files)

    def t_run_eof():
        res = node.run(audio_input(), 1.0, 0.0, 0.0, False)
        arr = unwrap(res["result"][0]["waveform"])
        assert arr.shape[2] == 22050, arr.shape  # 1.5 - 1.0 s
    check("run(): end=0 trims to end of file", t_run_eof)

    def t_run_snap():
        res = node.run(audio_input(), 0.5, 1.0, 0.0, True)
        arr = unwrap(res["result"][0]["waveform"])
        # snap may move either cut by at most +/-20 ms
        frames = arr.shape[2]
        assert abs(frames - 22050) <= 0.02 * 44100 + 2, frames
    check("run(): zero-crossing snap stays within +/-20 ms", t_run_snap)

    def t_run_gain():
        res = node.run(audio_input(), 0.5, 1.0, -6.0206, False)
        arr = unwrap(res["result"][0]["waveform"])
        assert approx(float(np.max(np.abs(arr))), 0.25, 0.02), np.max(np.abs(arr))
    check("run(): -6 dB gain halves amplitude", t_run_gain)

    def t_run_passthrough():
        # trim 0..0 + 0 dB: the socket output equals the socket input
        src_arr = unwrap(audio_input()["waveform"])
        res = node.run(audio_input(), 0.0, 0.0, 0.0, False)
        arr = unwrap(res["result"][0]["waveform"])
        assert arr.shape == src_arr.shape, (arr.shape, src_arr.shape)
        assert np.allclose(arr, src_arr, atol=1e-6)
    check("run(): 0/0 trim + 0 dB passes the input through unchanged",
          t_run_passthrough)

    def t_run_batch():
        data, _ = sf.read(str(src), always_2d=True)
        wf = np.ascontiguousarray(data.T, dtype=np.float32)
        batch = np.ascontiguousarray(np.stack([wf, wf * 0.5]))  # [2, C, N]
        res = node.run(audio_input(batch), 0.5, 1.0, 0.0, False)
        arr = unwrap(res["result"][0]["waveform"])
        assert arr.shape == (2, 2, 22050), arr.shape
        assert approx(float(np.max(np.abs(arr[0]))), 0.5, 0.01), np.max(np.abs(arr[0]))
        assert approx(float(np.max(np.abs(arr[1]))), 0.25, 0.01), np.max(np.abs(arr[1]))
    check("run(): batch items trimmed together through the socket", t_run_batch)

    def t_ui_full_mirror():
        res = node.run(audio_input(), 0.0, 0.5, 0.0, False)
        full = res["ui"].get("zat_full", [None])[0]
        assert full, sorted(res["ui"].keys())
        assert full["type"] == "temp" and full["subfolder"] == "", full
        assert full["filename"].startswith("zat_full_"), full
        assert full["filename"].endswith(".wav"), full
        p = temp_dir / full["filename"]
        assert p.is_file(), p
        data, sr = sf.read(str(p), always_2d=True)
        assert sr == 44100 and data.shape[0] == 66150, data.shape  # full input
    check("run(): untrimmed input mirrored to temp for the UI timeline",
          t_ui_full_mirror)

    def t_ui_fingerprint():
        res = node.run(audio_input(), 0.0, 0.5, 0.0, False)
        full = res["ui"].get("zat_full", [None])[0]
        assert full and isinstance(full.get("hash"), str), full
        h1 = full["hash"]
        assert len(h1) == 32 and all(c in "0123456789abcdef" for c in h1), h1
        # the SAME input re-queued must report the SAME fingerprint ...
        res2 = node.run(audio_input(), 0.0, 0.5, 0.0, False)
        assert res2["ui"]["zat_full"][0]["hash"] == h1
        # ... and altered content must report a DIFFERENT one
        data, _ = sf.read(str(src), always_2d=True)
        wf2 = np.ascontiguousarray(data.T, dtype=np.float32)[None] * 0.5
        res3 = node.run(audio_input(wf2), 0.0, 0.5, 0.0, False)
        assert res3["ui"]["zat_full"][0]["hash"] != h1
    check("run(): ui fingerprint — same audio keeps hash, new audio changes it",
          t_ui_fingerprint)

    def t_ui_mirror_best_effort():
        # a broken temp folder must not fail the node: the mirror is
        # best-effort, the socket result is still returned
        def _boom():
            raise RuntimeError("temp broken")
        orig = stub.get_temp_directory
        stub.get_temp_directory = _boom
        try:
            res = node.run(audio_input(), 0.0, 0.5, 0.0, False)
        finally:
            stub.get_temp_directory = orig
        assert "zat_full" not in res["ui"], sorted(res["ui"].keys())
        assert unwrap(res["result"][0]["waveform"]).shape[2] == 22050
    check("run(): temp-mirror failure stays best-effort (no crash)",
          t_ui_mirror_best_effort)

    def t_run_bad_input():
        for bad in ("take1.wav", 42, {"sample_rate": 44100}):
            try:
                node.run(bad, 0, 1, 0.0, False)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {bad!r}")
    check("run(): non-AUDIO-dict input raises a clear error", t_run_bad_input)

    def t_run_clamped_start():
        # start beyond EOF still produces a valid (clamped) tiny result
        res = node.run(audio_input(), 10.0, 11.0, 0.0, False)
        arr = unwrap(res["result"][0]["waveform"])
        assert arr.shape[2] >= 1, arr.shape
    check("run(): out-of-range start is clamped", t_run_clamped_start)

    def t_playhead_cue_wired():
        js = (PKG_DIR / "web" / "js" / "audio_trim.js").read_text("utf-8")
        assert '{ kind: "seek" }' in js          # body click/drag moves the line
        assert "function seekTo" in js           # live jump while playing
        assert "st.playFrom" in js               # playback starts at the cue
        assert "st.playing ? st.playhead : st.cue" in js  # idle marker drawn
        assert "play([st.selStart, st.selEnd], st.cue)" in js  # Trim button
    check("play-start marker: draggable line + playback start wired",
          t_playhead_cue_wired)

    def t_input_types():
        it = node_cls.INPUT_TYPES()
        audio_spec = it["required"]["audio"]
        assert tuple(audio_spec) == ("AUDIO",), audio_spec  # socket, not combo
        assert set(it["required"].keys()) == {
            "audio", "start_seconds", "end_seconds", "gain_db",
            "snap_to_zero_crossing"}, sorted(it["required"].keys())
        assert it["required"]["start_seconds"][1]["step"] == 0.001
        assert it["required"]["gain_db"][1]["default"] == 0.0
        assert node_cls.RETURN_TYPES == ("AUDIO",), node_cls.RETURN_TYPES
        assert node_cls.OUTPUT_NODE is True
    check("INPUT_TYPES: AUDIO socket in/out, no save-format/prefix widgets",
          t_input_types)

    def t_fingerprint_wired():
        py = (PKG_DIR / "nodes.py").read_text(encoding="utf-8")
        assert "audio_utils.audio_fingerprint(wf, sr)" in py
        js = (PKG_DIR / "web" / "js" / "audio_trim.js").read_text("utf-8")
        assert "entry.hash !== st.srcHash" in js       # same-file guard
        assert "serializeValue = () => st.srcHash" in js  # workflow persistence
    check("same-file trim persistence wired backend <-> frontend",
          t_fingerprint_wired)


def main():
    print(f"Package: {PKG_DIR}")

    # torch for the AUDIO socket: real torch when present (any ComfyUI), else
    # the minimal fake above so nodes.py's guarded import works in this sandbox
    torch_mod = load_torch_module()

    # load the package ONCE: nodes.py binds the stubbed folder_paths at import
    # time, so unit and integration tests must share that single binding
    stub = _FolderPathsStub("/tmp/nonexistent_in", "/tmp/nonexistent_out",
                            "/tmp/nonexistent_temp")
    sys.modules["folder_paths"] = stub
    spec = importlib.util.spec_from_file_location(
        "comfyui_audio_trim", PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_audio_trim"] = pkg
    spec.loader.exec_module(pkg)

    print("Unit tests (audio_utils):")
    run_unit_tests(pkg.audio_utils)

    with tempfile.TemporaryDirectory(prefix="zat_test_") as td:
        print("Integration tests (nodes.run with stubbed folder_paths):")
        run_integration_tests(pkg, stub, Path(td), torch_mod)

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed.")
    if failed:
        for name, _, err in failed:
            print(f"  FAILED: {name} — {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
