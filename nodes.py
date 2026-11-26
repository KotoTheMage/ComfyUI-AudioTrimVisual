"""Audio Trim (Visual) — ComfyUI node backend.

Mid-pipe audio node, wired like the native audio nodes: an AUDIO socket input
on the left (connect Load Audio, Generate Audio, TTS, ...) and an AUDIO socket
output on the right (connect Save Audio, Preview Audio, ...). The node trims
the incoming waveform to [start_seconds, end_seconds] (optionally snapped to
zero crossings), applies a dB gain and passes the trimmed AUDIO through the
output socket unchanged in format. Persisting the result is the job of the
node connected downstream (Save Audio owns the filename prefix and the
output format); this node only mirrors the untrimmed input to the temp folder
so the frontend timeline can display the full waveform after every run.

The AUDIO socket format is the one used by all core audio nodes:
{"waveform": torch tensor [batch, channels, frames] float32, "sample_rate": int}
"""

import os
import uuid

import numpy as np

import folder_paths

from . import audio_utils

try:
    import torch
    _HAS_TORCH = True
except Exception:  # torch ships with ComfyUI; tests may stub it
    torch = None
    _HAS_TORCH = False

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:  # pragma: no cover - soundfile ships with ComfyUI
    sf = None
    _HAS_SF = False


class AudioTrimVisual:
    OUTPUT_NODE = True
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "audio"
    DESCRIPTION = (
        "Mid-pipe visual audio trimmer. Connect an AUDIO source on the left "
        "(Load Audio, Generate Audio, TTS, ...) and feed the trimmed result "
        "from the right AUDIO socket into Save Audio etc. Drag the waveform "
        "handles or type exact start/end values (seconds or milliseconds), "
        "optionally snap cuts to zero crossings, preview with playback, apply "
        "a dB gain. end_seconds = 0 means 'end of file'. The trimmed audio "
        "goes to the output socket; connect Save Audio to write it to disk "
        "(it owns the filename prefix and the output format). Queue the node "
        "once to display the waveform."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # AUDIO socket input (same format as Load Audio / Save Audio)
                "audio": ("AUDIO",),
                "start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000000.0,
                    "step": 0.001, "round": 0.001,
                }),
                "end_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000000.0,
                    "step": 0.001, "round": 0.001,
                }),
                "gain_db": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 24.0, "step": 0.1,
                }),
                "snap_to_zero_crossing": ("BOOLEAN", {"default": True}),
            },
        }

    # ------------------------------------------------------ socket helpers
    @staticmethod
    def _to_numpy(waveform):
        """AUDIO waveform (torch [B, C, N] tensor or ndarray) -> float32 numpy."""
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        return np.ascontiguousarray(waveform, dtype=np.float32)

    @staticmethod
    def _to_socket(waveform_np, sample_rate):
        """[B, C, N] float32 numpy -> core AUDIO dict for the output socket."""
        if not _HAS_TORCH:
            raise RuntimeError(
                "torch is required for the AUDIO output socket (it always "
                "ships with ComfyUI)."
            )
        return {
            "waveform": torch.from_numpy(
                np.ascontiguousarray(waveform_np, dtype=np.float32)),
            "sample_rate": int(sample_rate),
        }

    # ------------------------------------------------------------------- run
    def run(self, audio, start_seconds, end_seconds, gain_db,
            snap_to_zero_crossing):
        if not isinstance(audio, dict) or "waveform" not in audio:
            raise ValueError(
                "'audio' must be a core AUDIO dict (waveform + sample_rate). "
                "Connect an audio source such as Load Audio or Generate Audio "
                "to this node's input socket."
            )
        wf = self._to_numpy(audio["waveform"])  # [B, C, N]
        if wf.ndim != 3 or wf.shape[2] == 0:
            raise ValueError(
                f"'audio' waveform has invalid shape {tuple(wf.shape)} "
                "(expected [batch, channels, frames] with at least one frame)."
            )
        sr = int(audio["sample_rate"])
        n_frames = int(wf.shape[2])
        total_s = n_frames / float(sr)

        # 1) numeric trim (ms-precision, clamped, end<=0 -> end of file)
        s_idx, e_idx = audio_utils.compute_trim_indices(
            n_frames, sr, start_seconds, end_seconds)

        # 2) zero-crossing snap on a mono mixdown of the first batch item
        #    (same algorithm as the UI); the indices apply to every item
        mono = audio_utils.mono_mixdown(wf[0])
        s_idx, e_idx = audio_utils.snap_indices(
            mono, s_idx, e_idx, sr, bool(snap_to_zero_crossing))

        # 3) gain (with clipping guard) on the trimmed region of all items
        trimmed = np.ascontiguousarray(wf[..., s_idx:e_idx], dtype=np.float32)
        trimmed, clipped = audio_utils.apply_gain(trimmed, gain_db)

        # 4) trimmed AUDIO through the output socket
        audio_out = self._to_socket(trimmed, sr)

        # 5) mirror the untrimmed input (first batch item) to the temp folder
        #    so the frontend timeline can show the full waveform to trim; the
        #    content fingerprint lets the UI keep the trim borders when the
        #    SAME audio is re-queued and reset them for a NEW file
        full_entry = None
        if _HAS_SF:
            try:
                temp_dir = folder_paths.get_temp_directory()
                os.makedirs(temp_dir, exist_ok=True)
                full_name = f"zat_full_{uuid.uuid4().hex[:12]}.wav"
                sf.write(os.path.join(temp_dir, full_name),
                         np.ascontiguousarray(wf[0].T, dtype=np.float32),
                         sr, format="WAV", subtype="FLOAT")
                full_entry = {"filename": full_name, "subfolder": "",
                              "type": "temp"}
                try:
                    full_entry["hash"] = audio_utils.audio_fingerprint(wf, sr)
                except Exception:
                    pass  # fingerprint is an optimization, mirror still works
            except Exception:
                full_entry = None  # UI mirror is best-effort only

        dur_s, dur_ms = audio_utils.format_duration(s_idx, e_idx, sr)
        snapped = "on" if snap_to_zero_crossing else "off"
        msg = (
            f"trimmed {dur_s}s ({dur_ms} ms) | "
            f"cut {s_idx / sr:.3f}s to {e_idx / sr:.3f}s of {total_s:.3f}s "
            f"@ {sr} Hz, {wf.shape[1]} ch | snap: {snapped} | "
            f"gain: {float(gain_db):+.1f} dB | AUDIO -> output socket"
        )
        if clipped > 0.0:
            msg += f" | WARNING: {clipped * 100:.2f}% of samples clipped by gain"

        ui = {"text": [msg]}
        if full_entry is not None:
            ui["zat_full"] = [full_entry]
        return {"ui": ui, "result": (audio_out,)}


NODE_CLASS_MAPPINGS = {"AudioTrimVisual": AudioTrimVisual}
NODE_DISPLAY_NAME_MAPPINGS = {"AudioTrimVisual": "Audio Trim (Visual)"}
