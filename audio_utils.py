"""Pure audio-trimming math used by the Audio Trim (Visual) node.

This module has no ComfyUI imports so it can be unit-tested standalone.
All functions work on numpy arrays shaped [channels, frames] (float32/float64,
nominal range -1.0 .. 1.0) and integer sample indices.
"""

import hashlib
import os
import re

import numpy as np

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:  # soundfile ships with ComfyUI; tests may stub it
    sf = None
    _HAS_SF = False

# Search window (seconds) when snapping a cut point to a zero crossing.
# The frontend uses the identical value so the preview and the saved file agree.
ZERO_SNAP_WINDOW_S = 0.02

MIN_SELECTION_S = 0.001  # keep at least 1 ms of audio between the cut points


def find_quietest_sample(mono, idx, window):
    """Return the index within [idx-window, idx+window] with the smallest |sample|.

    Ties (e.g. flat or all-zero regions) resolve to the candidate CLOSEST to
    idx, then to the LOWEST index. This makes snapping idempotent and is
    replicated exactly by the JS frontend, so both sides agree on the sample.
    """
    n = int(mono.shape[0])
    idx = int(np.clip(idx, 0, n - 1))
    lo = max(0, idx - int(window))
    hi = min(n - 1, idx + int(window))
    if hi <= lo:
        return idx
    seg = np.abs(mono[lo:hi + 1])
    cands = np.nonzero(seg == seg.min())[0]  # ascending order
    dist = np.abs((cands + lo) - idx)
    return lo + int(cands[int(np.argmin(dist))])


def snap_indices(mono, start_idx, end_idx, sr, enabled):
    """Snap both cut points to the nearest zero crossing (quietest sample).

    If snapping makes the selection empty or inverted (can only happen for
    selections shorter than the snap window), the raw indices are returned.
    """
    if not enabled:
        return int(start_idx), int(end_idx)
    window = max(1, int(round(ZERO_SNAP_WINDOW_S * float(sr))))
    s = find_quietest_sample(mono, start_idx, window)
    e = find_quietest_sample(mono, end_idx, window)
    if s >= e:
        return int(start_idx), int(end_idx)
    return s, e


def compute_trim_indices(n, sr, start_s, end_s):
    """Convert seconds to clamped sample indices [start_idx, end_idx).

    end_s <= 0 means "end of file". Values are clamped so that
    0 <= start_idx < end_idx <= n always holds (at least MIN_SELECTION_S).
    """
    n = int(n)
    sr = float(sr)
    start_idx = int(round(float(start_s) * sr))
    if float(end_s) <= 0.0:
        end_idx = n
    else:
        end_idx = int(round(float(end_s) * sr))
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(end_idx, start_idx + 1)
    end_idx = min(end_idx, n)
    if end_idx <= start_idx:  # degenerate 1-frame / empty file guard
        end_idx = min(n, start_idx + 1)
    return start_idx, end_idx


def apply_gain(data, gain_db):
    """Apply a dB gain to [channels, frames] float data and hard-clip to [-1, 1].

    Returns (clipped_data, clipped_fraction) where clipped_fraction is the
    fraction of samples that exceeded the full-scale range before clipping.
    """
    gain_db = float(gain_db)
    data = np.asarray(data, dtype=np.float32)
    if gain_db == 0.0 or data.size == 0:
        return data, 0.0
    linear = 10.0 ** (gain_db / 20.0)
    raw = data * np.float32(linear)
    clipped_fraction = float(np.mean(np.abs(raw) > 1.0)) if raw.size else 0.0
    out = np.clip(raw, -1.0, 1.0, out=raw)
    return out.astype(np.float32, copy=False), clipped_fraction


def mono_mixdown(data):
    """[channels, frames] -> mono 1-D array (mean of channels)."""
    data = np.asarray(data)
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    if data.shape[0] == 1:
        return data[0].astype(np.float32, copy=False)
    return data.mean(axis=0, dtype=np.float32)


def duration_seconds(start_idx, end_idx, sr):
    """Exact trimmed duration in seconds (float), for display and reporting."""
    return (int(end_idx) - int(start_idx)) / float(sr)


def format_duration(start_idx, end_idx, sr):
    """Return (seconds_text, milliseconds_int) for the trimmed region."""
    dur = duration_seconds(start_idx, end_idx, sr)
    ms = int(round(dur * 1000.0))
    return f"{dur:.3f}", ms


def audio_fingerprint(waveform, sample_rate):
    """Stable content identity of the input audio (first batch item).

    The frontend compares fingerprints to tell a re-queued SAME file (keep
    the trim borders) from a NEW file (reset them to the file's start/end).
    Deterministic: float32 bytes of batch item 0 + the sample rate, blake2b.
    """
    wf = np.ascontiguousarray(np.asarray(waveform)[0], dtype=np.float32)
    h = hashlib.blake2b(wf.tobytes(), digest_size=16)
    h.update(int(sample_rate).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def sanitize_prefix(prefix):
    """Save-node filename string -> (subfolder, base) for the output folder.

    Keeps '/' subfolder support (output/<sub>/<base>_00001_.flac) while
    refusing traversal: '', '.' and '..' segments are dropped, and the base
    keeps only filename-safe characters. A typed .wav/.flac extension is
    stripped (the node appends .flac itself).
    """
    prefix = str(prefix or "").strip().replace("\\", "/")
    parts = [p for p in prefix.split("/") if p not in ("", ".", "..")]
    sub = "/".join(parts[:-1])
    base = parts[-1] if parts else "trim"
    base = re.sub(r"\.(wav|flac)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[^\w\-. ]+", "_", base).strip() or "trim"
    return sub, base


def save_trimmed_flac(mirror_name, temp_dir, output_dir, start_s, end_s,
                      gain_db, prefix):
    """Trim the temp mirror of the last run and write a FLAC into output_dir.

    The UI POSTs its ALREADY-snapped cut values, so no re-snap happens here:
    the saved cut matches the timeline exactly. Gain and clipping use the
    same apply_gain() as the AUDIO output socket. Returns a status dict
    {name, subfolder, seconds, sample_rate, channels, clipped_pct}.
    """
    if not _HAS_SF:
        raise RuntimeError("soundfile is required for the FLAC export.")
    name = os.path.basename(str(mirror_name or ""))
    if not (name.startswith("zat_full_") and name.endswith(".wav")):
        raise ValueError("unknown audio mirror — queue the node once first.")
    mirror = os.path.join(temp_dir, name)
    if not os.path.isfile(mirror):
        raise RuntimeError(
            "the temp audio mirror is gone (temp was cleared?) — queue the "
            "node once more, then press Save again.")
    data, sr = sf.read(mirror, dtype="float32", always_2d=True)  # [N, C]
    data = np.ascontiguousarray(data.T)                          # [C, N]
    n = int(data.shape[1])
    s_idx, e_idx = compute_trim_indices(n, sr, start_s, end_s)
    trimmed, clipped = apply_gain(
        np.ascontiguousarray(data[:, s_idx:e_idx]), gain_db)
    sub, base = sanitize_prefix(prefix)
    full_dir = os.path.join(output_dir, sub) if sub else output_dir
    os.makedirs(full_dir, exist_ok=True)
    # ComfyUI-style unique name: <base>_00001_.flac, counter per base
    counter = 1
    pat = re.compile(re.escape(base) + r"_(\d{5})_\.flac$")
    for fn in os.listdir(full_dir):
        m = pat.match(fn)
        if m:
            counter = max(counter, int(m.group(1)) + 1)
    fname = f"{base}_{counter:05}_.flac"
    sf.write(os.path.join(full_dir, fname), trimmed.T, sr,
             format="FLAC", subtype="PCM_24")
    return {
        "name": fname,
        "subfolder": sub,
        "seconds": round(duration_seconds(s_idx, e_idx, sr), 3),
        "sample_rate": int(sr),
        "channels": int(trimmed.shape[0]),
        "clipped_pct": round(clipped * 100.0, 2),
    }
