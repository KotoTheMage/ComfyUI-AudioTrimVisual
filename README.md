# Audio Trim (Visual) for ComfyUI

A **mid-pipe** ComfyUI custom node that trims audio **by dragging handles on a
real waveform timeline or by typing exact numbers** into the node's standard
float fields, wired exactly like the native audio nodes: an `AUDIO` socket in
on the left, an `AUDIO` socket out on the right. Queue once and the node's
waveform timeline shows the incoming audio, so you can place the cut points
visually before (and after) every run.

```
┌────────────┐ AUDIO ┌────────────────────────────────────────────┐ AUDIO ┌────────────┐
│ Load Audio ├──────►│ Audio Trim (Visual)                        ├──────►│ Save Audio │
└────────────┘       │  ▸ start (s) [0 = file end]        1.234   │       └────────────┘
 (or Generate        │  ▸ end (s)                         3.500   │        (or Preview
  Audio, TTS, ...)   │  ▸ gain (dB)                       0.0     │         Audio, ...)
                     │  ▸ snap to zero crossing           true    │
                     │ ┌──────────────────────────────────────────┐│
                     │ │ 0.5      1.0      1.234   2.0        3.0 ││ ← ruler (drag = pan)
                     │ │ ▁▂▅█▆▂▁▂▄█▇▅▃▁░▁▃▅▇█▇▅▃▂▁▁▂▃▄▅▆▇         ││ ← waveform (after queue)
                     │ │       ┃(dim) ▐start ▐end ▐(dim)          ││ ← drag the two bars
                     │ └──────────────────────────────────────────┘│
                     │ [▶ Trim] [▶ Full] [■ Stop] [Loop] [⤢ Fit]   │
                     │ Trim: 2.266s (2266 ms) | Total: 3.500s      │
                     └────────────────────────────────────────────┘
```

## Install

**Option A — zip:** copy `ComfyUI-AudioTrimVisual/` (or unzip
`ComfyUI-AudioTrimVisual.zip`) into `ComfyUI/custom_nodes/` and restart
ComfyUI, then refresh the browser tab.

**Option B — git:**

```bash
cd ComfyUI/custom_nodes
git clone <this repository> ComfyUI-AudioTrimVisual
```

No extra Python packages are needed — the node uses `torch` (ships with
ComfyUI) and `soundfile` (for the temp waveform mirror).

## The node: "Audio Trim (Visual)"

Search for `AudioTrimVisual` (category **audio**). Connect any node that
produces `AUDIO` — Load Audio, Generate Audio, TTS nodes, ... — to the left
socket, and feed the right `AUDIO` output into Save Audio, Preview Audio, or
anything else that accepts `AUDIO`. The node is an **output node**: even with
nothing connected downstream it still runs (so the waveform stays available),
and it reports the exact duration, cut points and clipping warning in its
result area. Writing the trimmed result to disk is the job of the Save Audio
node you connect — it owns the filename prefix and the output format.

### Sockets

| Socket | Direction | Meaning |
| --- | --- | --- |
| `audio` | input (left) | Core `AUDIO` dict (`waveform` tensor `[batch, channels, frames]` + `sample_rate`), identical to what Load Audio outputs. |
| `AUDIO` | output (right) | The same audio, trimmed to the cut range, gain applied, batch preserved — connect it straight into Save Audio. |

### Widgets

| Widget | Meaning |
| --- | --- |
| `start (s)` | Cut start in seconds, millisecond precision (0.001 steps). |
| `end (s)` | Cut end in seconds. **0 = end of file.** |
| `gain (dB)` | Volume change applied to the trimmed region, −60…+24 dB, default 0. Samples that exceed full scale are hard-clipped (the UI warns you when that happens). |
| `snap to zero crossing` | When on, every cut point (drag *and* typed values) is moved to the quietest sample (≈ zero crossing) within ±20 ms to avoid clicks. |

### Timeline interactions

- **Queue the node once** — the backend mirrors the untrimmed input to the
  temp folder and the timeline fetches it automatically. The waveform then
  stays editable between runs. A content fingerprint decides whether the
  trim borders persist: the **same audio re-queued keeps the trim**, a
  **new file resets the trim** to the file's start/end.
- **Drag the two colored bars** — move the start (blue) / end (pink) cut
  points; each drag commits to the float widgets when you release, and the
  bar can be moved back and forth freely in either direction.
- **Type into the standard `start (s)` / `end (s)` float widgets** — one field
  each, 3 decimals (millisecond precision), exactly like other ComfyUI number
  fields. Typed values go through the same snapping as drags.
- **Mouse wheel** — zoom around the cursor; **drag the top ruler** (or
  shift-drag / middle-drag anywhere) — pan; **double-click** — fit the whole file.
- **Playback:** `▶ Trim` plays only the selection, `▶ Full` plays the whole
  file, `■ Stop` stops, `Loop` repeats the selection (restarts playback with
  loop applied), `⤢ Fit` resets the zoom. A moving playhead shows the position,
  and the view follows it.
- **Trim / Total readout** shows the duration in seconds (3 decimals) and
  integer milliseconds, plus sample rate and channel count.

## How precise is it?

- All values are quantized to **1 ms** and every number the UI displays is
  computed **after** zero-snap, so what you see is exactly what is trimmed.
- The timeline fetches the exact input waveform the backend received (full
  float WAV mirror in the temp folder), and both sides use the identical snap
  algorithm (minimum |sample| within ±20 ms, ties → closest to the requested
  point) — preview and output agree to the sample.
- `end (s) = 0` means "end of file" in both the UI and Python; the first load
  of a file fills the end field with the file duration for convenience.
- Batch inputs (e.g. from batched generation) are trimmed with the same cut
  points; snap points are taken from the first batch item.

## Output

Two things happen on every run:

1. **AUDIO socket out** — the trimmed waveform `[batch, channels, frames]` +
   sample rate, ready for Save Audio / Preview Audio / any AUDIO input.
   Connect Save Audio to write the result to disk: it owns the filename
   prefix and the output format, so nothing is configured twice here.
2. **UI mirror** — the untrimmed input is written to the temp folder as a
   float WAV so the timeline can display it. That mirror is internal only;
   the node itself writes nothing into the output folder.

The result text prints `trimmed X.XXXs (N ms) | cut A.BBBs to C.CCCs of
T.TTTs @ rate | snap | gain`, plus a clipping warning when the gain pushed
samples past full scale.

## Troubleshooting

- **Waveform empty:** it appears only after the node has run at least once
  (there is no file combo anymore — the audio comes through the socket).
  Connect a source and queue. Temp files are cleaned by ComfyUI eventually;
  queue again to refresh. Before the first run the timeline is a bare
  placeholder — no ruler, no handles (versions before 1.1.2 drew two stacked,
  undraggable bars at t=0 there). Once audio is loaded the selection starts
  as the full file: start handle at 0 s, end handle at the file end.
- **Handles grab next to the bar, worse the further right you go (or a bar
  can't be dragged back):** versions before 1.2.2 read the pointer in screen
  pixels while the timeline draws in the widget's own layout pixels — the
  graph zoom (Ctrl+wheel) scaled DOM widgets, so the clickable area drifted
  away from the visible bar proportionally with x. Fixed in 1.2.2; also fixed
  there: ruler-pan speed under graph zoom.
- **TypeError `Cannot read properties of undefined (reading 'options')` when
  typing into the number widgets:** you are running a copy older than 1.1.1 —
  its widget-callback wrapper invoked the core's `onFloatValueChange` without
  the widget bound as `this` (the newer Vue frontend reads `this.options`
  there). Replace the installed folder with this version and hard-refresh
  (Ctrl+F5).
- **Old saved workflow shows a missing `audio` input:** versions before 1.1.0
  had a file-combo widget instead of the socket. Delete and re-add the node,
  then rewire (Load Audio → this node → Save Audio).
- **Crash on node creation** (`Cannot read properties of undefined (reading
  'element')` in an `AUDIOUPLOAD` frame): you are running a version older
  than 1.0.1, which used the `audio_upload: True` combo flag (the frontend's
  AUDIOUPLOAD factory crashes on some frontend versions for custom nodes).
  Replace the installed folder with this version and hard-refresh (Ctrl+F5).
- **"Could not load audio" in the timeline:** the temp mirror could not be
  fetched (usually right after a ComfyUI restart cleared the temp folder) —
  queue the node again.
- **Node missing from the menu:** check the ComfyUI console for import errors
  from this folder; run `python tests/test_all.py` inside the package to
  verify the Python side in isolation.

## Tests

```bash
python tests/test_all.py
```

30 checks: trim-index math, zero-snap behaviour and idempotence, dB gain and
clipping, duration formatting, INPUT_TYPES socket shape, plus end-to-end
`run()` tests through the AUDIO socket round-trip (trim/eof/snap/gain/batch,
untrimmed temp mirror, no stray output files, best-effort mirror) against a
stubbed `folder_paths` and a minimal torch stand-in.

## Files

```
ComfyUI-AudioTrimVisual/
├── __init__.py                  # node registration + WEB_DIRECTORY
├── nodes.py                     # AudioTrimVisual backend (socket in -> trim/gain -> socket out + save)
├── audio_utils.py               # pure trimming math (shared logic, tested)
├── web/js/audio_trim.js         # waveform UI, drag/snap/zoom, playback, executed-event loader
├── tests/test_all.py            # standalone test suite
├── example_workflows/
│   └── audio_trim_basic.json    # Load Audio -> Audio Trim (Visual) -> Save Audio
├── pyproject.toml / requirements.txt / LICENSE
└── README.md
```
