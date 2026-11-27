/**
 * Audio Trim (Visual) — frontend for the ComfyUI node "AudioTrimVisual".
 *
 * Mid-pipe node: the AUDIO waveform arrives through the left socket (Load
 * Audio, Generate Audio, TTS, ...) and the trimmed AUDIO leaves through the
 * right socket. Waveform timeline with:
 *  - full-input waveform loaded automatically after every queue (the backend
 *    mirrors the untrimmed input to the temp folder and the UI fetches it);
 *    a re-queued SAME file keeps the trim borders, a NEW file resets them
 *  - draggable start/end handles (trim by dragging); exact values go through
 *    the node's standard start/end float fields (3-decimal floats)
 *  - zero-crossing snap (mirrors the Python algorithm, argmin |sample| ±20 ms)
 *  - wheel zoom, ruler drag to pan, double-click to fit
 *  - playback: Play Trim / Play Full / Stop / Loop with a moving playhead;
 *    click/drag the yellow line inside the trim area to set the play start
 *  - trimmed + total duration shown in seconds and milliseconds
 *
 * The visual block is a single DOM widget appended AFTER the node's native
 * widgets (never reordered), so widget serialization order stays stable.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SNAP_WINDOW_S = 0.02; // same as audio_utils.ZERO_SNAP_WINDOW_S
const RULER_H = 20;
const CANVAS_H = 170;
const HANDLE_W = 8;
const HANDLE_HIT_PX = 9;
const MIN_SPAN_S = 0.02;

const COL = {
    bg: "#14171c",
    rulerBg: "#0f1216",
    rulerTick: "#5d6c80",
    rulerText: "#9fb0c7",
    wave: "rgba(90,162,232,0.85)",
    waveCenter: "rgba(90,162,232,0.35)",
    dim: "rgba(5,7,10,0.55)",
    start: "#4cc9f0",
    end: "#f72585",
    playhead: "#ffd60a",
    text: "#c7d3e0",
};

let styleInjected = false;
function injectStyle() {
    if (styleInjected) return;
    styleInjected = true;
    const st = document.createElement("style");
    st.id = "zat-style";
    st.textContent = `
.zat-root { width: 100%; font-family: Arial, sans-serif; color: ${COL.text};
    padding: 2px 4px 6px 4px; box-sizing: border-box; user-select: none; }
.zat-stage { position: relative; width: 100%; height: ${CANVAS_H}px;
    border-radius: 6px; overflow: hidden; background: ${COL.bg};
    border: 1px solid #2a313c; }
.zat-stage canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
.zat-over { touch-action: none; }
.zat-status { position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; pointer-events: none; font-size: 12px; color: #aab6c5;
    text-shadow: 0 1px 2px rgba(0,0,0,0.8); padding: 0 8px; text-align: center; }
.zat-bar { display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; }
.zat-btn { flex: 1 1 auto; min-width: 58px; padding: 4px 8px; font-size: 12px;
    background: #232a34; color: ${COL.text}; border: 1px solid #38414e;
    border-radius: 5px; cursor: pointer; }
.zat-btn:hover:not(:disabled) { background: #2c3542; }
.zat-btn:disabled { opacity: 0.45; cursor: default; }
.zat-btn.active { background: #2b4a63; border-color: ${COL.start}; color: #eaf4fb; }
.zat-info { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 5px;
    font-size: 11px; color: #8fa3bf; }
.zat-info .zat-warn { color: #ffb347; }
.zat-hint { margin-top: 4px; font-size: 10px; color: #5d6c80; line-height: 1.4; }
`;
    document.head.appendChild(st);
}

// ---------------------------------------------------------------- helpers

function getAudioCtx() {
    if (!getAudioCtx._ctx) {
        getAudioCtx._ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return getAudioCtx._ctx;
}

function num(v) {
    const f = parseFloat(v);
    return Number.isFinite(f) ? f : 0.0;
}
function q(v) { // quantize to milliseconds
    return Math.round(v * 1000) / 1000;
}
function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
}

function fmtClock(t) {
    const ms = Math.max(0, Math.round(t * 1000));
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    return `${m}:${String(s % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
}

// ---------------------------------------------------------------- node UI

function setupNode(node) {
    const getW = (name) => node.widgets.find((w) => w.name === name);

    // note: "audio" is a socket input (no widget); the waveform to display
    // arrives via the executed event instead of a file combo
    const startW = getW("start_seconds");
    const endW = getW("end_seconds");
    const gainW = getW("gain_db");
    const snapW = getW("snap_to_zero_crossing");
    if (!startW || !endW || !gainW || !snapW) return;

    // friendlier labels (label only affects drawing, not serialization)
    startW.label = "start (s)";
    endW.label = "end (s)  [0 = file end]";
    gainW.label = "gain (dB)";
    snapW.label = "snap to zero crossing";

    const st = {
        buffer: null,
        mono: null,
        duration: 0,
        peak: 0,
        rate: 44100,
        channels: 1,
        viewStart: 0,
        viewSpan: 1,
        selStart: 0,
        selEnd: 0,
        cols: null,
        colsKey: "",
        loadToken: 0,
        srcHash: null,
        lastStart: NaN,
        lastEnd: NaN,
        lastGain: NaN,
        loop: false,
        playing: false,
        playSrc: null,
        playRange: [0, 0],
        playLoop: false,
        playStartedAt: 0,
        playFrom: 0,
        playhead: null,
        cue: null,
        drag: null,
        dead: false,
    };

    // ------------------------------------------------------------ DOM block
    const root = document.createElement("div");
    root.className = "zat-root";
    root.innerHTML = `
<div class="zat-stage">
  <canvas class="zat-base"></canvas>
  <canvas class="zat-over"></canvas>
  <div class="zat-status"></div>
</div>
<div class="zat-bar">
  <button class="zat-btn" data-act="trim"  title="Play the trimmed region">▶ Trim</button>
  <button class="zat-btn" data-act="full"  title="Play the whole file">▶ Full</button>
  <button class="zat-btn" data-act="stop"  title="Stop playback">■ Stop</button>
  <button class="zat-btn zat-loop" data-act="loop" title="Loop the trimmed region during playback">Loop: Off</button>
  <button class="zat-btn" data-act="fit"   title="Zoom so the whole file is visible">⤢ Fit</button>
</div>
<div class="zat-info">
  <span class="zat-dur"></span>
  <span class="zat-meta"></span>
  <span class="zat-warn"></span>
</div>
<div class="zat-hint">
  waveform appears after the node runs &nbsp;·&nbsp; wheel = zoom
  &nbsp;·&nbsp; drag top ruler = pan &nbsp;·&nbsp; drag the two edge bars = trim
  &nbsp;·&nbsp; click / drag the yellow line = play start &nbsp;·&nbsp;
  double-click = fit &nbsp;·&nbsp; typed values follow the
  "snap to zero crossing" toggle
</div>`;
    const base = root.querySelector(".zat-base");
    const over = root.querySelector(".zat-over");
    const statusEl = root.querySelector(".zat-status");
    const durEl = root.querySelector(".zat-dur");
    const metaEl = root.querySelector(".zat-meta");
    const warnEl = root.querySelector(".zat-warn");
    const loopBtn = root.querySelector('[data-act="loop"]');

    const vis = node.addDOMWidget("trim_visualizer", "trim_ui", root, {});
    // persist the last input fingerprint in the workflow JSON so "same file,
    // same trim" survives page reloads and workflow switches
    vis.serializeValue = () => st.srcHash || "";
    vis.value = "";

    // ---------------------------------------------------------- view / math
    const cssW = () => Math.max(60, root.querySelector(".zat-stage").clientWidth);
    const pxPerSec = () => cssW() / Math.max(MIN_SPAN_S, st.viewSpan);
    const xToT = (x) => st.viewStart + x / pxPerSec();
    const tToX = (t) => (t - st.viewStart) * pxPerSec();

    function clampView(vs, span) {
        span = clamp(span, MIN_SPAN_S, Math.max(MIN_SPAN_S, st.duration));
        vs = clamp(vs, 0, Math.max(0, st.duration - span));
        return [vs, span];
    }

    function setView(vs, span) {
        [st.viewStart, st.viewSpan] = clampView(vs, span);
        drawBase();
    }

    function fitView() {
        setView(0, Math.max(MIN_SPAN_S, st.duration));
    }

    // Snap to the quietest (|sample| minimal) point within ±20 ms.
    // Ties resolve to the candidate closest to the requested index (then to
    // the lowest index) — identical to the Python side, so both agree.
    function snapTime(t) {
        if (!st.mono || !st.mono.length) return t;
        const n = st.mono.length;
        const rate = st.rate;
        const idx = clamp(Math.round(t * rate), 0, n - 1);
        const w = Math.max(1, Math.round(SNAP_WINDOW_S * rate));
        const lo = Math.max(0, idx - w);
        const hi = Math.min(n - 1, idx + w);
        let best = idx;
        let bestVal = Infinity;
        let bestDist = Infinity;
        for (let i = lo; i <= hi; i++) {
            const v = Math.abs(st.mono[i]);
            const d = Math.abs(i - idx);
            if (v < bestVal || (v === bestVal && d < bestDist)) {
                bestVal = v;
                bestDist = d;
                best = i;
            }
        }
        return best / rate;
    }

    function snapOn() {
        return snapW.value === true || snapW.value === 1 || snapW.value === "true";
    }

    function applySnapPair(a, b) {
        if (!snapOn()) return [a, b];
        let s = snapTime(a);
        let e = snapTime(b);
        if (e <= s) return [a, b]; // snap collision on tiny selection: keep raw
        return [s, e];
    }

    // Re-derive the selection from the (ms-quantized) widget values, snap it,
    // write the snapped values back to the widgets and refresh the UI.
    function refreshSelection() {
        const sRaw = q(clamp(num(startW.value), 0, 1e9));
        const eRaw = num(endW.value) <= 0 ? 0 : q(clamp(num(endW.value), 0, 1e9));
        if (!st.buffer) {
            // no audio yet: keep the widget values exactly as preset (0 for
            // end still means "file end") instead of flattening them —
            // presets typed before the first run must survive to that run
            st.selStart = sRaw;
            st.selEnd = eRaw;
            clampCue();
            updateInfo();
            drawBase();
            st.lastStart = sRaw;
            st.lastEnd = eRaw;
            return;
        }
        // clamp the start into the file: values saved for a longer take must
        // not push the handles past the end of a shorter one
        let a = Math.min(sRaw, Math.max(0, st.duration - 0.001));
        let b = eRaw <= 0 ? st.duration : Math.min(eRaw, st.duration);
        if (b <= a) b = Math.min(st.duration, a + 0.001);
        [a, b] = applySnapPair(a, b);
        st.selStart = a;
        st.selEnd = Math.min(st.duration, Math.max(a + 0.001, b));
        clampCue(); // play-start marker follows the trim area
        // write snapped values back so the widgets and the saved file agree
        startW.value = q(a);
        endW.value = q(st.selEnd);
        updateInfo();
        drawBase();
        st.lastStart = num(startW.value);
        st.lastEnd = num(endW.value);
    }

    // --------------------------------------------- play-start marker (cue)
    function clampCue() {
        // the marker lives inside the current trim area
        if (st.cue != null) st.cue = clamp(st.cue, st.selStart, st.selEnd);
    }

    function cueFromX(x) {
        let t = q(clamp(xToT(x), 0, st.duration));
        if (snapOn()) t = q(snapTime(t)); // same snap discipline as the cuts
        return clamp(t, st.selStart, st.selEnd);
    }

    function seekTo(t) {
        st.cue = t;
        drawOverlay();
        if (st.playing) {
            // live-seek, throttled so a fast drag does not machine-gun the
            // audio source; endDrag plays the exact release position
            const now = performance.now();
            if (!seekTo._last || now - seekTo._last > 80) {
                seekTo._last = now;
                play(st.playRange, t);
            }
        }
    }

    function updateInfo() {
        const len = st.selEnd - st.selStart;
        durEl.textContent = st.buffer
            ? `Trim: ${len.toFixed(3)} s · ${Math.round(len * 1000)} ms   |   ` +
              `Total: ${st.duration.toFixed(3)} s · ${Math.round(st.duration * 1000)} ms`
            : "Trim: — | Total: —";
        metaEl.textContent = st.buffer
            ? `${st.rate} Hz · ${st.channels === 1 ? "mono" : st.channels + " ch"} · ${fmtClock(st.duration)}`
            : "";
        const gain = num(gainW.value);
        if (st.buffer && gain !== 0) {
            const lin = Math.pow(10, gain / 20);
            warnEl.textContent = lin * st.peak > 1.0
                ? "⚠ gain clips the output (preview is clamped the same way)"
                : "";
        } else {
            warnEl.textContent = "";
        }
        st.lastGain = gain;
        updateButtons();
    }

    function updateButtons() {
        const canPlay = !!st.buffer && st.selEnd - st.selStart > 0.0005;
        root.querySelector('[data-act="trim"]').disabled = !canPlay;
        root.querySelector('[data-act="full"]').disabled = !st.buffer;
        root.querySelector('[data-act="stop"]').disabled = !st.playing;
        loopBtn.textContent = st.loop ? "Loop: On" : "Loop: Off";
        loopBtn.classList.toggle("active", st.loop);
    }

    function setStatus(msg) {
        statusEl.textContent = msg || "";
    }

    // ------------------------------------------------------------- loading
    // The backend mirrors the untrimmed input to the temp folder on every run
    // and reports it via ui.zat_full in the executed event; fetch and decode.
    async function loadAudioFromEntry(entry) {
        st.loadToken += 1;
        const token = st.loadToken;
        const { filename, subfolder, type } = entry;
        const url = api.apiURL(
            `/view?filename=${encodeURIComponent(filename)}` +
            `&subfolder=${encodeURIComponent(subfolder || "")}` +
            `&type=${encodeURIComponent(type || "temp")}` +
            `&freshness=${Date.now()}`
        );
        setStatus("Loading " + filename + " …");
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const arr = await res.arrayBuffer();
            const buf = await getAudioCtx().decodeAudioData(arr);
            if (token !== st.loadToken || st.dead) return; // stale load
            st.buffer = buf;
            st.rate = buf.sampleRate;
            st.channels = buf.numberOfChannels;
            st.duration = buf.duration;
            // mono mixdown for waveform + snapping
            const L = buf.length;
            const mono = new Float32Array(L);
            for (let c = 0; c < buf.numberOfChannels; c++) {
                const d = buf.getChannelData(c);
                for (let i = 0; i < L; i++) mono[i] += d[i];
            }
            if (buf.numberOfChannels > 1) {
                for (let i = 0; i < L; i++) mono[i] /= buf.numberOfChannels;
            }
            st.mono = mono;
            let peak = 0;
            for (let i = 0; i < L; i++) {
                const v = Math.abs(mono[i]);
                if (v > peak) peak = v;
            }
            st.peak = peak;
            st.cols = null;
            st.colsKey = "";
            // same-file detection (backend content hash): re-queued SAME
            // audio keeps the trim borders, a NEW sound resets them to the
            // file's start/end; no hash in the payload -> keep borders
            if (entry.hash && entry.hash !== st.srcHash) {
                startW.value = 0;
                endW.value = 0;
                st.cue = null; // new file: play-start marker resets too
            }
            if (entry.hash) st.srcHash = entry.hash;
            setStatus("");
            // first load: end = file end (widget 0 means "end of file")
            if (!(num(endW.value) > 0)) endW.value = q(st.duration);
            fitView();
            refreshSelection();
        } catch (err) {
            if (token !== st.loadToken) return;
            st.buffer = null;
            st.mono = null;
            st.cue = null;
            setStatus(`⚠ Could not load audio: ${err.message || err}`);
            drawBase();
            updateInfo();
        }
    }

    // ------------------------------------------------- execution results
    // The trimmed AUDIO goes through the output socket; the ui payload tells
    // the timeline where to fetch the FULL input for editing the next cut.
    const onExecuted = (e) => {
        if (st.dead) return;
        const d = e.detail || {};
        if (String(d.node ?? "") !== String(node.id)) return;
        const full = d.output?.zat_full?.[0];
        if (full && full.filename) loadAudioFromEntry(full);
    };
    api.addEventListener("executed", onExecuted);

    // -------------------------------------------------------------- drawing
    function dpr() {
        return window.devicePixelRatio || 1;
    }

    function resizeCanvases() {
        const w = cssW();
        const h = CANVAS_H;
        for (const c of [base, over]) {
            c.width = Math.max(1, Math.round(w * dpr()));
            c.height = Math.max(1, Math.round(h * dpr()));
        }
        st.cols = null;
        st.colsKey = "";
        drawBase();
        drawOverlay();
    }

    function ensurePeaks() {
        const w = cssW();
        const key = `${st.loadToken}|${st.viewStart.toFixed(6)}|${st.viewSpan.toFixed(6)}|${w}`;
        if (st.colsKey === key && st.cols) return;
        const W = Math.max(1, Math.floor(w));
        const cols = new Float32Array(W * 2);
        if (st.mono && st.mono.length) {
            const n = st.mono.length;
            const a = Math.max(0, Math.floor(st.viewStart * st.rate));
            const b = Math.min(n, Math.ceil((st.viewStart + st.viewSpan) * st.rate));
            const span = Math.max(1, b - a);
            for (let x = 0; x < W; x++) {
                const s0 = a + Math.floor((span * x) / W);
                const s1 = Math.max(s0 + 1, a + Math.floor((span * (x + 1)) / W));
                let mn = 1e9;
                let mx = -1e9;
                for (let i = s0; i < s1 && i < n; i++) {
                    const v = st.mono[i];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                if (mn > mx) { mn = 0; mx = 0; }
                cols[x * 2] = mn;
                cols[x * 2 + 1] = mx;
            }
        }
        st.cols = cols;
        st.colsKey = key;
    }

    function niceStep(pxPerSecVal) {
        const steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
            1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
        for (const s of steps) {
            if (s * pxPerSecVal >= 64) return s;
        }
        return steps[steps.length - 1];
    }

    function tickLabel(t, step) {
        if (step >= 1) {
            const dec = step >= 10 ? 0 : 1;
            return t.toFixed(dec);
        }
        if (step >= 0.1) return t.toFixed(1);
        if (step >= 0.01) return t.toFixed(2);
        return t.toFixed(3);
    }

    function drawBase() {
        const ctx = base.getContext("2d");
        const d = dpr();
        const w = base.width / d;
        const h = base.height / d;
        ctx.save();
        ctx.setTransform(d, 0, 0, d, 0, 0);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = COL.bg;
        ctx.fillRect(0, 0, w, h);

        ensurePeaks();
        const pps = pxPerSec();

        // ruler (empty state: nothing to measure yet)
        if (st.buffer) {
            ctx.fillStyle = COL.rulerBg;
            ctx.fillRect(0, 0, w, RULER_H);
            const step = niceStep(pps);
            const t0 = Math.ceil(st.viewStart / step) * step;
            ctx.strokeStyle = COL.rulerTick;
            ctx.fillStyle = COL.rulerText;
            ctx.lineWidth = 1;
            ctx.font = "10px Arial, sans-serif";
            ctx.textBaseline = "top";
            ctx.beginPath();
            for (let t = t0; t <= st.viewStart + st.viewSpan + 1e-9; t += step) {
                const x = Math.round(tToX(t)) + 0.5;
                if (x < -40 || x > w + 40) continue;
                ctx.moveTo(x, RULER_H - 7);
                ctx.lineTo(x, RULER_H);
                ctx.fillText(tickLabel(t, step), x + 3, 3);
            }
            ctx.stroke();
        }

        // waveform
        const top = RULER_H;
        const wh = h - top;
        const mid = top + wh / 2;
        if (st.cols) {
            const W = st.cols.length / 2;
            const scaleY = (wh / 2) * 0.92;
            ctx.fillStyle = COL.wave;
            for (let x = 0; x < W && x < w; x++) {
                const mn = st.cols[x * 2];
                const mx = st.cols[x * 2 + 1];
                const y0 = mid - mx * scaleY;
                const y1 = mid - mn * scaleY;
                const hh = Math.max(1, y1 - y0);
                ctx.fillRect(x, y0, 1, hh);
            }
            ctx.fillStyle = COL.waveCenter;
            ctx.fillRect(0, mid, w, 1);
        } else {
            ctx.fillStyle = "#3a424e";
            ctx.font = "12px Arial, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(st.mono ? "" : "no audio loaded — queue the node once", w / 2, mid - 6);
            ctx.textAlign = "left";
        }

        // dim outside the selection + drag handles. Only with audio loaded:
        // on the empty placeholder there is no selection yet, and two
        // stacked, undraggable bars at t=0 just look broken.
        if (st.buffer) {
            const xs = tToX(st.selStart);
            const xe = tToX(st.selEnd);
            ctx.fillStyle = COL.dim;
            if (xs > 0) ctx.fillRect(0, RULER_H, Math.min(xs, w), wh);
            if (xe < w) ctx.fillRect(Math.max(0, xe), RULER_H, w - Math.max(0, xe), wh);
            drawHandle(ctx, xs, top, wh, COL.start);
            drawHandle(ctx, xe, top, wh, COL.end);
        }
        ctx.restore();
    }

    function drawHandle(ctx, x, top, wh, color) {
        ctx.save();
        ctx.fillStyle = color;
        const r = 3;
        const x0 = x - HANDLE_W / 2;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x0, top + 2, HANDLE_W, wh - 4, r);
        } else {
            ctx.rect(x0, top + 2, HANDLE_W, wh - 4);
        }
        ctx.fill();
        // notch triangle at the ruler edge
        ctx.beginPath();
        ctx.moveTo(x0 - 3, top + 2);
        ctx.lineTo(x0 + HANDLE_W + 3, top + 2);
        ctx.lineTo(x0 + HANDLE_W / 2, top + 10);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    function drawOverlay() {
        const ctx = over.getContext("2d");
        const d = dpr();
        const w = over.width / d;
        const h = over.height / d;
        ctx.save();
        ctx.setTransform(d, 0, 0, d, 0, 0);
        ctx.clearRect(0, 0, w, h);
        // playing: the moving playhead; idle: the draggable play-start marker
        const marker = st.playing ? st.playhead : st.cue;
        if (marker != null) {
            const x = tToX(marker);
            if (x >= -2 && x <= w + 2) {
                ctx.strokeStyle = COL.playhead;
                ctx.fillStyle = COL.playhead;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(x - 5, 0);
                ctx.lineTo(x + 5, 0);
                ctx.lineTo(x, 8);
                ctx.closePath();
                ctx.fill();
            }
        }
        ctx.restore();
    }

    // ------------------------------------------------------------- playback
    function stopPlayback() {
        if (st.playSrc) {
            try { st.playSrc.onended = null; st.playSrc.stop(); } catch (e) { /* noop */ }
            st.playSrc = null;
        }
        st.playing = false;
        st.playhead = null;
        drawOverlay();
        updateButtons();
    }

    function play(range, startAt) {
        if (!st.buffer || range[1] - range[0] <= 0.0005) return;
        stopPlayback();
        const ctx = getAudioCtx();
        if (ctx.state === "suspended") ctx.resume();
        const src = ctx.createBufferSource();
        src.buffer = st.buffer;
        const g = ctx.createGain();
        g.gain.value = Math.pow(10, num(gainW.value) / 20);
        src.connect(g);
        g.connect(ctx.destination);
        const a = range[0];
        const b = range[1];
        // start at the cue when one is set (clamped into the range)
        const from = startAt == null ? a : clamp(num(startAt), a, b);
        if (st.loop) {
            src.loop = true;
            src.loopStart = a;
            src.loopEnd = b;
            src.start(0, from);
        } else {
            src.start(0, from, b - from);
        }
        src.onended = () => {
            if (st.playSrc === src) stopPlayback();
        };
        st.playSrc = src;
        st.playRange = [a, b];
        st.playLoop = !!src.loop;
        st.playFrom = from;
        st.playStartedAt = ctx.currentTime;
        st.playing = true;
        updateButtons();
        rafPlayhead();
    }

    function rafPlayhead() {
        if (!st.playing) return;
        const ctx = getAudioCtx();
        const el = ctx.currentTime - st.playStartedAt;
        const a = st.playRange[0];
        const b = st.playRange[1];
        const len = Math.max(0.001, b - a);
        const off = st.playFrom - a; // playback may have started at the cue
        st.playhead = st.playLoop ? (a + ((off + el) % len))
            : Math.min(b, st.playFrom + el);
        // auto-follow the playhead when it leaves the current view
        const x = tToX(st.playhead);
        if (x < 0 || x > cssW()) {
            const span = st.viewSpan;
            setView(st.playhead - span * 0.1, span);
        }
        drawOverlay();
        requestAnimationFrame(rafPlayhead);
    }

    // ---------------------------------------------------------- interaction
    function pos(e) {
        const rect = over.getBoundingClientRect();
        // DOM widgets sit inside the graph canvas' CSS transform, so the
        // element's visual rect is layout px * graphZoom. Map the pointer
        // (screen px) back into layout px — the space the drawing and all
        // tToX()/xToT() math use. Without this the hit zones drift left/right
        // proportional to x as soon as the graph is zoomed (exact only at 0).
        const k = (over.clientWidth || cssW()) / (rect.width || 1);
        return { x: (e.clientX - rect.left) * k, y: (e.clientY - rect.top) * k };
    }

    function hitHandle(x) {
        const xs = tToX(st.selStart);
        const xe = tToX(st.selEnd);
        if (Math.abs(x - xs) <= HANDLE_HIT_PX) return "start";
        if (Math.abs(x - xe) <= HANDLE_HIT_PX) return "end";
        return null;
    }

    over.addEventListener("pointerdown", (e) => {
        if (!st.buffer) return;
        st.drag = null; // safety: a previous drag that never saw pointerup
        const { x, y } = pos(e);
        const h = hitHandle(x);
        if (h) {
            st.drag = { kind: h };
        } else if (y <= RULER_H || e.button === 1 || e.shiftKey) {
            st.drag = { kind: "pan", startX: pos(e).x, vs: st.viewStart };
            over.style.cursor = "grabbing";
        } else {
            // body click/drag: move the yellow play-start line within the
            // trim area (and jump a running playback to the new position)
            st.drag = { kind: "seek" };
            seekTo(cueFromX(x));
        }
        over.setPointerCapture(e.pointerId);
        e.preventDefault();
    });

    over.addEventListener("pointermove", (e) => {
        const { x } = pos(e);
        if (!st.drag) {
            if (st.buffer) {
                const h = hitHandle(x);
                over.style.cursor = h ? "ew-resize"
                    : (pos(e).y <= RULER_H ? "grab" : "pointer");
            }
            return;
        }
        if (st.drag.kind === "pan") {
            const dx = pos(e).x - st.drag.startX;
            setView(st.drag.vs - dx / pxPerSec(), st.viewSpan);
            return;
        }
        if (st.drag.kind === "seek") {
            seekTo(cueFromX(x));
            return;
        }
        // dragging a handle: quantize to ms, snap, clamp against the other edge
        let t = q(clamp(xToT(x), 0, st.duration));
        if (snapOn()) t = q(snapTime(t));
        if (st.drag.kind === "start") {
            t = Math.min(t, q(st.selEnd - 0.001));
            st.selStart = Math.max(0, t);
        } else {
            t = Math.max(t, q(st.selStart + 0.001));
            st.selEnd = Math.min(st.duration, t);
        }
        // NO widget writes here: per-move writes run through the frontend's
        // widget setters (re-normalize / re-render) which re-layouts the node
        // mid-drag and fights the live handle — commit once at drag end
        updateInfo();
        drawBase();
    });

    const endDrag = (e) => {
        if (!st.drag) return;
        const kind = st.drag.kind;
        st.drag = null;
        try { over.releasePointerCapture(e.pointerId); } catch (err) { /* noop */ }
        over.style.cursor = "default";
        if ((kind === "start" || kind === "end") && st.buffer) {
            // commit the final trim to the standard float widgets once; the
            // write triggers refreshSelection (snap is idempotent, so the
            // values stay exactly what the drag produced)
            startW.value = q(st.selStart);
            endW.value = q(st.selEnd);
        }
        if (kind === "seek" && st.playing) {
            play(st.playRange, st.cue); // land the audio exactly at release
        }
        st.lastStart = num(startW.value);
        st.lastEnd = num(endW.value);
    };
    over.addEventListener("pointerup", endDrag);
    over.addEventListener("pointercancel", endDrag);
    // safety net: if the frontend re-renders and steals the capture mid-drag,
    // finish the drag instead of leaving a stuck handle behind
    over.addEventListener("lostpointercapture", endDrag);

    over.addEventListener("wheel", (e) => {
        if (!st.buffer) return;
        e.preventDefault();
        const { x } = pos(e);
        const tAt = xToT(x);
        const factor = Math.exp(e.deltaY * 0.0012);
        const span = clamp(st.viewSpan * factor, MIN_SPAN_S,
            Math.max(MIN_SPAN_S, st.duration));
        setView(tAt - (tAt - st.viewStart) * (span / st.viewSpan), span);
    }, { passive: false });

    over.addEventListener("dblclick", () => {
        if (st.buffer) fitView();
    });

    // toolbar
    root.querySelector('[data-act="trim"]').addEventListener("click",
        () => play([st.selStart, st.selEnd], st.cue));
    root.querySelector('[data-act="full"]').addEventListener("click",
        () => play([0, st.duration], st.cue));
    root.querySelector('[data-act="stop"]').addEventListener("click", stopPlayback);
    loopBtn.addEventListener("click", () => {
        st.loop = !st.loop;
        // restart trim playback with the new loop mode for instant feedback,
        // keeping the current position when one is already moving
        if (st.playing) play(st.playRange, st.playhead);
        updateButtons();
    });
    root.querySelector('[data-act="fit"]').addEventListener("click", () => {
        if (st.buffer) fitView();
    });

    // ------------------------------------------------- widget sync plumbing
    const wrap = (w, fn) => {
        const orig = w.callback;
        // Bind the widget as `this` and forward EVERY argument: the core's
        // handlers are declared like `function onFloatValueChange(
        // this: INumericWidget, v)` and read `this.options` — calling
        // `orig(v)` bare leaves `this` undefined and crashes newer
        // frontends ("Cannot read properties of undefined (reading
        // 'options')" when typing into number fields).
        w.callback = function (...args) {
            const r = orig ? orig.apply(w, args) : undefined;
            if (!st.dead) fn(args[0]);
            return r;
        };
    };
    wrap(startW, () => { if (!st.drag) refreshSelection(); });
    wrap(endW, () => { if (!st.drag) refreshSelection(); });
    wrap(gainW, () => updateInfo());
    wrap(snapW, () => refreshSelection());

    // safety poll: catches value changes that bypass callbacks (workflow
    // load, right-click widgets, clip start on queue, ...)
    st.pollTimer = setInterval(() => {
        if (st.dead || !node.graph) {
            clearInterval(st.pollTimer);
            return;
        }
        if (num(startW.value) !== st.lastStart || num(endW.value) !== st.lastEnd) {
            if (!st.drag) refreshSelection();
        } else if (num(gainW.value) !== st.lastGain) {
            updateInfo();
        }
    }, 400);

    const ro = new ResizeObserver(() => {
        if (!st.dead) resizeCanvases();
    });
    ro.observe(root);
    st.ro = ro;

    node.onRemoved = () => {
        st.dead = true;
        clearInterval(st.pollTimer);
        if (st.ro) st.ro.disconnect();
        api.removeEventListener("executed", onExecuted);
        stopPlayback();
    };

    // workflow load: widget values were restored from the serialized graph
    node.onConfigure = () => {
        // restore the fingerprint saved with the workflow so a re-queued
        // same file keeps its trim borders even after a page reload
        st.srcHash = String(vis.value ?? "") || null;
        setTimeout(() => {
            if (st.dead) return;
            st.lastStart = NaN; // force refreshSelection via poll logic
            refreshSelection();
        }, 80);
    };

    // --------------------------------------------------------------- boot
    requestAnimationFrame(() => {
        if (st.dead) return;
        resizeCanvases();
        updateInfo();
        setStatus("Connect an audio source (left) and queue the node to see the waveform.");
    });
    // make sure the node is tall enough for the visual block
    node.setSize([Math.max(node.size[0], 380), node.size[1]]);
}

app.registerExtension({
    name: "ZAI.AudioTrimVisual",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "AudioTrimVisual") return;
        injectStyle();
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig ? orig.apply(this, arguments) : undefined;
            try {
                setupNode(this);
            } catch (err) {
                console.error("[AudioTrimVisual] UI setup failed:", err);
            }
            return r;
        };
    },
});
