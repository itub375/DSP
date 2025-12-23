"""
test code Phil / GPT
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.io import wavfile
from scipy.signal import get_window
import time

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    INPUT_FILE: str = r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/output.wav"
    OUT_DIR: str = "output_signals"

    WINDOW_MS: float = 5.0
    HOP_MS: float = 1.0

    MIN_SEGMENT_MS: float = 10.0
    CHANGE_THRESHOLD_PERCENTILE: float = 85.0

    # Change score weights
    W_CENTROID: float = 2.0
    W_RMS: float = 1.0
    W_ROLLOFF: float = 1.5
    W_ZCR: float = 1.5
    W_FLUX: float = 4.0
    W_BW: float = 1.0

    FADE_MS: float = 5.0     # ⭐ extrem wichtig für Qualität
    EXPORT_FMT: str = "wav"

# ============================================================
# AUDIO I/O (WAV ONLY)
# ============================================================

def load_wav(path):
    sr, y = wavfile.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y) + 1e-9)
    return y, sr

def save_wav(y, sr, path):
    y = np.clip(y, -1.0, 1.0)
    wavfile.write(path, sr, (y * 32767).astype(np.int16))

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_features(y, sr, window_ms, hop_ms):
    win = int(window_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    win = win + win % 2

    window = np.hanning(win)
    n = 1 + (len(y) - win) // hop

    freqs = np.fft.rfftfreq(win, 1 / sr)

    centroid = np.zeros(n)
    rms = np.zeros(n)
    rolloff = np.zeros(n)
    zcr = np.zeros(n)
    flux = np.zeros(n)
    bw = np.zeros(n)

    prev_spec = None

    for i in range(n):
        frame = y[i * hop:i * hop + win]
        rms[i] = np.sqrt(np.mean(frame ** 2))
        zcr[i] = np.mean(np.abs(np.diff(np.sign(frame)))) / 2

        spec = np.abs(np.fft.rfft(frame * window))
        power = spec ** 2

        if power.sum() > 0:
            centroid[i] = np.sum(freqs * power) / power.sum()
            bw[i] = np.sqrt(np.sum(((freqs - centroid[i]) ** 2) * power) / power.sum())
            cs = np.cumsum(power)
            rolloff[i] = freqs[np.searchsorted(cs, 0.85 * cs[-1])]

        if prev_spec is not None:
            flux[i] = np.sum((spec - prev_spec) ** 2)
        prev_spec = spec

    times = np.arange(n) * hop / sr

    return dict(
        times=times,
        centroid=centroid,
        rms=rms,
        rolloff=rolloff,
        zcr=zcr,
        flux=flux,
        bandwidth=bw,
        hop=hop
    )

# ============================================================
# CHANGE DETECTION
# ============================================================

def normalize(x):
    lo, hi = np.percentile(x, [5, 95])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

def detect_changes(cfg, f):
    c = normalize(f["centroid"])
    r = normalize(f["rms"])
    ro = normalize(f["rolloff"])
    z = normalize(f["zcr"])
    fl = normalize(f["flux"])
    bw = normalize(f["bandwidth"])

    change = (
        cfg.W_CENTROID * np.abs(np.diff(c, prepend=c[0])) +
        cfg.W_RMS * np.abs(np.diff(r, prepend=r[0])) +
        cfg.W_ROLLOFF * np.abs(np.diff(ro, prepend=ro[0])) +
        cfg.W_ZCR * np.abs(np.diff(z, prepend=z[0])) +
        cfg.W_FLUX * np.abs(np.diff(fl, prepend=fl[0])) +
        cfg.W_BW * np.abs(np.diff(bw, prepend=bw[0]))
    )

    change = np.convolve(change, np.ones(5) / 5, mode="same")
    threshold = np.percentile(change, cfg.CHANGE_THRESHOLD_PERCENTILE)

    return change, threshold

def find_boundaries(change, thresh, times, min_ms):
    min_s = min_ms / 1000
    peaks = [times[i] for i in range(1, len(change)-1)
             if change[i] > thresh and change[i] >= change[i-1] and change[i] >= change[i+1]]

    boundaries = [0.0]
    for p in peaks:
        if p - boundaries[-1] > min_s:
            boundaries.append(p)
    boundaries.append(times[-1])
    return boundaries

# ============================================================
# SEGMENTS & CLUSTERING (SIMPLIFIED, ROBUST)
# ============================================================

def segments_from_boundaries(b):
    return [(b[i], b[i+1]) for i in range(len(b)-1)]

def cluster_segments_simple(y, sr, segments):
    freqs = []
    for s, e in segments:
        seg = y[int(s*sr):int(e*sr)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.argmax(spec)
        freqs.append(f)
    freqs = np.array(freqs)
    return np.argsort(np.argsort(freqs)) % 3   # robust heuristic

# ============================================================
# ⭐ HIGH QUALITY RECONSTRUCTION (OPTION 1)
# ============================================================

def reconstruct_overlap_add(y, sr, segments, labels, fade_ms):
    fade = int(fade_ms * sr / 1000)
    outputs = {}

    for lab in sorted(set(labels)):
        out = np.zeros_like(y)

        for (s, e), l in zip(segments, labels):
            if l != lab:
                continue

            si, ei = int(s * sr), int(e * sr)
            seg = y[si:ei]

            win = np.ones(len(seg))
            if len(seg) > 2 * fade:
                win[:fade] = np.linspace(0, 1, fade)
                win[-fade:] = np.linspace(1, 0, fade)

            out[si:ei] += seg * win

        outputs[lab] = out

    return outputs

# ============================================================
# MAIN
# ============================================================

def main():
    cfg = Config()
    y, sr = load_wav(cfg.INPUT_FILE)

    features = compute_features(y, sr, cfg.WINDOW_MS, cfg.HOP_MS)
    change, thresh = detect_changes(cfg, features)
    boundaries = find_boundaries(change, thresh, features["times"], cfg.MIN_SEGMENT_MS)

    segments = segments_from_boundaries(boundaries)
    labels = cluster_segments_simple(y, sr, segments)

    signals = reconstruct_overlap_add(
        y, sr, segments, labels, cfg.FADE_MS
    )

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    for k, sig in signals.items():
        save_wav(sig, sr, f"{cfg.OUT_DIR}/signal_{chr(65+k)}.wav")

    print("✔ Rekonstruktion abgeschlossen (High Quality)")

if __name__ == "__main__":
    main()
