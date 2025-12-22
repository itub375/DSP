# ---------------------------------------
# Interleaved Audio Segmentation + Labeling (Freq + Amp)
# Author: (für dich angepasst)
# ---------------------------------------
# Requirements:
#   pip install numpy matplotlib pydub
#   + ffmpeg installiert (für mp3 via pydub)
#
# Output:
#   - Plots (waveform + change points + boundaries)
#   - Export pro Label (A, B, C, ...) als MP3 (mask mode: nur Label hörbar, Rest still)
#
# ---------------------------------------

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pydub import AudioSegment


# ---------------------------------------
# CONFIG
# ---------------------------------------

@dataclass
class Config:
    # --- I/O ---
    INPUT_FILE: str = r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/50ms/interleaved_1k_8k_vio_50ms.mp3"
    OUT_DIR: str = "output_segments_v5"

    # --- Framing ---
    FRAME_DURATION_S: float = 0.002          # z.B. 2 ms
    HOP_DURATION_S: Optional[float] = None   # None => hop=frame (kein overlap)
    WINDOW: str = "hann"                    # "hann" oder "hamming" oder "rect"

    # --- Thresholds / Peak Picking ---
    PERCENTILE_FEATURE: float = 80.0        # (Default 80)
    MIN_GAP_S: float = 0.01                 # min Abstand Peaks pro Methode
    CONSOLIDATE_WITHIN_S: float = 0.003      # Peaks in +/- Window -> stärksten behalten
    HYSTERESIS_FRAMES: int = 2              # Wechsel nur, wenn N Frames "über Schwelle"

    # --- Jump Detektor (Boundary jump) ---
    JUMP_PERCENTILE: float = 98.0           # (Default 98)
    JUMP_MIN_GAP_S: float = 0.005

    # --- Fusion ---
    JOINT_MAX_DIFF_S: float = 0.003         # Cluster "close times"
    IGNORE_ONLY_ENERGY_AND_JUMP: bool = False

    # Gewichte in der Fusion (Score-Summe)
    W_CENTROID: float = 1.0
    W_ENERGY: float = 0.7
    W_JUMP: float = 0.6
    W_SHAPE: float = 1.0

    # --- Segmentierung ---
    USE_PRIO1_IN_BOUNDARIES: bool = True
    USE_PRIO2_IN_BOUNDARIES: bool = True
    USE_PRIO3_IN_BOUNDARIES: bool = True

    MIN_SEGMENT_DURATION_S: float = 0.005     # harte Untergrenze
    MERGE_SHORT_SEGMENTS: bool = True

    # Optional: Ground Truth Marker (wenn du echte Blocklänge kennst, z.B. 0.05)
    TRUE_SEGMENT_DURATION_S: Optional[float] = 0.05

    # --- Labeling / Clustering (Freq + Amp) ---
    NUM_SIGNALS: Optional[int] = None        # z.B. 2 oder 3; None => auto (heuristisch)
    MIN_SEGMENTS_PER_CLUSTER: int = 3
    KMEANS_ITERS: int = 60
    KMEANS_SEED: int = 7

    # Skalierung / Distanz
    MERGE_THRESH: float = 1.8                # Merge kleiner Cluster in nächsten

    # --- Export ---
    EXPORT_MODE: str = "mask"                # "mask" oder "concat"
    EXPORT_FORMAT: str = "mp3"               # "mp3" (pydub) oder "wav" (numpy->wav wäre extra)

    # --- Debug / Plots ---
    VERBOSE: bool = True
    PLOT_MAX_SECONDS: Optional[float] = None  # z.B. 3.0 für kurze Plots
    PLOT_MAX_FREQ_HZ: Optional[float] = None  # nur relevant, wenn du Spektrum-Plot ergänzen willst


# ---------------------------------------
# AUDIO I/O
# ---------------------------------------

def load_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    """
    Lädt MP3/WAV über pydub, konvertiert auf mono float32 [-1,1], gibt (y, sr) zurück.
    """
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())

    # sample width bestimmen (pydub liefert int)
    if audio.sample_width == 2:
        y = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        y = samples.astype(np.float32) / 2147483648.0
    else:
        # fallback
        maxv = np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else 1
        y = samples.astype(np.float32) / float(maxv)

    return y, sr


def float_to_audiosegment(y: np.ndarray, sr: int) -> AudioSegment:
    """
    float32 [-1,1] -> int16 -> AudioSegment
    """
    y = np.clip(y, -1.0, 1.0)
    int16 = (y * 32767.0).astype(np.int16)
    raw = int16.tobytes()
    return AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sr,
        channels=1
    )


# ---------------------------------------
# FRAMING
# ---------------------------------------

def get_window(win: str, n: int) -> np.ndarray:
    if win == "hann":
        return np.hanning(n).astype(np.float32)
    if win == "hamming":
        return np.hamming(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)


def frame_signal(y: np.ndarray, sr: int, frame_len: int, hop_len: int) -> np.ndarray:
    """
    Gibt Frames als 2D Array: shape (n_frames, frame_len)
    """
    n = len(y)
    if n < frame_len:
        pad = frame_len - n
        y = np.pad(y, (0, pad), mode="constant")
        n = len(y)

    n_frames = 1 + (n - frame_len) // hop_len
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frames[i] = y[start:start + frame_len]
    return frames


# ---------------------------------------
# ROBUST UTILS
# ---------------------------------------

def robust_scale(X: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust scaling: (X - median) / (MAD*1.4826)
    """
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = (mad * 1.4826) + eps
    Xn = (X - med) / scale
    return Xn, med, scale


def median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    """
    Einfacher Medianfilter (k muss ungerade sein).
    """
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    # sliding window view ohne scipy:
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(xp, k)
        return np.median(w, axis=-1)
    except Exception:
        # fallback langsam
        out = np.zeros_like(x)
        for i in range(len(x)):
            out[i] = np.median(xp[i:i + k])
        return out


def local_maxima_peaks(score: np.ndarray, thr: float) -> np.ndarray:
    """
    Indizes, wo score lokales Maximum und > thr.
    """
    s = score
    peaks = []
    for i in range(1, len(s) - 1):
        if s[i] > thr and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def enforce_min_gap(peaks: np.ndarray, score: np.ndarray, min_gap_frames: int) -> np.ndarray:
    """
    Greedy: behält stärkste Peaks und entfernt welche zu nah dran liegen.
    """
    if len(peaks) == 0:
        return peaks
    # sort nach score desc
    order = peaks[np.argsort(score[peaks])[::-1]]
    kept = []
    for p in order:
        if all(abs(p - k) >= min_gap_frames for k in kept):
            kept.append(p)
    kept = np.array(sorted(kept), dtype=int)
    return kept


def consolidate_within_window(peaks: np.ndarray, score: np.ndarray, win_frames: int) -> np.ndarray:
    """
    Peaks in einem Fenster zusammenfassen: in +/- win_frames nur den stärksten behalten.
    """
    if len(peaks) == 0:
        return peaks
    peaks = np.array(sorted(peaks), dtype=int)

    clusters = []
    cur = [peaks[0]]
    for p in peaks[1:]:
        if abs(p - cur[-1]) <= win_frames:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)

    out = []
    for c in clusters:
        c = np.array(c, dtype=int)
        best = c[np.argmax(score[c])]
        out.append(best)
    return np.array(sorted(out), dtype=int)


def apply_hysteresis(score: np.ndarray, thr: float, n_frames: int) -> np.ndarray:
    """
    Markiert Indizes, wo score für >= n_frames am Stück über thr ist (grob).
    Gibt bool-Array gleicher Länge zurück.
    """
    over = score > thr
    out = np.zeros_like(over, dtype=bool)
    run = 0
    for i, v in enumerate(over):
        run = run + 1 if v else 0
        if run >= n_frames:
            out[i] = True
    return out


# ---------------------------------------
# FEATURES (FFT basiert, ohne librosa)
# ---------------------------------------

def compute_frame_features(frames: np.ndarray, sr: int, window: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Features pro Frame:
      - energy, rms_db
      - centroid_hz
      - bandwidth_hz
      - rolloff_hz (0.85)
      - flatness
      - zcr
      - band_energies (4 Bänder)
      - spec_norm (für shape detector)
    """
    eps = 1e-12
    n_frames, n = frames.shape
    freqs = np.fft.rfftfreq(n, d=1.0 / sr).astype(np.float32)
    nyq = sr / 2.0

    energy = np.zeros(n_frames, dtype=np.float32)
    rms_db = np.zeros(n_frames, dtype=np.float32)
    centroid = np.zeros(n_frames, dtype=np.float32)
    bandwidth = np.zeros(n_frames, dtype=np.float32)
    rolloff = np.zeros(n_frames, dtype=np.float32)
    flatness = np.zeros(n_frames, dtype=np.float32)
    zcr = np.zeros(n_frames, dtype=np.float32)
    bandE = np.zeros((n_frames, 4), dtype=np.float32)
    spec_norm = []

    # Bandgrenzen (relativ robust)
    b1, b2, b3 = 1000.0, 3000.0, 6000.0
    b1 = min(b1, nyq)
    b2 = min(b2, nyq)
    b3 = min(b3, nyq)

    i1 = np.searchsorted(freqs, b1)
    i2 = np.searchsorted(freqs, b2)
    i3 = np.searchsorted(freqs, b3)

    for i in range(n_frames):
        x = frames[i] * window

        # energy / rms
        e = float(np.mean(x * x))
        energy[i] = e
        rms = math.sqrt(e + eps)
        rms_db[i] = 20.0 * math.log10(rms + eps)

        # zcr
        s = np.sign(x)
        zcr[i] = float(np.mean(s[1:] != s[:-1]))

        # spectrum
        X = np.fft.rfft(x)
        mag = np.abs(X).astype(np.float32)
        pwr = (mag * mag).astype(np.float32)

        p_sum = float(np.sum(pwr) + eps)

        # centroid
        centroid[i] = float(np.sum(freqs * pwr) / p_sum)

        # bandwidth
        bw = np.sum(((freqs - centroid[i]) ** 2) * pwr) / p_sum
        bandwidth[i] = float(math.sqrt(max(bw, 0.0)))

        # rolloff 0.85
        csum = np.cumsum(pwr)
        idx = int(np.searchsorted(csum, 0.85 * csum[-1]))
        idx = min(idx, len(freqs) - 1)
        rolloff[i] = float(freqs[idx])

        # flatness (geometric / arithmetic mean)
        gm = math.exp(float(np.mean(np.log(pwr + eps))))
        am = float(np.mean(pwr) + eps)
        flatness[i] = float(gm / am)

        # band energies
        bandE[i, 0] = float(np.sum(pwr[:i1]))
        bandE[i, 1] = float(np.sum(pwr[i1:i2]))
        bandE[i, 2] = float(np.sum(pwr[i2:i3]))
        bandE[i, 3] = float(np.sum(pwr[i3:]))

        # normalized spectrum vector for shape similarity
        denom = float(np.linalg.norm(mag) + eps)
        spec_norm.append(mag / denom)

    spec_norm = np.stack(spec_norm, axis=0).astype(np.float32)

    return {
        "energy": energy,
        "rms_db": rms_db,
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "rolloff_hz": rolloff,
        "flatness": flatness,
        "zcr": zcr,
        "bandE": bandE,
        "spec_norm": spec_norm,
    }


# ---------------------------------------
# DETECTORS
# ---------------------------------------

def detector_centroid_change(centroid_hz: np.ndarray) -> np.ndarray:
    d = np.zeros_like(centroid_hz)
    d[1:] = np.abs(np.diff(centroid_hz))
    return d


def detector_energy_change(rms_db: np.ndarray) -> np.ndarray:
    d = np.zeros_like(rms_db)
    d[1:] = np.abs(np.diff(rms_db))
    return d


def detector_boundary_jump(y: np.ndarray, hop_len: int, n_frames: int) -> np.ndarray:
    """
    Sprung an Frame-Grenzen: |y[start] - y[start-1]|
    """
    d = np.zeros(n_frames, dtype=np.float32)
    for i in range(1, n_frames):
        start = i * hop_len
        if 0 < start < len(y):
            d[i] = abs(float(y[start] - y[start - 1]))
    return d


def detector_shape_change(spec_norm: np.ndarray) -> np.ndarray:
    """
    1 - cosine similarity zwischen normierten Spektren
    """
    d = np.zeros(spec_norm.shape[0], dtype=np.float32)
    for i in range(1, len(d)):
        a = spec_norm[i - 1]
        b = spec_norm[i]
        sim = float(np.dot(a, b))  # beide normiert ~ cos
        sim = max(min(sim, 1.0), -1.0)
        d[i] = 1.0 - sim
    return d


# ---------------------------------------
# PEAK PICKING + FUSION
# ---------------------------------------

def pick_peaks(score: np.ndarray,
              percentile: float,
              min_gap_frames: int,
              consolidate_frames: int,
              hysteresis_frames: int) -> Tuple[np.ndarray, float]:
    """
    Peak picking mit:
      - Threshold via percentile
      - Hysteresis (optional)
      - lokale maxima
      - min gap
      - consolidate
    """
    thr = float(np.percentile(score[1:], percentile)) if len(score) > 2 else float(np.max(score))
    if hysteresis_frames > 1:
        gate = apply_hysteresis(score, thr, hysteresis_frames)
        gated_score = score.copy()
        gated_score[~gate] = 0.0
    else:
        gated_score = score

    peaks = local_maxima_peaks(gated_score, thr)
    peaks = enforce_min_gap(peaks, gated_score, min_gap_frames)
    peaks = consolidate_within_window(peaks, gated_score, consolidate_frames)
    return peaks, thr


def times_from_peaks(peaks: np.ndarray, hop_len: int, sr: int) -> np.ndarray:
    return (peaks.astype(np.float32) * hop_len) / float(sr)


def cluster_close_times(events: List[Tuple[float, str, float]],
                        max_diff_s: float) -> List[Dict]:
    """
    events: Liste von (time_s, method_name, method_score)
    Gibt Cluster-Objekte zurück.
    """
    if not events:
        return []

    events = sorted(events, key=lambda x: x[0])
    clusters = []
    cur = [events[0]]

    for e in events[1:]:
        if abs(e[0] - cur[-1][0]) <= max_diff_s:
            cur.append(e)
        else:
            clusters.append(cur)
            cur = [e]
    clusters.append(cur)

    out = []
    for c in clusters:
        times = [x[0] for x in c]
        methods = [x[1] for x in c]
        scores = [x[2] for x in c]
        out.append({
            "time_s": float(np.median(times)),
            "methods": set(methods),
            "score_sum": float(np.sum(scores)),
            "score_max": float(np.max(scores)),
            "members": c
        })
    return out


def fuse_changes(cfg: Config,
                 peaks_cent: np.ndarray, score_cent: np.ndarray, hop_len: int, sr: int,
                 peaks_eng: np.ndarray, score_eng: np.ndarray,
                 peaks_jump: np.ndarray, score_jump: np.ndarray,
                 peaks_shape: np.ndarray, score_shape: np.ndarray) -> Dict[str, List[float]]:
    """
    Fusion: erstellt priorisierte Change-Times.
    prio basiert auf Anzahl Methoden (und score_sum als Tie-Breaker).
    """
    # Events sammeln (time, method, weighted_score_at_peak)
    events = []

    for p in peaks_cent:
        t = float(p * hop_len / sr)
        events.append((t, "centroid", cfg.W_CENTROID * float(score_cent[p])))

    for p in peaks_eng:
        t = float(p * hop_len / sr)
        events.append((t, "energy", cfg.W_ENERGY * float(score_eng[p])))

    for p in peaks_jump:
        t = float(p * hop_len / sr)
        events.append((t, "jump", cfg.W_JUMP * float(score_jump[p])))

    for p in peaks_shape:
        t = float(p * hop_len / sr)
        events.append((t, "shape", cfg.W_SHAPE * float(score_shape[p])))

    clusters = cluster_close_times(events, cfg.JOINT_MAX_DIFF_S)

    prio1, prio2, prio3 = [], [], []
    for cl in clusters:
        m = cl["methods"]

        if cfg.IGNORE_ONLY_ENERGY_AND_JUMP and m.issubset({"energy", "jump"}) and len(m) <= 2:
            continue

        n = len(m)
        # prio: 4 Methoden -> prio1, 3->prio1, 2->prio2, 1->prio3
        if n >= 3:
            prio1.append(cl["time_s"])
        elif n == 2:
            prio2.append(cl["time_s"])
        else:
            prio3.append(cl["time_s"])

    return {
        "prio1": sorted(prio1),
        "prio2": sorted(prio2),
        "prio3": sorted(prio3),
        "clusters": clusters
    }


# ---------------------------------------
# SEGMENTS
# ---------------------------------------

def build_boundaries(cfg: Config, duration_s: float, fused: Dict[str, List[float]]) -> List[float]:
    bounds = [0.0, float(duration_s)]

    if cfg.USE_PRIO1_IN_BOUNDARIES:
        bounds += fused["prio1"]
    if cfg.USE_PRIO2_IN_BOUNDARIES:
        bounds += fused["prio2"]
    if cfg.USE_PRIO3_IN_BOUNDARIES:
        bounds += fused["prio3"]

    # sort + unique + clamp
    bounds = [b for b in bounds if 0.0 <= b <= duration_s]
    bounds = sorted(bounds)
    # unique with tolerance
    out = []
    eps = 1e-6
    for b in bounds:
        if not out or abs(b - out[-1]) > eps:
            out.append(b)

    # remove too-close boundaries
    cleaned = [out[0]]
    for b in out[1:]:
        if (b - cleaned[-1]) >= cfg.MIN_SEGMENT_DURATION_S:
            cleaned.append(b)
    if cleaned[-1] < duration_s:
        cleaned.append(duration_s)
    return cleaned


def segments_from_boundaries(bounds: List[float]) -> List[Tuple[float, float]]:
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def merge_short_segments(cfg: Config,
                         segs: List[Tuple[float, float]],
                         seg_feats: np.ndarray) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Merged Segmente < MIN_SEGMENT_DURATION_S in Nachbarn (heuristisch: nearest in feature space).
    seg_feats: pro Segment 2D/ND Features (z.B. centroid,rms)
    """
    if not cfg.MERGE_SHORT_SEGMENTS or len(segs) <= 2:
        return segs, seg_feats

    segs = list(segs)
    feats = seg_feats.copy()

    changed = True
    while changed and len(segs) > 2:
        changed = False
        lens = np.array([b - a for a, b in segs], dtype=np.float32)
        idx_short = np.where(lens < cfg.MIN_SEGMENT_DURATION_S)[0]
        if len(idx_short) == 0:
            break

        i = int(idx_short[0])
        # choose neighbor (left or right) by feature distance
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = len(segs) - 2
        else:
            dl = float(np.linalg.norm(feats[i] - feats[i - 1]))
            dr = float(np.linalg.norm(feats[i] - feats[i + 1]))
            j = i - 1 if dl <= dr else i + 1

        # merge i into j
        a1, b1 = segs[i]
        a2, b2 = segs[j]
        new_seg = (min(a1, a2), max(b1, b2))

        # new feature weighted by length
        w1 = b1 - a1
        w2 = b2 - a2
        new_feat = (feats[i] * w1 + feats[j] * w2) / max(w1 + w2, 1e-9)

        keep = [k for k in range(len(segs)) if k not in (i, j)]
        segs = [segs[k] for k in keep] + [new_seg]
        feats = np.vstack([feats[k] for k in keep] + [new_feat])

        # sort by start time
        order = np.argsort([s[0] for s in segs])
        segs = [segs[k] for k in order]
        feats = feats[order]

        changed = True

    return segs, feats


# ---------------------------------------
# SEGMENT FEATURES (Freq + Amp)
# ---------------------------------------

def segment_features_from_frames(segments: List[Tuple[float, float]],
                                 frame_feats: Dict[str, np.ndarray],
                                 sr: int,
                                 hop_len: int) -> np.ndarray:
    """
    Für jedes Segment: mean centroid_hz + mean rms_db (2D)
    """
    cent = frame_feats["centroid_hz"]
    rms_db = frame_feats["rms_db"]

    feats = []
    for (a, b) in segments:
        i0 = int(round(a * sr / hop_len))
        i1 = int(round(b * sr / hop_len))
        i0 = max(i0, 0)
        i1 = min(i1, len(cent))
        if i1 <= i0:
            feats.append([float(cent[min(i0, len(cent)-1)]), float(rms_db[min(i0, len(rms_db)-1)])])
        else:
            feats.append([float(np.mean(cent[i0:i1])), float(np.mean(rms_db[i0:i1]))])

    return np.array(feats, dtype=np.float32)


# ---------------------------------------
# KMEANS (fallback, ohne sklearn)
# ---------------------------------------

def kmeans_simple(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k <= 1 or n == 0:
        return np.zeros(n, dtype=int)

    # init centers from random points
    idx = rng.choice(n, size=min(k, n), replace=False)
    centers = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # assign
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=-1)  # (n,k)
        new_labels = np.argmin(dists, axis=1)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        # update centers
        for j in range(k):
            pts = X[labels == j]
            if len(pts) > 0:
                centers[j] = np.mean(pts, axis=0)
            else:
                # re-seed empty cluster
                centers[j] = X[rng.integers(0, n)]
    return labels


def assign_segments_to_sources(cfg: Config, seg_feats_2d: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """
    Clustering im (centroid_hz, rms_db) Raum.
    Gibt labels (int) + cluster_info zurück.
    """
    Xn, med, scale = robust_scale(seg_feats_2d)

    # NUM_SIGNALS: wenn None -> einfache Heuristik: 2 oder 3 nach Spread im centroid
    if cfg.NUM_SIGNALS is None:
        spread = float(np.percentile(seg_feats_2d[:, 0], 90) - np.percentile(seg_feats_2d[:, 0], 10))
        k = 3 if spread > 1500.0 else 2
    else:
        k = int(cfg.NUM_SIGNALS)

    labels = kmeans_simple(Xn, k=k, iters=cfg.KMEANS_ITERS, seed=cfg.KMEANS_SEED)

    # cluster infos
    info = {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            info[c] = {"count": 0, "mean_centroid": 0.0, "mean_rms_db": -999.0}
        else:
            info[c] = {
                "count": int(len(idx)),
                "mean_centroid": float(np.mean(seg_feats_2d[idx, 0])),
                "mean_rms_db": float(np.mean(seg_feats_2d[idx, 1])),
            }

    # merge too-small clusters into nearest (by centroid/rms)
    changed = True
    while changed:
        changed = False
        for c in list(info.keys()):
            if info[c]["count"] > 0 and info[c]["count"] < cfg.MIN_SEGMENTS_PER_CLUSTER:
                # find nearest other cluster
                c_feat = np.array([info[c]["mean_centroid"], info[c]["mean_rms_db"]], dtype=np.float32)
                best = None
                best_d = 1e9
                for d in info.keys():
                    if d == c or info[d]["count"] == 0:
                        continue
                    d_feat = np.array([info[d]["mean_centroid"], info[d]["mean_rms_db"]], dtype=np.float32)
                    dist = float(np.linalg.norm((c_feat - d_feat) / np.array([500.0, 6.0], dtype=np.float32)))
                    if dist < best_d:
                        best_d = dist
                        best = d
                if best is not None and best_d < cfg.MERGE_THRESH:
                    labels[labels == c] = best
                    changed = True

        # rebuild info if changed
        if changed:
            uniq = sorted(set(labels.tolist()))
            remap = {old: i for i, old in enumerate(uniq)}
            labels = np.array([remap[x] for x in labels], dtype=int)
            k2 = len(uniq)
            info = {}
            for c in range(k2):
                idx = np.where(labels == c)[0]
                info[c] = {
                    "count": int(len(idx)),
                    "mean_centroid": float(np.mean(seg_feats_2d[idx, 0])) if len(idx) else 0.0,
                    "mean_rms_db": float(np.mean(seg_feats_2d[idx, 1])) if len(idx) else -999.0,
                }

    # sort clusters by centroid, map to A,B,C...
    order = sorted(info.keys(), key=lambda c: info[c]["mean_centroid"])
    map_sorted = {old: new for new, old in enumerate(order)}
    labels = np.array([map_sorted[x] for x in labels], dtype=int)

    info2 = {}
    for old, new in map_sorted.items():
        info2[new] = info[old]

    return labels, info2


# ---------------------------------------
# RECONSTRUCTION
# ---------------------------------------

def reconstruct_tracks(cfg: Config,
                       y: np.ndarray,
                       sr: int,
                       segments: List[Tuple[float, float]],
                       seg_labels: np.ndarray,
                       out_dir: str) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    os.makedirs(out_dir, exist_ok=True)

    uniq = sorted(set(seg_labels.tolist()))
    exported: Dict[str, str] = {}
    tracks: Dict[str, np.ndarray] = {}

    for lab in uniq:
        name = chr(ord('A') + lab)

        if cfg.EXPORT_MODE == "mask":
            out = np.zeros_like(y, dtype=np.float32)
            for (a, b), L in zip(segments, seg_labels):
                if L != lab:
                    continue
                s0 = int(round(a * sr))
                s1 = int(round(b * sr))
                s0 = max(0, min(s0, len(y)))
                s1 = max(0, min(s1, len(y)))
                out[s0:s1] = y[s0:s1]

        elif cfg.EXPORT_MODE == "concat":
            parts = []
            for (a, b), L in zip(segments, seg_labels):
                if L != lab:
                    continue
                s0 = int(round(a * sr))
                s1 = int(round(b * sr))
                parts.append(y[s0:s1])
            out = np.concatenate(parts, axis=0) if parts else np.zeros(1, dtype=np.float32)

        else:
            raise ValueError("EXPORT_MODE must be 'mask' or 'concat'")

        # --- Track im RAM behalten fürs Plotten ---
        tracks[name] = out

        # --- Export ---
        seg = float_to_audiosegment(out, sr)
        out_path = os.path.join(out_dir, f"reconstruct_{name}.{cfg.EXPORT_FORMAT}")

        if cfg.EXPORT_FORMAT.lower() == "mp3":
            seg.export(out_path, format="mp3", bitrate="192k")
        else:
            seg.export(out_path, format="wav")

        exported[name] = out_path

        if cfg.VERBOSE:
            print(f"[EXPORT] Label {name}: {out_path}")

    return exported, tracks

# ---------------------------------------
# PLOTTING / EVAL
# ---------------------------------------

def plot_results(cfg: Config,
                 y: np.ndarray,
                 sr: int,
                 hop_len: int,
                 frame_feats: Dict[str, np.ndarray],
                 peaks: Dict[str, np.ndarray],
                 fused: Dict[str, List[float]],
                 boundaries: List[float]) -> None:
    t = np.arange(len(y), dtype=np.float32) / float(sr)

    tmax = cfg.PLOT_MAX_SECONDS
    if tmax is not None:
        nmax = int(round(tmax * sr))
        y_plot = y[:nmax]
        t_plot = t[:nmax]
        t_end = tmax
    else:
        y_plot = y
        t_plot = t
        t_end = float(t[-1]) if len(t) else 0.0

    # --- 2 Plots übereinander ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    # =========================
    # OBERER PLOT: Peaks (ohne Final Boundaries)
    # =========================
    ax1.plot(t_plot, y_plot, linewidth=0.8, color="black", label="Signal")

    def vlines_from_peaks(ax, p, color, ls, label_once: str):
        first = True
        for idx in p:
            ts = float(idx * hop_len / sr)
            if tmax is None or ts <= tmax:
                ax.axvline(ts, color=color, linestyle=ls, linewidth=1.0,
                           label=(label_once if first else None))
                first = False

    vlines_from_peaks(ax1, peaks["centroid"], color="tab:orange", ls="--", label_once="Centroid peaks")
    vlines_from_peaks(ax1, peaks["energy"],   color="tab:green",  ls="-.", label_once="Energy peaks")
    vlines_from_peaks(ax1, peaks["jump"],     color="tab:purple", ls=":",  label_once="Jump peaks")
    vlines_from_peaks(ax1, peaks["shape"],    color="tab:cyan",   ls=(0, (3, 1, 1, 1)), label_once="Shape peaks")

    # Ground Truth Marker (optional) – bleibt im oberen Plot
    if cfg.TRUE_SEGMENT_DURATION_S is not None:
        gt = cfg.TRUE_SEGMENT_DURATION_S
        x = gt
        first_gt = True
        while x < t_end:
            ax1.axvline(x, color="gray", linestyle=(0, (1, 3)), linewidth=1.0,
                        label=("Ground truth" if first_gt else None))
            first_gt = False
            x += gt

    ax1.set_title("Waveform + Detected Change Points (Peaks)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")

    # =========================
    # UNTERER PLOT: Final Boundaries (ohne Peaks)
    # =========================
    ax2.plot(t_plot, y_plot, linewidth=0.8, color="black", label="Signal")

    first_b = True
    for b in boundaries:
        if b == 0.0:
            continue
        if tmax is not None and b > tmax:
            continue
        ax2.axvline(b, color="tab:red", linewidth=2.5,
                    label=("Final boundaries" if first_b else None))
        first_b = False

    # Ground Truth Marker (optional) – auch unten, damit Vergleich Boundaries↔GT leicht ist
    if cfg.TRUE_SEGMENT_DURATION_S is not None:
        gt = cfg.TRUE_SEGMENT_DURATION_S
        x = gt
        first_gt2 = True
        while x < t_end:
            ax2.axvline(x, color="gray", linestyle=(0, (1, 3)), linewidth=1.0,
                        label=("Ground truth" if first_gt2 else None))
            first_gt2 = False
            x += gt

    ax2.set_title("Waveform + Final Boundaries")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # --- Centroid-Plot bleibt wie bisher ---
    cent = frame_feats["centroid_hz"]
    tt = (np.arange(len(cent)) * hop_len) / float(sr)
    if tmax is not None:
        mask = tt <= tmax
        tt = tt[mask]
        cent = cent[mask]

    plt.figure()
    plt.plot(tt, cent, linewidth=1.0, color="black", label="Centroid (Hz)")
    first_b2 = True
    for b in boundaries:
        if tmax is None or b <= tmax:
            plt.axvline(b, color="tab:red", linewidth=2, label=("Final boundaries" if first_b2 else None))
            first_b2 = False
    plt.title("Centroid (Hz) + Boundaries")
    plt.xlabel("Time (s)")
    plt.ylabel("Centroid (Hz)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    #plt.show()


def eval_against_ground_truth(cfg: Config, boundaries: List[float]) -> None:
    """
    Wenn TRUE_SEGMENT_DURATION_S gesetzt ist: grobe Trefferstatistik.
    """
    if cfg.TRUE_SEGMENT_DURATION_S is None:
        return

    gt = cfg.TRUE_SEGMENT_DURATION_S
    if len(boundaries) < 3:
        return

    duration = boundaries[-1]
    gt_marks = np.arange(gt, duration, gt)
    det = np.array(boundaries[1:-1], dtype=np.float32)  # exclude 0 and end

    tol = max(0.002, cfg.JOINT_MAX_DIFF_S)  # Toleranz
    hits = 0
    for g in gt_marks:
        if np.any(np.abs(det - g) <= tol):
            hits += 1

    print(f"[EVAL] GroundTruth step={gt:.3f}s, tol={tol:.3f}s -> hits {hits}/{len(gt_marks)}")


def plot_reconstructed_signals(cfg: Config, tracks: Dict[str, np.ndarray], sr: int) -> None:
    if not tracks:
        return

    for name in sorted(tracks.keys()):
        y = tracks[name]
        t = np.arange(len(y), dtype=np.float32) / float(sr)

        if cfg.PLOT_MAX_SECONDS is not None:
            nmax = int(round(cfg.PLOT_MAX_SECONDS * sr))
            y = y[:nmax]
            t = t[:nmax]

        plt.figure()
        plt.plot(t, y, linewidth=0.8, color="black")
        plt.title(f"Reconstructed Signal {name} ({cfg.EXPORT_MODE})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

# ---------------------------------------
# MAIN PIPELINE
# ---------------------------------------

def main():
    cfg = Config()

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # --- load ---
    y, sr = load_audio_mono(cfg.INPUT_FILE)
    duration_s = len(y) / float(sr)
    if cfg.VERBOSE:
        print(f"[LOAD] sr={sr} Hz, samples={len(y)}, duration={duration_s:.3f}s")

    # --- frame params ---
    frame_len = max(16, int(round(cfg.FRAME_DURATION_S * sr)))
    hop_s = cfg.HOP_DURATION_S if cfg.HOP_DURATION_S is not None else cfg.FRAME_DURATION_S
    hop_len = max(1, int(round(hop_s * sr)))
    if cfg.VERBOSE:
        print(f"[FRAME] frame_len={frame_len} samples ({frame_len/sr:.6f}s), hop_len={hop_len} samples ({hop_len/sr:.6f}s)")

    window = get_window(cfg.WINDOW, frame_len)

    frames = frame_signal(y, sr, frame_len, hop_len)
    n_frames = frames.shape[0]

    # --- features ---
    feats = compute_frame_features(frames, sr, window)

    # --- smoothing (reduces multiple triggers) ---
    feats["centroid_hz"] = median_filter_1d(feats["centroid_hz"], k=5)
    feats["rms_db"] = median_filter_1d(feats["rms_db"], k=5)

    # --- detectors (scores) ---
    score_cent = detector_centroid_change(feats["centroid_hz"])
    score_eng  = detector_energy_change(feats["rms_db"])
    score_jump = detector_boundary_jump(y, hop_len, n_frames)
    score_shape= detector_shape_change(feats["spec_norm"])

    # optional smoothing of scores
    score_cent = median_filter_1d(score_cent, k=3)
    score_eng  = median_filter_1d(score_eng,  k=3)
    score_shape= median_filter_1d(score_shape, k=3)

    # --- peak picking ---
    min_gap_frames = max(1, int(round(cfg.MIN_GAP_S * sr / hop_len)))
    cons_frames    = max(1, int(round(cfg.CONSOLIDATE_WITHIN_S * sr / hop_len)))

    peaks_cent, thr_cent = pick_peaks(score_cent, cfg.PERCENTILE_FEATURE, min_gap_frames, cons_frames, cfg.HYSTERESIS_FRAMES)
    peaks_eng,  thr_eng  = pick_peaks(score_eng,  cfg.PERCENTILE_FEATURE, min_gap_frames, cons_frames, cfg.HYSTERESIS_FRAMES)

    min_gap_jump_frames = max(1, int(round(cfg.JUMP_MIN_GAP_S * sr / hop_len)))
    peaks_jump, thr_jump = pick_peaks(score_jump, cfg.JUMP_PERCENTILE, min_gap_jump_frames, cons_frames, cfg.HYSTERESIS_FRAMES)

    peaks_shape, thr_shape = pick_peaks(score_shape, cfg.PERCENTILE_FEATURE, min_gap_frames, cons_frames, cfg.HYSTERESIS_FRAMES)

    if cfg.VERBOSE:
        print(f"[PEAKS] centroid={len(peaks_cent)} (thr={thr_cent:.3g}) | energy={len(peaks_eng)} (thr={thr_eng:.3g}) | jump={len(peaks_jump)} (thr={thr_jump:.3g}) | shape={len(peaks_shape)} (thr={thr_shape:.3g})")

    # --- fusion ---
    fused = fuse_changes(cfg,
                         peaks_cent, score_cent, hop_len, sr,
                         peaks_eng, score_eng,
                         peaks_jump, score_jump,
                         peaks_shape, score_shape)

    if cfg.VERBOSE:
        print(f"[FUSE] prio1={len(fused['prio1'])}, prio2={len(fused['prio2'])}, prio3={len(fused['prio3'])}")

    # --- boundaries + segments ---
    boundaries = build_boundaries(cfg, duration_s, fused)
    segments = segments_from_boundaries(boundaries)

    # --- segment features (freq+amp) ---
    seg_feats = segment_features_from_frames(segments, feats, sr, hop_len)

    # merge short segments (optional)
    segments, seg_feats = merge_short_segments(cfg, segments, seg_feats)

    # --- labeling ---
    seg_labels, cluster_info = assign_segments_to_sources(cfg, seg_feats)

    if cfg.VERBOSE:
        print("[CLUSTERS]")
        for c in sorted(cluster_info.keys()):
            name = chr(ord('A') + c)
            print(f"  {name}: count={cluster_info[c]['count']}, centroid≈{cluster_info[c]['mean_centroid']:.0f} Hz, rms≈{cluster_info[c]['mean_rms_db']:.1f} dB")

    # --- reconstruct all labels ---
    exported, tracks = reconstruct_tracks(cfg, y, sr, segments, seg_labels, cfg.OUT_DIR)

    # --- plots ---
    peaks_dict = {
        "centroid": peaks_cent,
        "energy": peaks_eng,
        "jump": peaks_jump,
        "shape": peaks_shape
    }

    plot_results(cfg, y, sr, hop_len, feats, peaks_dict, fused, boundaries)
    #plot_reconstructed_signals(cfg, tracks, sr)

    # --- eval against known true segment duration (optional) ---
    eval_against_ground_truth(cfg, boundaries)

    if cfg.VERBOSE:
        print("[DONE] Exported tracks:")
        for k, v in exported.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
