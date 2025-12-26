"""
Änderungen zu v5.6:
- Hinzufügen von Deadzone   v5.5
- Hinzufügen von CFAR       v5.4
- Hinzufügen von loop       v5.3

Neu in v5.8:
- MIN_AMPLITUDE_THRESHOLD: Ignoriert Frames unter Minimalamplitude
- valid_frames Flag: Markiert gültige/ungültige Frames
- Frames unter Schwelle werden als Störung behandelt
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pydub import AudioSegment
import time
from typing import Optional


@dataclass
class Config:
    # I/O
    INPUT_FILE: str = r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_1k_GOD_30sec_rand.mp3"
    OUT_DIR: str = "output_segments"
    
    # Analyse-Parameter
    WINDOW_MS: float = 2.0
    HOP_MS: float = 0.5
    
    # ⭐ NEU: Minimale Amplitude für gültige Frames
    MIN_AMPLITUDE_THRESHOLD: float = 0.01  # Frames unter diesem RMS-Wert werden ignoriert
    
    CHANGE_DETECTION_CFAR: bool = False

    # Change Detection
    CHANGE_THRESHOLD_PERCENTILE: float = 85.0
    MIN_SEGMENT_MS: float = 5.0
    MERGE_TOLERANCE_MS: float = 5.0

    # CFAR Change Detection
    CFAR_METHOD: str = "OS"
    CFAR_GUARD_CELLS: int = 3
    CFAR_TRAIN_CELLS: int = 15
    CFAR_ALPHA: float = 3.5
    CFAR_K_FRACTION: float = 0.75
    
    # Clustering
    NUM_CLUSTERS: int = None
    
    # Export
    EXPORT_FORMAT: str = "mp3"
    VERBOSE: bool = True

    # Weight Change Score
    WEIGHT_CENTROID: int = 2.0
    WEIGHT_RMS: int = 1.0
    WEIGHT_Rolloff: int = 1.5
    WEIGHT_ZCR: int = 1.5
    WEIGHT_FLUX: int = 5
    WEIGHT_BANDWIDTH: int = 1.0
    
    # Dead-Zone um Wechselstellen
    DEADZONE_MS: float = 0

    #Überlagerung von zwei segmenten bei wiederherstellung für bessere ergebnisse
    FADE_IN: float = 0

# ============================================================================
# AUDIO I/O
# ============================================================================

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Lädt Audio als mono float32 [-1,1]"""
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())
    
    if audio.sample_width == 2:
        y = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        y = samples.astype(np.float32) / 2147483648.0
    else:
        y = samples.astype(np.float32) / max(np.abs(samples).max(), 1)
    
    return y, sr

def save_audio(y: np.ndarray, sr: int, path: str, fmt: str = "mp3"):
    """Speichert Audio"""
    y = np.clip(y, -1.0, 1.0)
    int16 = (y * 32767.0).astype(np.int16)
    seg = AudioSegment(
        data=int16.tobytes(),
        sample_width=2,
        frame_rate=sr,
        channels=1
    )
    seg.export(path, format=fmt, bitrate="192k" if fmt == "mp3" else None)

# ============================================================================
# FEATURE EXTRACTION - MIT AMPLITUDENSCHWELLE
# ============================================================================

def compute_stft_features(y: np.ndarray, sr: int, window_ms: float, hop_ms: float, 
                          min_amplitude: float = 0.0):
    """
    Berechnet erweiterte Spektral-Features mit STFT
    
    ⭐ NEU: Frames unter min_amplitude werden als ungültig markiert
    
    Returns:
        dict mit 'valid_frames' boolean array - True = gültiger Frame
    """
    
    window_samples = int(window_ms * sr / 1000)
    hop_samples = int(hop_ms * sr / 1000)
    
    if window_samples % 2 != 0:
        window_samples += 1
    
    window = np.hanning(window_samples)
    n_frames = 1 + (len(y) - window_samples) // hop_samples
    
    # Features Arrays
    centroids = np.zeros(n_frames)
    rms_values = np.zeros(n_frames)
    rolloffs = np.zeros(n_frames)
    zcr_values = np.zeros(n_frames)
    flux_values = np.zeros(n_frames)
    bandwidth_values = np.zeros(n_frames)
    valid_frames = np.ones(n_frames, dtype=bool)  # ⭐ NEU: Gültigkeits-Flag
    
    freqs = np.fft.rfftfreq(window_samples, 1/sr)
    prev_spec = None
    
    for i in range(n_frames):
        start = i * hop_samples
        frame = y[start:start + window_samples]
        
        # ⭐ NEU: Prüfe Amplitude des Frames
        frame_rms = np.sqrt(np.mean(frame**2))
        
        if frame_rms < min_amplitude:
            # Frame ist zu leise - markiere als ungültig
            valid_frames[i] = False
            rms_values[i] = frame_rms  # Speichere trotzdem RMS für Visualisierung
            # Alle anderen Features bleiben 0
            continue
        
        # Frame ist gültig - berechne normale Features
        rms_values[i] = frame_rms
        zcr_values[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        
        windowed_frame = frame * window
        spec = np.abs(np.fft.rfft(windowed_frame))
        spec_power = spec ** 2
        
        if spec_power.sum() > 1e-10:
            centroids[i] = np.sum(freqs * spec_power) / spec_power.sum()
            
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                rolloffs[i] = freqs[rolloff_idx[0]]
            
            bandwidth_values[i] = np.sqrt(
                np.sum(((freqs - centroids[i]) ** 2) * spec_power) / spec_power.sum()
            )
            
            if prev_spec is not None:
                flux_values[i] = np.sum((spec - prev_spec) ** 2)
            
            prev_spec = spec.copy()
    
    times = np.arange(n_frames) * hop_samples / sr
    
    return {
        'times': times,
        'centroid': centroids,
        'rms': rms_values,
        'rolloff': rolloffs,
        'zcr': zcr_values,
        'flux': flux_values,
        'bandwidth': bandwidth_values,
        'hop_samples': hop_samples,
        'valid_frames': valid_frames  # ⭐ NEU
    }

# ============================================================================
# CHANGE DETECTION - MIT GÜLTIGKEITSPRÜFUNG
# ============================================================================

def detect_changes(wc:float, wrms:float, wroll:float, wzcr:float, wflux:float, wbw:float, 
                   features: Dict, threshold_percentile: float) -> np.ndarray:
    """
    Erkennt Änderungspunkte - ignoriert ungültige Frames
    """
    
    centroid = features['centroid']
    rms = features['rms']
    rolloff = features['rolloff']
    zcr = features['zcr']
    flux = features['flux']
    bandwidth = features['bandwidth']
    valid_frames = features['valid_frames']  # ⭐ NEU
    
    def normalize(x, valid_mask):
        x = np.copy(x)
        # Nur gültige Frames für Percentile verwenden
        valid_values = x[valid_mask]
        if len(valid_values) > 0:
            x_min, x_max = np.percentile(valid_values, [5, 95])
            if x_max - x_min > 0:
                x = (x - x_min) / (x_max - x_min)
        return np.clip(x, 0, 1)
    
    cent_norm = normalize(centroid, valid_frames)
    rms_norm = normalize(rms, valid_frames)
    roll_norm = normalize(rolloff, valid_frames)
    zcr_norm = normalize(zcr, valid_frames)
    flux_norm = normalize(flux, valid_frames)
    bw_norm = normalize(bandwidth, valid_frames)
    
    cent_change = np.abs(np.diff(cent_norm, prepend=cent_norm[0]))
    rms_change = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    roll_change = np.abs(np.diff(roll_norm, prepend=roll_norm[0]))
    zcr_change = np.abs(np.diff(zcr_norm, prepend=zcr_norm[0]))
    flux_change = np.abs(np.diff(flux_norm, prepend=flux_norm[0]))
    bw_change = np.abs(np.diff(bw_norm, prepend=bw_norm[0]))
    
    combined_change = (
        wc * cent_change +
        wrms * rms_change +
        wroll * roll_change +
        wzcr * zcr_change +
        wflux * flux_change +
        wbw * bw_change
    ) / (wc + wrms + wroll + wzcr + wflux + wbw)
    
    # ⭐ NEU: Ungültige Frames auf 0 setzen
    combined_change[~valid_frames] = 0.0
    
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    combined_change = np.convolve(combined_change, kernel, mode='same')
    
    # Threshold nur aus gültigen Frames berechnen
    valid_changes = combined_change[valid_frames]
    if len(valid_changes) > 0:
        threshold = np.percentile(valid_changes, threshold_percentile)
    else:
        threshold = 0.0
    
    return combined_change, threshold

def find_boundaries(change_score: np.ndarray, threshold: float, 
                   times: np.ndarray, min_segment_s: float,
                   valid_frames: np.ndarray) -> List[float]:
    """
    Findet Segment-Grenzen - ignoriert ungültige Bereiche
    """
    
    candidates = []
    for i in range(1, len(change_score)-1):
        # ⭐ NEU: Nur gültige Frames als Kandidaten
        if (valid_frames[i] and 
            change_score[i] > threshold and 
            change_score[i] >= change_score[i-1] and 
            change_score[i] >= change_score[i+1]):
            candidates.append(times[i])
    
    if len(candidates) < 2:
        return [0.0, times[-1]]
    
    boundaries = [0.0]
    for t in candidates:
        if t - boundaries[-1] >= min_segment_s:
            boundaries.append(t)
    
    boundaries.append(times[-1])
    
    return boundaries

# ============================================================================
# CHANGE DETECTION mit CFAR - MIT GÜLTIGKEITSPRÜFUNG
# ============================================================================

def detect_changes_cfar(wc:float, wrms:float, wroll:float, wzcr:float, wflux:float, wbw:float, 
                        features: Dict, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erkennt Änderungspunkte mit CFAR - ignoriert ungültige Frames
    """
    
    centroid = features['centroid']
    rms = features['rms']
    rolloff = features['rolloff']
    zcr = features['zcr']
    flux = features['flux']
    bandwidth = features['bandwidth']
    valid_frames = features['valid_frames']  # ⭐ NEU
    
    def normalize(x, valid_mask):
        x = np.copy(x)
        valid_values = x[valid_mask]
        if len(valid_values) > 0:
            x_min, x_max = np.percentile(valid_values, [5, 95])
            if x_max - x_min > 0:
                x = (x - x_min) / (x_max - x_min)
        return np.clip(x, 0, 1)
    
    cent_norm = normalize(centroid, valid_frames)
    rms_norm = normalize(rms, valid_frames)
    roll_norm = normalize(rolloff, valid_frames)
    zcr_norm = normalize(zcr, valid_frames)
    flux_norm = normalize(flux, valid_frames)
    bw_norm = normalize(bandwidth, valid_frames)

    cent_change = np.abs(np.diff(cent_norm, prepend=cent_norm[0]))
    rms_change = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    roll_change = np.abs(np.diff(roll_norm, prepend=roll_norm[0]))
    zcr_change = np.abs(np.diff(zcr_norm, prepend=zcr_norm[0]))
    flux_change = np.abs(np.diff(flux_norm, prepend=flux_norm[0]))
    bw_change = np.abs(np.diff(bw_norm, prepend=bw_norm[0]))

    combined_change = (
        wc * cent_change +
        wrms * rms_change +
        wroll * roll_change +
        wzcr * zcr_change +
        wflux * flux_change +
        wbw * bw_change
    ) / (wc + wrms + wroll + wzcr + wflux + wbw)
    
    # ⭐ NEU: Ungültige Frames auf 0 setzen
    combined_change[~valid_frames] = 0.0
    
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    combined_change = np.convolve(combined_change, kernel, mode='same')
    
    if cfg.CFAR_METHOD == "PERCENTILE":
        threshold_val = np.percentile(combined_change[valid_frames], 85.0)
        threshold = np.full_like(combined_change, threshold_val)
    else:
        threshold = cfar_1d(
            combined_change,
            guard=cfg.CFAR_GUARD_CELLS,
            train=cfg.CFAR_TRAIN_CELLS,
            alpha=cfg.CFAR_ALPHA,
            method=cfg.CFAR_METHOD,
            k_fraction=cfg.CFAR_K_FRACTION
        )
    
    return combined_change, threshold

def find_boundaries_cfar(change_score: np.ndarray, threshold: np.ndarray, 
                        times: np.ndarray, min_segment_s: float,
                        valid_frames: np.ndarray) -> List[float]:
    """
    Findet Boundaries mit adaptivem Threshold - ignoriert ungültige Bereiche
    """
    
    candidates = []
    for i in range(1, len(change_score)-1):
        # ⭐ NEU: Nur gültige Frames als Kandidaten
        if (valid_frames[i] and
            change_score[i] > threshold[i] and 
            change_score[i] >= change_score[i-1] and 
            change_score[i] >= change_score[i+1]):
            candidates.append(times[i])
    
    if len(candidates) < 2:
        return [0.0, times[-1]]
    
    boundaries = [0.0]
    for t in candidates:
        if t - boundaries[-1] >= min_segment_s:
            boundaries.append(t)
    
    boundaries.append(times[-1])
    
    return boundaries

# ============================================================================
# CFAR IMPLEMENTATIONS
# ============================================================================

def cfar_1d(signal: np.ndarray, guard: int, train: int, alpha: float, 
            method: str = "CA", k_fraction: float = 0.75) -> np.ndarray:
    """1D CFAR Detector"""
    
    n = len(signal)
    threshold = np.zeros(n)
    window_half = guard + train
    
    for i in range(n):
        left_start = max(0, i - window_half)
        left_end = max(0, i - guard)
        right_start = min(n, i + guard + 1)
        right_end = min(n, i + window_half + 1)
        
        left_cells = signal[left_start:left_end]
        right_cells = signal[right_start:right_end]
        train_cells = np.concatenate([left_cells, right_cells])
        
        if len(train_cells) == 0:
            threshold[i] = 0
            continue
        
        if method == "CA":
            noise_level = np.mean(train_cells)
        elif method == "OS":
            k = int(len(train_cells) * k_fraction)
            k = max(0, min(k, len(train_cells) - 1))
            sorted_cells = np.sort(train_cells)
            noise_level = sorted_cells[k]
        elif method == "SO":
            left_avg = np.mean(left_cells) if len(left_cells) > 0 else 0
            right_avg = np.mean(right_cells) if len(right_cells) > 0 else 0
            noise_level = min(left_avg, right_avg) if left_avg > 0 and right_avg > 0 else max(left_avg, right_avg)
        else:
            raise ValueError(f"Unknown CFAR method: {method}")
        
        threshold[i] = alpha * noise_level
    
    return threshold

# ============================================================================
# SEGMENTATION & CLUSTERING
# ============================================================================

def create_segments(boundaries: List[float], features: Dict,
                    deadzone_ms: float = 0.0,
                    min_segment_ms: float = 0.0,
                    verbose: bool = False) -> List[Tuple[float, float]]:
    """
    Erstellt Segment-Paare - filtert Segmente mit zu vielen ungültigen Frames
    """
    dz = deadzone_ms / 1000.0
    segments: List[Tuple[float, float]] = []
    
    times = features['times']
    valid_frames = features['valid_frames']
    hop_samples = features['hop_samples']

    n = len(boundaries)
    for i in range(n - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        if i > 0:
            start += dz
        if i < n - 2:
            end -= dz

        if end <= start:
            if verbose:
                print(f"  [skip] Segment {i}: zu kurz nach Dead-Zone")
            continue

        if min_segment_ms > 0 and (end - start) * 1000.0 < min_segment_ms:
            if verbose:
                print(f"  [skip] Segment {i}: {(end-start)*1000:.2f} ms < MIN_SEGMENT_MS")
            continue
        
        # ⭐ NEU: Prüfe ob Segment genügend gültige Frames hat
        start_idx = np.searchsorted(times, start)
        end_idx = np.searchsorted(times, end)
        segment_valid = valid_frames[start_idx:end_idx]
        
        if len(segment_valid) > 0:
            valid_ratio = np.sum(segment_valid) / len(segment_valid)
            if valid_ratio < 0.3:  # Mindestens 30% gültige Frames
                if verbose:
                    print(f"  [skip] Segment {i}: nur {valid_ratio*100:.1f}% gültige Frames")
                continue

        segments.append((start, end))

    return segments

def extract_segment_features(y: np.ndarray, sr: int, 
                            segments: List[Tuple[float, float]],
                            min_amplitude: float = 0.0) -> np.ndarray:
    """
    Extrahiert Features für jedes Segment - prüft Amplitude
    """
    
    features = []
    
    for start_t, end_t in segments:
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        segment = y[start_idx:end_idx]
        
        if len(segment) < 10:
            features.append([0, 0, 0, 0, 0, 0])
            continue
        
        # ⭐ NEU: Prüfe Segment-RMS
        segment_rms = np.sqrt(np.mean(segment**2))
        if segment_rms < min_amplitude:
            features.append([0, 0, 0, 0, 0, 0])
            continue
        
        rms = segment_rms
        zcr = np.sum(np.abs(np.diff(np.sign(segment)))) / (2 * len(segment))
        
        window = np.hanning(len(segment))
        spec = np.abs(np.fft.rfft(segment * window))
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        
        spec_power = spec ** 2
        
        if spec_power.sum() > 1e-10:
            centroid = np.sum(freqs * spec_power) / spec_power.sum()
            
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            bandwidth = np.sqrt(
                np.sum(((freqs - centroid) ** 2) * spec_power) / spec_power.sum()
            )
            
            flux = np.std(spec_power)
        else:
            centroid = 0
            rolloff = 0
            bandwidth = 0
            flux = 0
        
        features.append([centroid, rms, rolloff, zcr, flux, bandwidth])
    
    return np.array(features)

def cluster_segments(features: np.ndarray, n_clusters: int = None) -> np.ndarray:
    """Clustert Segmente basierend auf 6 Features"""
    
    if len(features) == 0:
        return np.array([])
    
    features_norm = features.copy()
    for i in range(features.shape[1]):
        col = features[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 0:
            features_norm[:, i] = (col - col_min) / (col_max - col_min)
    
    if n_clusters is None:
        centroids = features[:, 0]
        zcr_values = features[:, 3]
        
        centroid_range = centroids.max() - centroids.min()
        zcr_range = zcr_values.max() - zcr_values.min()
        
        if centroid_range > 3000 or zcr_range > 0.1:
            n_clusters = 3
        elif centroid_range > 1000 or zcr_range > 0.05:
            n_clusters = 2
        else:
            n_clusters = 2
    
    np.random.seed(42)
    n_samples = len(features_norm)
    
    if n_clusters >= n_samples:
        return np.arange(n_samples)
    
    centers = [features_norm[np.random.randint(n_samples)]]
    for _ in range(n_clusters - 1):
        distances = np.min([np.sum((features_norm - c)**2, axis=1) for c in centers], axis=0)
        probs = distances / distances.sum()
        centers.append(features_norm[np.random.choice(n_samples, p=probs)])
    centers = np.array(centers)
    
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(100):
        distances = np.sum((features_norm[:, None, :] - centers[None, :, :])**2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        
        if np.all(new_labels == labels):
            break
        
        labels = new_labels
        
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = features_norm[mask].mean(axis=0)
    
    centroid_means = [features[labels == k, 0].mean() for k in range(n_clusters)]
    sorted_order = np.argsort(centroid_means)
    label_map = {old: new for new, old in enumerate(sorted_order)}
    labels = np.array([label_map[l] for l in labels])
    
    return labels

# ============================================================================
# RECONSTRUCTION
# ============================================================================

def reconstruct_signals(y: np.ndarray, sr: int, segments: List[Tuple[float, float]], 
                        labels: np.ndarray, out_dir: str, fmt: str) -> Dict[str, str]:
    """Rekonstruiert Signale für jedes Label"""
    
    os.makedirs(out_dir, exist_ok=True)
    
    unique_labels = sorted(set(labels))
    exported = {}
    
    for label in unique_labels:
        name = chr(ord('A') + label)
        reconstructed = np.zeros_like(y)
        
        for (start_t, end_t), seg_label in zip(segments, labels):
            if seg_label == label:
                start_idx = int(start_t * sr)
                end_idx = int(end_t * sr)
                reconstructed[start_idx:end_idx] = y[start_idx:end_idx]
        
        out_path = os.path.join(out_dir, f"signal_{name}.{fmt}")
        save_audio(reconstructed, sr, out_path, fmt)
        exported[name] = out_path
    
    return exported

def fade_audio(a: np.ndarray, sr: int, fade_ms: float,
               b: Optional[np.ndarray] = None,
               curve: str = "linear") -> np.ndarray:
    """
    1) Wenn b=None: Edge-Fade auf 'a' (Fade-in + Fade-out) -> gegen Klicks an Rändern.
    2) Wenn b!=None: Crossfade-Join von 'a' und 'b' -> gegen Klicks beim Aneinanderfügen.
    """
    if fade_ms <= 0:
        return a if b is None else np.concatenate([a, b], axis=0)

    a_f = a.astype(np.float32, copy=True)
    fade_len = int(sr * fade_ms / 1000.0)
    fade_len = max(1, fade_len)

    # --- Edge-Fade ---
    if b is None:
        n = len(a_f)
        f = min(fade_len, n // 2)  # verhindert Überlappung bei sehr kurzen Segmenten
        if f <= 0:
            return a_f

        t = np.linspace(0.0, 1.0, f, endpoint=False, dtype=np.float32)
        if curve == "linear":
            fade_in = t
            fade_out = 1.0 - t
        elif curve in ("equal_power", "sin"):
            fade_in = np.sin(0.5 * np.pi * t)
            fade_out = np.cos(0.5 * np.pi * t)
        else:
            raise ValueError("curve must be 'linear' or 'equal_power'")

        if a_f.ndim == 2:
            fade_in = fade_in[:, None]
            fade_out = fade_out[:, None]

        a_f[:f] *= fade_in
        a_f[-f:] *= fade_out[::-1]
        return a_f

    # --- Crossfade-Join ---
    b_f = b.astype(np.float32, copy=False)
    f = min(fade_len, len(a_f), len(b_f))
    if f <= 0:
        return np.concatenate([a_f, b_f], axis=0)

    t = np.linspace(0.0, 1.0, f, endpoint=False, dtype=np.float32)
    if curve == "linear":
        fade_in = t
        fade_out = 1.0 - t
    elif curve in ("equal_power", "sin"):
        fade_in = np.sin(0.5 * np.pi * t)
        fade_out = np.cos(0.5 * np.pi * t)
    else:
        raise ValueError("curve must be 'linear' or 'equal_power'")

    if a_f.ndim == 2:
        fade_in = fade_in[:, None]
        fade_out = fade_out[:, None]

    overlap = a_f[-f:] * fade_out + b_f[:f] * fade_in
    return np.concatenate([a_f[:-f], overlap, b_f[f:]], axis=0)

def compact_from_masked(masked: np.ndarray, sr: int,
                        silence_threshold: float = 0.0,
                        crossfade_ms: float = 0.0,
                        fade_curve: str = "linear") -> np.ndarray:
    """
    Nimmt EIN rekonstruiertes Signal aus reconstruct_signals (A1-0-0-A2-0-0-...)
    und entfernt die 0-/Stille-Abschnitte -> A1-A2-A3-...
    Optional: Crossfade zwischen zusammengefügten Segmenten.
    """
    x = masked.astype(np.float32, copy=False)

    # Nicht-Stille Maske (mono/stereo)
    if x.ndim == 1:
        mag = np.abs(x)
    else:
        mag = np.max(np.abs(x), axis=1)

    idx = np.flatnonzero(mag > silence_threshold)
    if idx.size == 0:
        return np.array([], dtype=np.float32)

    # zusammenhängende Blöcke finden
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]] + 1

    # Chunks sammeln
    chunks = [x[s:e] for s, e in zip(starts, ends) if e > s]
    if not chunks:
        return np.array([], dtype=np.float32)

    # zusammenfügen (optional mit Crossfade)
    out = chunks[0].copy()
    for nxt in chunks[1:]:
        if crossfade_ms > 0:
            out = fade_audio(out, sr, crossfade_ms, b=nxt, curve=fade_curve)
        else:
            out = np.concatenate([out, nxt], axis=0)

    return out

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(y: np.ndarray, sr: int, cfar: bool, features: Dict, save_path, 
                change_score: np.ndarray, threshold: float,
                boundaries: List[float], segments: List[Tuple[float, float]], 
                labels: np.ndarray, max_seconds: float = None):
    """Visualisiert Ergebnisse - markiert ungültige Bereiche"""
    
    t = np.arange(len(y)) / sr
    
    if max_seconds:
        plot_mask = t <= max_seconds
        t = t[plot_mask]
        y = y[plot_mask]
    else:
        max_seconds = t[-1]
    
    fig, axes = plt.subplots(7, 1, figsize=(14, 14), sharex=True)
    
    # 1. Waveform mit Segmenten
    ax = axes[0]
    ax.plot(t, y, 'k-', linewidth=0.5, alpha=0.7)

    colors = plt.cm.Set3(np.linspace(0, 1, len(set(labels))))
    plotted_labels = set()

    for (start_t, end_t), label in zip(segments, labels):
        if start_t < max_seconds:
            show_label = label not in plotted_labels
            ax.axvspan(start_t, min(end_t, max_seconds), 
                      alpha=0.3, color=colors[label], 
                      label=f'Signal {chr(ord("A") + label)}' if show_label else '')
            if show_label:
                plotted_labels.add(label)
    
    # ⭐ NEU: Markiere ungültige Bereiche
    feat_t = features['times']
    valid_frames = features['valid_frames']
    for i in range(len(feat_t)-1):
        if not valid_frames[i] and feat_t[i] < max_seconds:
            ax.axvspan(feat_t[i], min(feat_t[i+1], max_seconds), 
                      alpha=0.2, color='red', linewidth=0)
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform (rot = ignorierte Bereiche unter MIN_AMPLITUDE_THRESHOLD)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Feature-Zeitachse
    if max_seconds:
        mask = feat_t <= max_seconds
        feat_t = feat_t[mask]
        valid_frames_plot = valid_frames[mask]
    else:
        valid_frames_plot = valid_frames
    
    # 2-6. Features (wie zuvor)
    feature_plots = [
        (1, 'centroid', 'b-', 'Frequency (Hz)', 'Spectral Centroid'),
        (2, 'rms', 'g-', 'RMS', 'RMS Energy'),
        (3, 'zcr', 'orange', 'ZCR', 'Zero-Crossing Rate'),
        (4, 'flux', 'cyan', 'Flux', 'Spectral Flux'),
        (5, 'bandwidth', 'magenta', 'Bandwidth (Hz)', 'Spectral Bandwidth')
    ]
    
    for idx, feat_name, color, ylabel, title in feature_plots:
        ax = axes[idx]
        feat_data = features[feat_name][mask] if max_seconds else features[feat_name]
        ax.plot(feat_t, feat_data, color, linewidth=1, alpha=0.7)
        
        # Markiere ungültige Bereiche
        for i in range(len(feat_t)-1):
            if not valid_frames_plot[i]:
                ax.axvspan(feat_t[i], feat_t[i+1], alpha=0.15, color='red', linewidth=0)
        
        for b in boundaries:
            if 0 < b < max_seconds:
                ax.axvline(b, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # 7. Combined Change Score
    ax = axes[6]
    if cfar:
        if max_seconds:
            change = change_score[mask]
            thresh = threshold[mask]
        else:
            change = change_score
            thresh = threshold
        
        ax.plot(feat_t, change, 'purple', linewidth=1, label='Change Score')
        ax.plot(feat_t, thresh, 'orange', linewidth=2, label='Adaptive CFAR Threshold')
    else:
        change = change_score[mask] if max_seconds else change_score
        ax.plot(feat_t, change, 'purple', linewidth=1)
        ax.axhline(threshold, color='orange', linestyle=':', linewidth=2, label='Threshold')
    
    # Markiere ungültige Bereiche
    for i in range(len(feat_t)-1):
        if not valid_frames_plot[i]:
            ax.axvspan(feat_t[i], feat_t[i+1], alpha=0.15, color='red', linewidth=0)
    
    for b in boundaries:
        if 0 < b < max_seconds:
            ax.axvline(b, color='r', linestyle='--', linewidth=1.5, alpha=0.7, 
                      label='Boundaries' if b == boundaries[1] else '')
    
    ax.set_ylabel('Change Score')
    ax.set_xlabel('Time (s)')
    ax.set_title('Combined Change Detection Score')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.perf_counter()
    cfg = Config()
    j = 0
    for a in audio_files:
        t1 = time.perf_counter()
        cfg.INPUT_FILE = a
        base_name = os.path.splitext(os.path.basename(cfg.INPUT_FILE))[0]
        cfg.OUT_DIR = "out/out_v001/" + str(base_name) + "/"        

        print("="*60)
        print("Interleaved Audio Segmentation & Reconstruction v5.8")
        print("MIT AMPLITUDENSCHWELLE")
        print("="*60)
        
        # Load
        print(f"\n[1/6] Loading: {cfg.INPUT_FILE}")
        y, sr = load_audio(cfg.INPUT_FILE)
        duration = len(y) / sr
        print(f"  → Sample rate: {sr} Hz")
        print(f"  → Duration: {duration:.2f} s")
        print(f"  → Samples: {len(y)}")
        
        # Features
        print(f"\n[2/6] Computing features...")
        print(f"  → Window: {cfg.WINDOW_MS} ms")
        print(f"  → Hop: {cfg.HOP_MS} ms")
        print(f"  ⭐ MIN_AMPLITUDE_THRESHOLD: {cfg.MIN_AMPLITUDE_THRESHOLD}")
        features = compute_stft_features(y, sr, cfg.WINDOW_MS, cfg.HOP_MS, 
                                        cfg.MIN_AMPLITUDE_THRESHOLD)
        n_valid = np.sum(features['valid_frames'])
        n_total = len(features['valid_frames'])
        print(f"  → Frames: {n_total} ({n_valid} gültig, {n_total-n_valid} ignoriert)")
        
        # Change Detection
        if cfg.CHANGE_DETECTION_CFAR:
            print(f"\n[3/6] Detecting changes (CFAR)...")
            change_score, threshold = detect_changes_cfar(
                cfg.WEIGHT_CENTROID, cfg.WEIGHT_RMS, cfg.WEIGHT_Rolloff,
                cfg.WEIGHT_ZCR, cfg.WEIGHT_FLUX, cfg.WEIGHT_BANDWIDTH,
                features, cfg)
            
            min_segment_s = cfg.MIN_SEGMENT_MS / 1000.0
            boundaries = find_boundaries_cfar(change_score, threshold, features['times'], 
                                            min_segment_s, features['valid_frames'])
            print(f"  → Boundaries found: {len(boundaries)-2}")
        else:
            print(f"\n[3/6] Detecting changes...")
            change_score, threshold = detect_changes(
                cfg.WEIGHT_CENTROID, cfg.WEIGHT_RMS, cfg.WEIGHT_Rolloff,
                cfg.WEIGHT_ZCR, cfg.WEIGHT_FLUX, cfg.WEIGHT_BANDWIDTH,
                features, cfg.CHANGE_THRESHOLD_PERCENTILE)
            print(f"  → Threshold: {threshold:.4f}")
            
            min_segment_s = cfg.MIN_SEGMENT_MS / 1000.0
            boundaries = find_boundaries(change_score, threshold, features['times'], 
                                        min_segment_s, features['valid_frames'])
            print(f"  → Boundaries found: {len(boundaries)-2}")
        
        # Segmentation
        print(f"\n[4/6] Creating segments...")
        segments = create_segments(
            boundaries,
            features,
            deadzone_ms=cfg.DEADZONE_MS,
            min_segment_ms=cfg.MIN_SEGMENT_MS,
            verbose=cfg.VERBOSE
        )
        print(f"  → Segments: {len(segments)}")
        
        segment_features = extract_segment_features(y, sr, segments, 
                                                   cfg.MIN_AMPLITUDE_THRESHOLD)
        
        # Clustering
        print(f"\n[5/6] Clustering segments...")
        labels = cluster_segments(segment_features, cfg.NUM_CLUSTERS)
        n_signals = len(set(labels))
        print(f"  → Number of signals detected: {n_signals}")
        
        for i in range(n_signals):
            count = np.sum(labels == i)
            mean_freq = segment_features[labels == i, 0].mean()
            mean_rms = segment_features[labels == i, 1].mean()
            print(f"  → Signal {chr(ord('A')+i)}: {count} segments, ~{mean_freq:.0f} Hz, RMS={mean_rms:.3f}")
        
        # Reconstruction
        print(f"\n[6/6] Reconstructing signals...")
        exported = reconstruct_signals(y, sr, segments, labels, cfg.OUT_DIR, cfg.EXPORT_FORMAT, cfg.FADE_IN)
        
        for name, path in sorted(exported.items()):
            print(f"  → {name}: {path}")
        
        t2 = time.perf_counter()

        # Visualization
        print(f"\n[PLOT] Generating visualization...")
        save_path_plt = os.path.join(cfg.OUT_DIR, f"plt_{base_name}.png")
        plot_results(y, sr, cfg.CHANGE_DETECTION_CFAR, features, save_path_plt, 
                    change_score, threshold, boundaries, segments, labels, max_seconds=3.0)
        
        t3 = time.perf_counter()

        t_run = t3 - t1
        t_plot = t3 - t2
        t_cal = t2 - t1

        print(f"\n{'='*60}")
        print(f"Run time: {t_run:.2f}s")
        print(f"Plot time: {t_plot:.2f}s")
        print(f"Calculation time: {t_cal:.2f}s")
        print(f"{'='*60}\n")
        
        j += 1

    print(f"\n{'='*60}")
    print("Done! 🎉")
    print(f"{'='*60}\n")

    t_all = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"All time: {t_all:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    audio_files = [
    
    #SINUS + Musik
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_1k_8k_vio_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_vio_8k_drum_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_vio_8k_jing_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_vio_jingle_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_1k_GOD_30sec_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_pod_1k_30sec_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Raw_Signals/violin.mp3",
    r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_pod_1k_60sec_rand.mp3",
    
    # SINUS + WHITE/0
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_white_1k_8k_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_silence_1k_8k_rand.mp3",

    # NUR SINUS
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_30_1k_8k_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_100_1k_8k_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_200_1k_8k_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_500_1k_8k_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_700_1k_8k_rand.mp3",
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_1k_8k_20k_rand.mp3",

    #...
    ]
    main()