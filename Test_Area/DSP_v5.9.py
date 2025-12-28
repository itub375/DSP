"""
Audio Segmentation v5.8 - NUR FLUX-BASIERT
Änderungen:
- Alle Features außer Flux werden auf 0 gewichtet
- Segmentation erfolgt ausschließlich anhand von Spectral Flux
- MIN_AMPLITUDE_THRESHOLD weiterhin aktiv
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pydub import AudioSegment
import time
import csv

@dataclass
class Config:
    # I/O
    INPUT_FILE: str = r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_1k_GOD_30sec_rand.mp3"
    OUT_DIR: str = "output_segments"
    
    # Analyse-Parameter
    WINDOW_MS: float = 2.0
    HOP_MS: float = 0.5
    
    # Minimale Amplitude für gültige Frames
    MIN_AMPLITUDE_THRESHOLD: float = 0.0
    
    CHANGE_DETECTION_CFAR: bool = False

    # Change Detection
    CHANGE_THRESHOLD_PERCENTILE: float = 95.0
    MIN_SEGMENT_MS: float = 5.0
    MERGE_TOLERANCE_MS: float = 5.0

    # CFAR Change Detection
    CFAR_METHOD: str = "OS"
    CFAR_GUARD_CELLS: int = 3
    CFAR_TRAIN_CELLS: int = 15
    CFAR_ALPHA: float = 2.0
    CFAR_K_FRACTION: float = 0.75
    
    # Clustering
    NUM_CLUSTERS: int = None
    
    # Export
    EXPORT_FORMAT: str = "mp3"
    VERBOSE: bool = True

    # ⭐ NUR FLUX WIRD VERWENDET
    WEIGHT_CENTROID: int    = 0.0      # 0 = ignoriert
    WEIGHT_RMS:      int    = 0.0      # 0 = ignoriert
    WEIGHT_Rolloff:  int    = 0.0      # 0 = ignoriert
    WEIGHT_ZCR:      int    = 0.0      # 0 = ignoriert
    WEIGHT_FLUX:     int    = 1.0      # NUR FLUX!
    WEIGHT_BANDWIDTH:int    = 0.0      # 0 = ignoriert
    
    # Dead-Zone um Wechselstellen
    DEADZONE_MS: float = 0

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
# FEATURE EXTRACTION - NUR FLUX (+ RMS für Amplitude-Check)
# ============================================================================

def compute_stft_features(y: np.ndarray, sr: int, window_ms: float, hop_ms: float, 
                          min_amplitude: float = 0.0):
    """
    Berechnet NUR Spectral Flux (und RMS für Amplitude-Checking)
    Alle anderen Features werden auf 0 gesetzt für Konsistenz
    """
    
    window_samples = int(window_ms * sr / 1000)
    hop_samples = int(hop_ms * sr / 1000)
    
    if window_samples % 2 != 0:
        window_samples += 1
    
    window = np.hanning(window_samples)
    n_frames = 1 + (len(y) - window_samples) // hop_samples
    
    # Nur Flux und RMS werden berechnet
    flux_values = np.zeros(n_frames)
    rms_values = np.zeros(n_frames)
    valid_frames = np.ones(n_frames, dtype=bool)
    
    # Dummy-Arrays für Kompatibilität (werden nicht verwendet)
    centroids = np.zeros(n_frames)
    rolloffs = np.zeros(n_frames)
    zcr_values = np.zeros(n_frames)
    bandwidth_values = np.zeros(n_frames)
    
    prev_spec = None
    
    for i in range(n_frames):
        start = i * hop_samples
        frame = y[start:start + window_samples]
        
        # Amplitude-Check
        frame_rms = np.sqrt(np.mean(frame**2))
        
        if frame_rms < min_amplitude:
            valid_frames[i] = False
            rms_values[i] = frame_rms
            continue
        
        rms_values[i] = frame_rms
        
        # Berechne nur Spectral Flux
        windowed_frame = frame * window
        spec = np.abs(np.fft.rfft(windowed_frame))
        
        if prev_spec is not None:
            flux_values[i] = np.sum((spec - prev_spec) ** 2)
        
        prev_spec = spec.copy()
    
    times = np.arange(n_frames) * hop_samples / sr
    
    return {
        'times': times,
        'centroid': centroids,      # Dummy
        'rms': rms_values,
        'rolloff': rolloffs,        # Dummy
        'zcr': zcr_values,          # Dummy
        'flux': flux_values,        # ⭐ HAUPTFEATURE
        'bandwidth': bandwidth_values,  # Dummy
        'hop_samples': hop_samples,
        'valid_frames': valid_frames
    }

# ============================================================================
# CHANGE DETECTION - NUR FLUX
# ============================================================================

def detect_changes(wflux: float, features: Dict, threshold_percentile: float) -> np.ndarray:
    """
    Erkennt Änderungspunkte direkt anhand von Spectral Flux PEAKS
    """
    
    flux = features['flux']
    valid_frames = features['valid_frames']
    
    def normalize(x, valid_mask):
        x = np.copy(x)
        valid_values = x[valid_mask]
        if len(valid_values) > 0:
            x_min, x_max = np.percentile(valid_values, [5, 95])
            if x_max - x_min > 0:
                x = (x - x_min) / (x_max - x_min)
        return np.clip(x, 0, 1)
    
    flux_norm = normalize(flux, valid_frames)
    
    # ⭐ DIREKT FLUX verwenden (keine Differenz!)
    combined_change = flux_norm.copy()
    
    # Ungültige Frames auf 0 setzen
    combined_change[~valid_frames] = 0.0
    
    # Leichte Glättung für stabilere Peaks
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    combined_change = np.convolve(combined_change, kernel, mode='same')
    
    # Threshold aus gültigen Frames
    valid_changes = combined_change[valid_frames]
    if len(valid_changes) > 0:
        threshold = np.percentile(valid_changes, threshold_percentile)
    else:
        threshold = 0.0
    
    return combined_change, threshold

def find_boundaries(change_score: np.ndarray, threshold: float, 
                   times: np.ndarray, min_segment_s: float,
                   valid_frames: np.ndarray) -> List[float]:
    """Findet Segment-Grenzen"""
    
    candidates = []
    for i in range(1, len(change_score)-1):
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
# CHANGE DETECTION mit CFAR - NUR FLUX
# ============================================================================

def detect_changes_cfar(wflux: float, features: Dict, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erkennt Änderungspunkte mit CFAR - direkt auf Spectral Flux PEAKS
    """
    
    flux = features['flux']
    valid_frames = features['valid_frames']
    
    def normalize(x, valid_mask):
        x = np.copy(x)
        valid_values = x[valid_mask]
        if len(valid_values) > 0:
            x_min, x_max = np.percentile(valid_values, [5, 95])
            if x_max - x_min > 0:
                x = (x - x_min) / (x_max - x_min)
        return np.clip(x, 0, 1)
    
    flux_norm = normalize(flux, valid_frames)

    # ⭐ DIREKT FLUX verwenden (keine Differenz!)
    combined_change = flux_norm.copy()
    
    # Ungültige Frames auf 0 setzen
    combined_change[~valid_frames] = 0.0
    
    # Leichte Glättung
    kernel_size = 3
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
    """Findet Boundaries mit adaptivem Threshold"""
    
    candidates = []
    for i in range(1, len(change_score)-1):
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
# CSV EXPORT - WECHSELSTELLEN (BOUNDARIES)
# ============================================================================

def save_change_points_csv(out_dir: str,
                           base_name: str,
                           boundaries: List[float],
                           features: Dict,
                           change_score: np.ndarray,
                           threshold,
                           cfg: Config) -> str:
    """
    Speichert die gefundenen Wechselstellen (Boundaries ohne Start/Ende) als CSV.
    Zusätzlich werden Score und Threshold an der jeweiligen Stelle mit abgelegt.
   """
    os.makedirs(out_dir, exist_ok=True)

    times = features["times"]
    if len(boundaries) < 2:
        inner = []
    else:
        inner = boundaries[1:-1]  # ohne 0.0 und times[-1]

    csv_path = os.path.join(out_dir, f"wechselstellen_{base_name}_FLUX_ONLY.csv")

    is_cfar = isinstance(threshold, np.ndarray)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([
            "file",
            "cfar",
            "boundary_idx",
            "time_s",
            "time_ms",
            "score",
            "threshold"
        ])

        for k, t_b in enumerate(inner):
           # nächster Frame-Index zur Boundary-Zeit
            i = int(np.searchsorted(times, t_b))
            i = max(0, min(i, len(times) - 1))

            score = float(change_score[i]) if len(change_score) else 0.0
            thr = float(threshold[i]) if is_cfar else float(threshold)

            w.writerow([
                base_name,
                int(is_cfar),
                k,
                f"{t_b:.6f}",
                f"{t_b*1000.0:.3f}",
                f"{score:.6f}",
                f"{thr:.6f}",
            ])

    return csv_path


# ============================================================================
# SEGMENTATION & CLUSTERING
# ============================================================================

def create_segments(boundaries: List[float], features: Dict,
                    deadzone_ms: float = 0.0,
                    min_segment_ms: float = 0.0,
                    verbose: bool = False) -> List[Tuple[float, float]]:
    """Erstellt Segment-Paare"""
    dz = deadzone_ms / 1000.0
    segments: List[Tuple[float, float]] = []
    
    times = features['times']
    valid_frames = features['valid_frames']

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
        
        # Prüfe Gültigkeitsrate
        start_idx = np.searchsorted(times, start)
        end_idx = np.searchsorted(times, end)
        segment_valid = valid_frames[start_idx:end_idx]
        
        if len(segment_valid) > 0:
            valid_ratio = np.sum(segment_valid) / len(segment_valid)
            if valid_ratio < 0.3:
                if verbose:
                    print(f"  [skip] Segment {i}: nur {valid_ratio*100:.1f}% gültige Frames")
                continue

        segments.append((start, end))

    return segments

def extract_segment_features(y: np.ndarray, sr: int, 
                            segments: List[Tuple[float, float]],
                            min_amplitude: float = 0.0) -> np.ndarray:
    """
    Extrahiert Features für jedes Segment
    NUR FLUX wird für Clustering verwendet (+ RMS für Validierung)
    """
    
    features = []
    
    for start_t, end_t in segments:
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        segment = y[start_idx:end_idx]
        
        if len(segment) < 10:
            features.append([0, 0])  # [flux, rms]
            continue
        
        segment_rms = np.sqrt(np.mean(segment**2))
        if segment_rms < min_amplitude:
            features.append([0, 0])
            continue
        
        # Berechne nur Flux für dieses Segment
        window = np.hanning(len(segment))
        spec = np.abs(np.fft.rfft(segment * window))
        flux = np.std(spec)  # Vereinfachte Flux-Berechnung
        
        features.append([flux, segment_rms])
    
    return np.array(features)

def cluster_segments(features: np.ndarray, n_clusters: int = None) -> np.ndarray:
    """
    Clustert Segmente NUR basierend auf Flux
    """
    
    if len(features) == 0:
        return np.array([])
    
    # Normalisierung
    features_norm = features.copy()
    for i in range(features.shape[1]):
        col = features[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 0:
            features_norm[:, i] = (col - col_min) / (col_max - col_min)
    
    # Auto-bestimme Cluster-Anzahl basierend auf Flux-Range
    if n_clusters is None:
        flux_values = features[:, 0]
        flux_range = flux_values.max() - flux_values.min()
        
        if flux_range > 0.5:
            n_clusters = 3
        elif flux_range > 0.2:
            n_clusters = 2
        else:
            n_clusters = 2
    
    np.random.seed(42)
    n_samples = len(features_norm)
    
    if n_clusters >= n_samples:
        return np.arange(n_samples)
    
    # K-Means++ Initialisierung
    centers = [features_norm[np.random.randint(n_samples)]]
    for _ in range(n_clusters - 1):
        distances = np.min([np.sum((features_norm - c)**2, axis=1) for c in centers], axis=0)
        probs = distances / distances.sum()
        centers.append(features_norm[np.random.choice(n_samples, p=probs)])
    centers = np.array(centers)
    
    # K-Means Iteration
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
    
    # Sortiere nach Flux-Wert
    flux_means = [features[labels == k, 0].mean() for k in range(n_clusters)]
    sorted_order = np.argsort(flux_means)
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

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(y: np.ndarray, sr: int, cfar: bool, features: Dict, save_path, 
                change_score: np.ndarray, threshold: float,
                boundaries: List[float], segments: List[Tuple[float, float]], 
                labels: np.ndarray, max_seconds: float = None):
    """Visualisiert Ergebnisse - fokussiert auf FLUX"""
    
    t = np.arange(len(y)) / sr
    
    if max_seconds:
        plot_mask = t <= max_seconds
        t = t[plot_mask]
        y = y[plot_mask]
    else:
        max_seconds = t[-1]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
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
    
    # Markiere ungültige Bereiche
    feat_t = features['times']
    valid_frames = features['valid_frames']
    for i in range(len(feat_t)-1):
        if not valid_frames[i] and feat_t[i] < max_seconds:
            ax.axvspan(feat_t[i], min(feat_t[i+1], max_seconds), 
                      alpha=0.2, color='red', linewidth=0)
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform + Segmentation (NUR FLUX-basiert)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Feature-Zeitachse
    if max_seconds:
        mask = feat_t <= max_seconds
        feat_t = feat_t[mask]
        valid_frames_plot = valid_frames[mask]
    else:
        valid_frames_plot = valid_frames
    
    # 2. RMS Energy (für Kontext)
    ax = axes[1]
    rms_data = features['rms'][mask] if max_seconds else features['rms']
    ax.plot(feat_t, rms_data, 'g-', linewidth=1, alpha=0.7)
    
    for i in range(len(feat_t)-1):
        if not valid_frames_plot[i]:
            ax.axvspan(feat_t[i], feat_t[i+1], alpha=0.15, color='red', linewidth=0)
    
    for b in boundaries:
        if 0 < b < max_seconds:
            ax.axvline(b, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('RMS')
    ax.set_title('RMS Energy (Kontext)')
    ax.grid(True, alpha=0.3)
    
    # 3. Spectral Flux (HAUPTFEATURE)
    ax = axes[2]
    flux_data = features['flux'][mask] if max_seconds else features['flux']
    ax.plot(feat_t, flux_data, 'cyan', linewidth=1.5, alpha=0.8)
    
    for i in range(len(feat_t)-1):
        if not valid_frames_plot[i]:
            ax.axvspan(feat_t[i], feat_t[i+1], alpha=0.15, color='red', linewidth=0)
    
    for b in boundaries:
        if 0 < b < max_seconds:
            ax.axvline(b, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Flux')
    ax.set_title('⭐ Spectral Flux (HAUPTFEATURE für Segmentation)')
    ax.grid(True, alpha=0.3)
    
    # 4. Flux-basierter Detection Score (= normalisierter Flux)
    ax = axes[3]
    if cfar:
        if max_seconds:
            change = change_score[mask]
            thresh = threshold[mask]
        else:
            change = change_score
            thresh = threshold
        
        ax.plot(feat_t, change, 'purple', linewidth=1.5, label='Flux Score (normalisiert)')
        ax.plot(feat_t, thresh, 'orange', linewidth=2, label='Adaptive CFAR Threshold')
    else:
        change = change_score[mask] if max_seconds else change_score
        ax.plot(feat_t, change, 'purple', linewidth=1.5, label='Flux Score (normalisiert)')
        ax.axhline(threshold, color='orange', linestyle=':', linewidth=2, label='Threshold')
    
    for i in range(len(feat_t)-1):
        if not valid_frames_plot[i]:
            ax.axvspan(feat_t[i], feat_t[i+1], alpha=0.15, color='red', linewidth=0)
    
    for b in boundaries:
        if 0 < b < max_seconds:
            ax.axvline(b, color='r', linestyle='--', linewidth=1.5, alpha=0.7, 
                      label='Boundaries' if b == boundaries[1] else '')
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Time (s)')
    ax.set_title('Peak Detection auf Flux-Werten')
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

    for idx, audio_file in enumerate(audio_files):
        t1 = time.perf_counter()
        cfg.INPUT_FILE = audio_file
        base_name = os.path.splitext(os.path.basename(cfg.INPUT_FILE))[0]
        cfg.OUT_DIR = "out/out_v5.9_FLUX_ONLY/" + str(base_name) + "/"  

        print("="*60)
        print("Interleaved Audio Segmentation v5.9 - NUR FLUX")
        print("="*60)
        
        # Load
        print(f"\n[1/6] Loading: {cfg.INPUT_FILE}")
        y, sr = load_audio(cfg.INPUT_FILE)
        duration = len(y) / sr
        print(f"  → Sample rate: {sr} Hz")
        print(f"  → Duration: {duration:.2f} s")
        print(f"  → Samples: {len(y)}")
        
        # Features
        print(f"\n[2/6] Computing features (NUR FLUX)...")
        print(f"  → Window: {cfg.WINDOW_MS} ms")
        print(f"  → Hop: {cfg.HOP_MS} ms")
        print(f"  → MIN_AMPLITUDE_THRESHOLD: {cfg.MIN_AMPLITUDE_THRESHOLD}")
        features = compute_stft_features(y, sr, cfg.WINDOW_MS, cfg.HOP_MS, 
                                        cfg.MIN_AMPLITUDE_THRESHOLD)
        n_valid = np.sum(features['valid_frames'])
        n_total = len(features['valid_frames'])
        print(f"  → Frames: {n_total} ({n_valid} gültig, {n_total-n_valid} ignoriert)")
        
        # Change Detection
        if cfg.CHANGE_DETECTION_CFAR:
            print(f"\n[3/6] Detecting changes (CFAR - NUR FLUX)...")
            change_score, threshold = detect_changes_cfar(cfg.WEIGHT_FLUX, features, cfg)
            
            min_segment_s = cfg.MIN_SEGMENT_MS / 1000.0
            boundaries = find_boundaries_cfar(change_score, threshold, features['times'], 
                                            min_segment_s, features['valid_frames'])
            print(f"  → Boundaries found: {len(boundaries)-2}")
        else:
            print(f"\n[3/6] Detecting changes (NUR FLUX)...")
            change_score, threshold = detect_changes(cfg.WEIGHT_FLUX, features, 
                                                     cfg.CHANGE_THRESHOLD_PERCENTILE)
            print(f"  → Threshold: {threshold:.4f}")
            
            min_segment_s = cfg.MIN_SEGMENT_MS / 1000.0
            boundaries = find_boundaries(change_score, threshold, features['times'], 
                                        min_segment_s, features['valid_frames'])
            print(f"  → Boundaries found: {len(boundaries)-2}")
       
        # CSV Export: Wechselstellen speichern (zum Vergleichen)
        csv_path = save_change_points_csv(cfg.OUT_DIR, base_name, boundaries, features, change_score, threshold, cfg        )
        print(f"\n[CSV] Wechselstellen gespeichert: {csv_path}")
        
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
        print(f"\n[5/6] Clustering segments (NUR FLUX)...")
        labels = cluster_segments(segment_features, cfg.NUM_CLUSTERS)
        n_signals = len(set(labels))
        print(f"  → Number of signals detected: {n_signals}")
        
        for i in range(n_signals):
            count = np.sum(labels == i)
            mean_flux = segment_features[labels == i, 0].mean()
            mean_rms = segment_features[labels == i, 1].mean()
            print(f"  → Signal {chr(ord('A')+i)}: {count} segments, Flux={mean_flux:.3f}, RMS={mean_rms:.3f}")
        
        # Reconstruction
        print(f"\n[6/6] Reconstructing signals...")
        exported = reconstruct_signals(y, sr, segments, labels, cfg.OUT_DIR, cfg.EXPORT_FORMAT)
        
        for name, path in sorted(exported.items()):
            print(f"  → {name}: {path}")
        
        t2 = time.perf_counter()

        # Visualization
        print(f"\n[PLOT] Generating visualization...")
        save_path_plt = os.path.join(cfg.OUT_DIR, f"plt_{base_name}_FLUX_ONLY.png")
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
    #r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_pod_1k_60sec_rand.mp3",
    r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/50ms/interleaved_pod_1k_60sec_50ms.mp3",
    
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