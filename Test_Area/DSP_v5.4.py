"""
Audio Segmentation mit adaptivem CFAR-Threshold

Änderungen zu v5.3:
- CFAR (Constant False Alarm Rate) für adaptive Threshold-Berechnung
- Mehrere CFAR-Varianten: CA-CFAR, OS-CFAR, SO-CFAR
- Robustere Boundary-Detection

Neu:

    Config    : Mehrere Variabeln zum anpassen des CFAR algorythmusses
    detect_changes_cfar : Neue methode zum Berechnen der übergänge mit CFAR
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pydub import AudioSegment
import time

@dataclass
class Config:
    # I/O
    INPUT_FILE: str = r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/rand/interleaved_30_1k_8k_rand.mp3"
    OUT_DIR: str = "output_segments"
    
    # Analyse-Parameter
    WINDOW_MS: float = 2.0          # Fenster für Feature-Berechnung (ms)
    HOP_MS: float = 0.5             # Hop-Größe (ms)
    
    # CFAR Change Detection
    CFAR_METHOD: str = "OS"         # "CA", "OS", "SO", "PERCENTILE"
    CFAR_GUARD_CELLS: int = 3       # Schutzzellen um Test-Zelle
    CFAR_TRAIN_CELLS: int = 15      # Trainingszellen für Background
    CFAR_ALPHA: float = 3.5         # Schwellenfaktor (höher = weniger detections)
    CFAR_K_FRACTION: float = 0.75   # Für OS-CFAR: k/N ratio
    
    MIN_SEGMENT_MS: float = 7.0     # Minimale Segmentlänge (ms)
    MERGE_TOLERANCE_MS: float = 5.0 # Toleranz für Segment-Merge
    
    # Clustering
    NUM_CLUSTERS: int = None        # None = automatisch bestimmen
    
    # Export
    EXPORT_FORMAT: str = "mp3"
    VERBOSE: bool = True

# ============================================================================
# AUDIO I/O (unverändert)
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
# FEATURE EXTRACTION (unverändert)
# ============================================================================

def compute_stft_features(y: np.ndarray, sr: int, window_ms: float, hop_ms: float):
    """Berechnet Spektral-Features mit STFT"""
    
    window_samples = int(window_ms * sr / 1000)
    hop_samples = int(hop_ms * sr / 1000)
    
    if window_samples % 2 != 0:
        window_samples += 1
    
    window = np.hanning(window_samples)
    n_frames = 1 + (len(y) - window_samples) // hop_samples
    
    centroids = np.zeros(n_frames)
    rms_values = np.zeros(n_frames)
    rolloffs = np.zeros(n_frames)
    
    freqs = np.fft.rfftfreq(window_samples, 1/sr)
    
    for i in range(n_frames):
        start = i * hop_samples
        frame = y[start:start + window_samples] * window
        
        rms_values[i] = np.sqrt(np.mean(frame**2))
        
        spec = np.abs(np.fft.rfft(frame))
        spec_power = spec ** 2
        
        if spec_power.sum() > 1e-10:
            centroids[i] = np.sum(freqs * spec_power) / spec_power.sum()
            
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                rolloffs[i] = freqs[rolloff_idx[0]]
    
    times = np.arange(n_frames) * hop_samples / sr
    
    return {
        'times': times,
        'centroid': centroids,
        'rms': rms_values,
        'rolloff': rolloffs,
        'hop_samples': hop_samples
    }

# ============================================================================
# CFAR IMPLEMENTATIONS
# ============================================================================

def cfar_1d(signal: np.ndarray, guard: int, train: int, alpha: float, 
            method: str = "CA", k_fraction: float = 0.75) -> np.ndarray:
    """
    1D CFAR (Constant False Alarm Rate) Detector
    
    Args:
        signal: Input signal (change scores)
        guard: Number of guard cells on each side
        train: Number of training cells on each side
        alpha: Threshold multiplier
        method: "CA" (Cell-Averaging), "OS" (Order-Statistic), "SO" (Smallest-Of)
        k_fraction: For OS-CFAR: which order statistic to use (0-1)
    
    Returns:
        threshold: Adaptive threshold for each sample
    """
    
    n = len(signal)
    threshold = np.zeros(n)
    
    # Total window size
    window_half = guard + train
    
    for i in range(n):
        # Left training cells
        left_start = max(0, i - window_half)
        left_end = max(0, i - guard)
        
        # Right training cells  
        right_start = min(n, i + guard + 1)
        right_end = min(n, i + window_half + 1)
        
        # Collect training samples
        left_cells = signal[left_start:left_end]
        right_cells = signal[right_start:right_end]
        train_cells = np.concatenate([left_cells, right_cells])
        
        if len(train_cells) == 0:
            threshold[i] = 0
            continue
        
        # Berechne Threshold basierend auf Methode
        if method == "CA":  # Cell-Averaging CFAR
            noise_level = np.mean(train_cells)
            
        elif method == "OS":  # Order-Statistic CFAR
            k = int(len(train_cells) * k_fraction)
            k = max(0, min(k, len(train_cells) - 1))
            sorted_cells = np.sort(train_cells)
            noise_level = sorted_cells[k]
            
        elif method == "SO":  # Smallest-Of CFAR
            left_avg = np.mean(left_cells) if len(left_cells) > 0 else 0
            right_avg = np.mean(right_cells) if len(right_cells) > 0 else 0
            noise_level = min(left_avg, right_avg) if left_avg > 0 and right_avg > 0 else max(left_avg, right_avg)
            
        else:
            raise ValueError(f"Unknown CFAR method: {method}")
        
        threshold[i] = alpha * noise_level
    
    return threshold

# ============================================================================
# CHANGE DETECTION mit CFAR
# ============================================================================

def detect_changes_cfar(features: Dict, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erkennt Änderungspunkte mit CFAR-Threshold
    """
    
    centroid = features['centroid']
    rms = features['rms']
    rolloff = features['rolloff']
    
    # Normalisiere Features
    def normalize(x):
        x = np.copy(x)
        x_min, x_max = np.percentile(x, [5, 95])
        if x_max - x_min > 0:
            x = (x - x_min) / (x_max - x_min)
        return np.clip(x, 0, 1)
    
    cent_norm = normalize(centroid)
    rms_norm = normalize(rms)
    roll_norm = normalize(rolloff)
    
    # Berechne Änderungen
    cent_change = np.abs(np.diff(cent_norm, prepend=cent_norm[0]))
    rms_change = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    roll_change = np.abs(np.diff(roll_norm, prepend=roll_norm[0]))
    
    # Kombiniere mit Gewichtung
    combined_change = (
        2.0 * cent_change +
        1.0 * rms_change +
        1.5 * roll_change
    ) / 4.5
    
    # Smooth
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    combined_change = np.convolve(combined_change, kernel, mode='same')
    
    # CFAR oder Percentile
    if cfg.CFAR_METHOD == "PERCENTILE":
        # Fallback auf alte Methode
        threshold_val = np.percentile(combined_change, 85.0)
        threshold = np.full_like(combined_change, threshold_val)
    else:
        # Adaptive CFAR
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
                        times: np.ndarray, min_segment_s: float) -> List[float]:
    """
    Findet Boundaries mit adaptivem Threshold
    """
    
    # Finde Peaks über lokalem Threshold
    candidates = []
    for i in range(1, len(change_score)-1):
        if (change_score[i] > threshold[i] and 
            change_score[i] >= change_score[i-1] and 
            change_score[i] >= change_score[i+1]):
            candidates.append(times[i])
    
    if len(candidates) < 2:
        return [0.0, times[-1]]
    
    # Entferne zu nahe Grenzen
    boundaries = [0.0]
    for t in candidates:
        if t - boundaries[-1] >= min_segment_s:
            boundaries.append(t)
    
    boundaries.append(times[-1])
    
    return boundaries

# ============================================================================
# SEGMENTATION & CLUSTERING (unverändert)
# ============================================================================

def create_segments(boundaries: List[float]) -> List[Tuple[float, float]]:
    """Erstellt Segment-Paare aus Boundaries"""
    return [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

def extract_segment_features(y: np.ndarray, sr: int, 
                            segments: List[Tuple[float, float]]) -> np.ndarray:
    """Extrahiert Features für jedes Segment"""
    
    features = []
    
    for start_t, end_t in segments:
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        segment = y[start_idx:end_idx]
        
        if len(segment) < 10:
            features.append([0, 0, 0])
            continue
        
        window = np.hanning(len(segment))
        spec = np.abs(np.fft.rfft(segment * window))
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        
        spec_power = spec ** 2
        rms = np.sqrt(np.mean(segment**2))
        
        if spec_power.sum() > 1e-10:
            centroid = np.sum(freqs * spec_power) / spec_power.sum()
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            centroid = 0
            rolloff = 0
        
        features.append([centroid, rms, rolloff])
    
    return np.array(features)

def cluster_segments(features: np.ndarray, n_clusters: int = None) -> np.ndarray:
    """Clustert Segmente mit K-Means"""
    
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
        centroid_range = centroids.max() - centroids.min()
        
        if centroid_range > 3000:
            n_clusters = 3
        elif centroid_range > 1000:
            n_clusters = 2
        else:
            n_clusters = 2
    
    np.random.seed(42)
    n_samples = len(features_norm)
    
    if n_clusters >= n_samples:
        return np.arange(n_samples)
    
    # K-Means++
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
# RECONSTRUCTION (unverändert)
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

def plot_results(y: np.ndarray, sr: int, features: Dict, 
                change_score: np.ndarray, threshold: np.ndarray,
                boundaries: List[float], out_dir: str, 
                segments: List[Tuple[float, float]], 
                labels: np.ndarray, max_seconds: float = None):
    """Visualisiert Ergebnisse mit adaptivem Threshold"""
    
    t = np.arange(len(y)) / sr
    
    if max_seconds:
        plot_mask = t <= max_seconds
        t = t[plot_mask]
        y = y[plot_mask]
    else:
        max_seconds = t[-1]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # 1. Waveform
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
    
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform mit erkannten Segmenten')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Spectral Centroid
    ax = axes[1]
    feat_t = features['times']
    if max_seconds:
        mask = feat_t <= max_seconds
        feat_t = feat_t[mask]
        cent = features['centroid'][mask]
    else:
        cent = features['centroid']
    
    ax.plot(feat_t, cent, 'b-', linewidth=1)
    for b in boundaries:
        if 0 < b < max_seconds:
            ax.axvline(b, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectral Centroid')
    ax.grid(True, alpha=0.3)
    
    # 3. RMS
    ax = axes[2]
    if max_seconds:
        rms = features['rms'][mask]
    else:
        rms = features['rms']
    
    ax.plot(feat_t, rms, 'g-', linewidth=1)
    for b in boundaries:
        if 0 < b < max_seconds:
            ax.axvline(b, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('RMS')
    ax.set_title('RMS Energy')
    ax.grid(True, alpha=0.3)
    
    # 4. Change Score mit adaptivem Threshold
    ax = axes[3]
    if max_seconds:
        change = change_score[mask]
        thresh = threshold[mask]
    else:
        change = change_score
        thresh = threshold
    
    ax.plot(feat_t, change, 'purple', linewidth=1, label='Change Score')
    ax.plot(feat_t, thresh, 'orange', linewidth=2, label='Adaptive CFAR Threshold')
    
    for b in boundaries:
        if 0 < b < max_seconds:
            ax.axvline(b, color='r', linestyle='--', linewidth=1.5, alpha=0.7, 
                      label='Boundaries' if b == boundaries[1] else '')
    
    ax.set_ylabel('Change Score')
    ax.set_xlabel('Time (s)')
    ax.set_title('Change Detection mit CFAR')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main(audio_files):
    t0 = time.perf_counter()
    cfg = Config()


    for j, audio_file in enumerate(audio_files):
        for k in range[25,51]:
            ki = k / 100
            for l in range[65,96]:
                li = l / 100

                cfg.CFAR_ALPHA = ki
                cfg.CFAR_K_FRACTION = li

                t1 = time.perf_counter()
                cfg.INPUT_FILE = audio_file
                cfg.OUT_DIR = f"out/output_segment_{j}"
            
                print("="*60)
                print("Interleaved Audio Segmentation (CFAR)")
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
                features = compute_stft_features(y, sr, cfg.WINDOW_MS, cfg.HOP_MS)
                print(f"  → Frames: {len(features['times'])}")
                
                # Change Detection mit CFAR
                print(f"\n[3/6] Detecting changes (CFAR)...")
                print(f"  → Method: {cfg.CFAR_METHOD}")
                print(f"  → Guard cells: {cfg.CFAR_GUARD_CELLS}")
                print(f"  → Train cells: {cfg.CFAR_TRAIN_CELLS}")
                print(f"  → Alpha: {cfg.CFAR_ALPHA}")
                
                change_score, threshold = detect_changes_cfar(features, cfg)
                
                min_segment_s = cfg.MIN_SEGMENT_MS / 1000.0
                boundaries = find_boundaries_cfar(change_score, threshold, features['times'], min_segment_s)
                print(f"  → Boundaries found: {len(boundaries)-2}")
                
                # Segmentation
                print(f"\n[4/6] Creating segments...")
                segments = create_segments(boundaries)
                print(f"  → Segments: {len(segments)}")
                
                segment_features = extract_segment_features(y, sr, segments)
                
                # Clustering
                print(f"\n[5/6] Clustering segments...")
                labels = cluster_segments(segment_features, cfg.NUM_CLUSTERS)
                n_signals = len(set(labels))
                print(f"  → Number of signals detected: {n_signals}")
                
                for i in range(n_signals):
                    count = np.sum(labels == i)
                    mean_freq = segment_features[labels == i, 0].mean()
                    print(f"  → Signal {chr(ord('A')+i)}: {count} segments, ~{mean_freq:.0f} Hz")
                
                # Reconstruction
                print(f"\n[6/6] Reconstructing signals...")
                exported = reconstruct_signals(y, sr, segments, labels, cfg.OUT_DIR, cfg.EXPORT_FORMAT)
                
                for name, path in sorted(exported.items()):
                    print(f"  → {name}: {path}")

                t2 = time.perf_counter()
                
                # Visualization
                print(f"\n[PLOT] Generating visualization...")
                plot_results(y, sr, features, change_score, threshold, boundaries, 
                            cfg.OUT_DIR, segments, labels, max_seconds=3.0)
                
                t3 = time.perf_counter()

                print(f"\n{'='*60}")
                print(f"Run time: {t3-t1:.3f}s")
                print(f"Calculation time: {t2-t1:.3f}s")
                print(f"Plot time: {t3-t2:.3f}s")
                print(f"{'='*60}\n")

    t_all = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"Total time: {t_all:.3f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    audio_files = [
        #r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/rand/interleaved_1k_8k_vio_rand.mp3",
        r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/rand/interleaved_30_1k_8k_rand.mp3",
        # ... weitere Dateien
    ]
    
    main(audio_files)