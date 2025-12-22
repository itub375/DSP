"""
Robuster Interleaved Audio Segmentierer
Findet Wechselstellen und rekonstruiert Original-Signale

Änderung zu 5.2:

    Versuch die Farben im plot anzupassen zur besseren visualisierung.
    (HINWEIS: Anscheinend zu kompliziert um dafür zeit zu nehmen. Ignorieren und aktzeptieren)

NEU:


"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pydub import AudioSegment

@dataclass
class Config:
    # I/O
    INPUT_FILE: str = r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/rand/interleaved_1k_8k_20k_rand.mp3"
    OUT_DIR: str = "output_segments"
    
    # Analyse-Parameter
    WINDOW_MS: float = 2.0          # Fenster für Feature-Berechnung (ms)
    HOP_MS: float = 0.5             # Hop-Größe (ms)
    
    # Change Detection
    CHANGE_THRESHOLD_PERCENTILE: float = 85.0  # Schwelle für Change-Score
    MIN_SEGMENT_MS: float = 7.0    # Minimale Segmentlänge (ms)
    MERGE_TOLERANCE_MS: float = 5.0 # Toleranz für Segment-Merge
    
    # Clustering
    NUM_CLUSTERS: int = None        # None = automatisch bestimmen
    
    # Farben für Segmente (RGB-Tupel oder Hex-Strings)
    # Signal A, Signal B, Signal C, etc.
    
    #SEGMENT_COLORS: List = None  # None = automatische Farbzuweisung
    # Beispiele:
    SEGMENT_COLORS: List = [(1.0, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 1.0)]  # RGB
    # SEGMENT_COLORS: List = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Hex
    # SEGMENT_COLORS: List = ['red', 'green', 'blue']  # Named colors
    
    # Export
    EXPORT_FORMAT: str = "mp3"
    VERBOSE: bool = True

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
# FEATURE EXTRACTION
# ============================================================================

def compute_stft_features(y: np.ndarray, sr: int, window_ms: float, hop_ms: float):
    """Berechnet Spektral-Features mit STFT"""
    
    window_samples = int(window_ms * sr / 1000)
    hop_samples = int(hop_ms * sr / 1000)
    
    # Ensure window_samples is even for rfft
    if window_samples % 2 != 0:
        window_samples += 1
    
    # Hann-Window
    window = np.hanning(window_samples)
    
    # Framing
    n_frames = 1 + (len(y) - window_samples) // hop_samples
    
    # Features
    centroids = np.zeros(n_frames)
    rms_values = np.zeros(n_frames)
    rolloffs = np.zeros(n_frames)
    
    freqs = np.fft.rfftfreq(window_samples, 1/sr)
    
    for i in range(n_frames):
        start = i * hop_samples
        frame = y[start:start + window_samples] * window
        
        # RMS
        rms_values[i] = np.sqrt(np.mean(frame**2))
        
        # Spektrum
        spec = np.abs(np.fft.rfft(frame))
        spec_power = spec ** 2
        
        if spec_power.sum() > 1e-10:
            # Spectral Centroid
            centroids[i] = np.sum(freqs * spec_power) / spec_power.sum()
            
            # Spectral Rolloff (85%)
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                rolloffs[i] = freqs[rolloff_idx[0]]
    
    # Frame-Zeiten
    times = np.arange(n_frames) * hop_samples / sr
    
    return {
        'times': times,
        'centroid': centroids,
        'rms': rms_values,
        'rolloff': rolloffs,
        'hop_samples': hop_samples
    }

# ============================================================================
# CHANGE DETECTION
# ============================================================================

def detect_changes(features: Dict, threshold_percentile: float) -> np.ndarray:
    """
    Erkennt Änderungspunkte im Signal durch Kombination mehrerer Features
    """
    
    centroid = features['centroid']
    rms = features['rms']
    rolloff = features['rolloff']
    
    # Normalisiere Features auf [0,1]
    def normalize(x):
        x = np.copy(x)
        x_min, x_max = np.percentile(x, [5, 95])
        if x_max - x_min > 0:
            x = (x - x_min) / (x_max - x_min)
        return np.clip(x, 0, 1)
    
    cent_norm = normalize(centroid)
    rms_norm = normalize(rms)
    roll_norm = normalize(rolloff)
    
    # Berechne Änderungen (erste Ableitung)
    cent_change = np.abs(np.diff(cent_norm, prepend=cent_norm[0]))
    rms_change = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    roll_change = np.abs(np.diff(roll_norm, prepend=roll_norm[0]))
    
    # Kombiniere mit Gewichtung
    # Centroid ist meist am wichtigsten für Frequenz-Unterschiede
    combined_change = (
        2.0 * cent_change +
        1.0 * rms_change +
        1.5 * roll_change
    ) / 4.5
    
    # Smooth mit kleinem Moving Average
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    combined_change = np.convolve(combined_change, kernel, mode='same')
    
    # Threshold
    threshold = np.percentile(combined_change, threshold_percentile)
    
    return combined_change, threshold

def find_boundaries(change_score: np.ndarray, threshold: float, 
                   times: np.ndarray, min_segment_s: float) -> List[float]:
    """
    Findet Segment-Grenzen basierend auf Change-Score
    """
    
    # Finde Peaks über Threshold
    candidates = []
    for i in range(1, len(change_score)-1):
        if (change_score[i] > threshold and 
            change_score[i] >= change_score[i-1] and 
            change_score[i] >= change_score[i+1]):
            candidates.append(times[i])
    
    # Entferne zu nahe Grenzen
    if len(candidates) < 2:
        return [0.0, times[-1]]
    
    boundaries = [0.0]
    for t in candidates:
        if t - boundaries[-1] >= min_segment_s:
            boundaries.append(t)
    
    boundaries.append(times[-1])
    
    return boundaries

# ============================================================================
# SEGMENTATION & CLUSTERING
# ============================================================================

def create_segments(boundaries: List[float]) -> List[Tuple[float, float]]:
    """Erstellt Segment-Paare aus Boundaries"""
    return [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

def extract_segment_features(y: np.ndarray, sr: int, 
                            segments: List[Tuple[float, float]]) -> np.ndarray:
    """
    Extrahiert Features für jedes Segment:
    - Mean Spectral Centroid
    - Mean RMS
    - Mean Rolloff
    """
    
    features = []
    
    for start_t, end_t in segments:
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        segment = y[start_idx:end_idx]
        
        if len(segment) < 10:
            features.append([0, 0, 0])
            continue
        
        # FFT für Segment
        window = np.hanning(len(segment))
        spec = np.abs(np.fft.rfft(segment * window))
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        
        spec_power = spec ** 2
        
        # RMS
        rms = np.sqrt(np.mean(segment**2))
        
        # Centroid
        if spec_power.sum() > 1e-10:
            centroid = np.sum(freqs * spec_power) / spec_power.sum()
            
            # Rolloff
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            centroid = 0
            rolloff = 0
        
        features.append([centroid, rms, rolloff])
    
    return np.array(features)

def cluster_segments(features: np.ndarray, n_clusters: int = None) -> np.ndarray:
    """
    Clustert Segmente basierend auf Features
    Verwendet einfaches K-Means
    """
    
    if len(features) == 0:
        return np.array([])
    
    # Normalisiere Features
    features_norm = features.copy()
    for i in range(features.shape[1]):
        col = features[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 0:
            features_norm[:, i] = (col - col_min) / (col_max - col_min)
    
    # Bestimme Anzahl Cluster automatisch wenn nicht gegeben
    if n_clusters is None:
        # Einfache Heuristik: Schaue auf Centroid-Verteilung
        centroids = features[:, 0]
        centroid_range = centroids.max() - centroids.min()
        
        if centroid_range > 3000:  # Große Frequenz-Unterschiede
            n_clusters = 3
        elif centroid_range > 1000:
            n_clusters = 2
        else:
            n_clusters = 2  # Default
    
    # K-Means
    np.random.seed(42)
    n_samples = len(features_norm)
    
    # Initialisiere Cluster-Zentren
    if n_clusters >= n_samples:
        return np.arange(n_samples)
    
    # Wähle initiale Zentren (K-Means++)
    centers = [features_norm[np.random.randint(n_samples)]]
    for _ in range(n_clusters - 1):
        distances = np.min([np.sum((features_norm - c)**2, axis=1) for c in centers], axis=0)
        probs = distances / distances.sum()
        centers.append(features_norm[np.random.choice(n_samples, p=probs)])
    centers = np.array(centers)
    
    # Iteriere
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(100):
        # Assign
        distances = np.sum((features_norm[:, None, :] - centers[None, :, :])**2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        
        if np.all(new_labels == labels):
            break
        
        labels = new_labels
        
        # Update centers
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = features_norm[mask].mean(axis=0)
    
    # Sortiere Labels nach mittlerem Centroid (niedrig -> hoch)
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
    """
    Rekonstruiert Signale für jedes Label (mask mode)
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    unique_labels = sorted(set(labels))
    exported = {}
    
    for label in unique_labels:
        name = chr(ord('A') + label)
        
        # Mask mode: Nullen außerhalb der Label-Segmente
        reconstructed = np.zeros_like(y)
        
        for (start_t, end_t), seg_label in zip(segments, labels):
            if seg_label == label:
                start_idx = int(start_t * sr)
                end_idx = int(end_t * sr)
                reconstructed[start_idx:end_idx] = y[start_idx:end_idx]
        
        # Speichern
        out_path = os.path.join(out_dir, f"signal_{name}.{fmt}")
        save_audio(reconstructed, sr, out_path, fmt)
        exported[name] = out_path
    
    return exported

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(y: np.ndarray, sr: int, features: Dict, 
                change_score: np.ndarray, threshold: float,
                boundaries: List[float], segments: List[Tuple[float, float]], 
                labels: np.ndarray, custom_colors: List = None, max_seconds: float = None):
    """Visualisiert Ergebnisse"""
    
    t = np.arange(len(y)) / sr
    
    if max_seconds:
        plot_mask = t <= max_seconds
        t = t[plot_mask]
        y = y[plot_mask]
    else:
        max_seconds = t[-1]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Farben bestimmen
    n_labels = len(set(labels))
    if custom_colors and len(custom_colors) >= n_labels:
        # Verwende benutzerdefinierte Farben
        colors = custom_colors[:n_labels]
        print(f"  → Verwende benutzerdefinierte Farben: {colors[:n_labels]}")
    else:
        # Fallback auf automatische Farbzuweisung
        colors = plt.cm.Set3(np.linspace(0, 1, n_labels))
        if custom_colors:
            print(f"  ⚠ Warnung: Nicht genug Farben definiert ({len(custom_colors)} < {n_labels}), verwende automatische Farben")
    
    # 1. Waveform mit Segmenten
    ax = axes[0]
    ax.plot(t, y, 'k-', linewidth=0.5, alpha=0.7)

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
    
    # 4. Change Score
    ax = axes[3]
    if max_seconds:
        change = change_score[mask]
    else:
        change = change_score
    
    ax.plot(feat_t, change, 'purple', linewidth=1)
    ax.axhline(threshold, color='orange', linestyle=':', linewidth=2, label='Threshold')
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
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()
    
    print("="*60)
    print("Interleaved Audio Segmentation & Reconstruction")
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
    
    # Change Detection
    print(f"\n[3/6] Detecting changes...")
    change_score, threshold = detect_changes(features, cfg.CHANGE_THRESHOLD_PERCENTILE)
    print(f"  → Threshold: {threshold:.4f}")
    
    min_segment_s = cfg.MIN_SEGMENT_MS / 1000.0
    boundaries = find_boundaries(change_score, threshold, features['times'], min_segment_s)
    print(f"  → Boundaries found: {len(boundaries)-2}")  # -2 weil Start/End dabei
    
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
    
    # Visualization
    print(f"\n[PLOT] Generating visualization...")
    plot_results(y, sr, features, change_score, threshold, boundaries, 
                segments, labels, custom_colors=cfg.SEGMENT_COLORS, max_seconds=3.0)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()