"""

"""
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pydub import AudioSegment

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
    
    # Clustering
    NUM_CLUSTERS: int = None

    # Change Detection
    CHANGE_THRESHOLD_PERCENTILE: float = 85.0
    MIN_SEGMENT_MS: float = 9.8
    MERGE_TOLERANCE_MS: float = 5.0
    
    # Export
    EXPORT_FORMAT: str = "mp3"
    VERBOSE: bool = True

    # Weight Change Score
    WEIGHT_CENTROID: int    = 2.0      # 2.0
    WEIGHT_RMS:      int    = 2.0      # 2.0
    WEIGHT_Rolloff:  int    = 1.5      # 1.5
    WEIGHT_ZCR:      int    = 1.5      # 1.5
    WEIGHT_FLUX:     int    = 20.0      # 5.0
    WEIGHT_BANDWIDTH:int    = 1.0      # 1.0
    
    # Dead-Zone um Wechselstellen
    DEADZONE_MS: float = 0

    # ⭐ NEU: Anti-Click Parameter
    FADE_MS: float = 3.0              # Fade-Länge in ms
    ALIGN_TO_ZERO_CROSSING: bool = True  # Nulldurchgang-Alignment aktivieren

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

        #if centroid_range > 6000 or zcr_range > 0.18:
        #    n_clusters = 4
        if centroid_range > 3000 or zcr_range > 0.1:
            n_clusters = 3
        elif centroid_range > 1000 or zcr_range > 0.05:
            n_clusters = 2
        else:
            n_clusters = 1
    
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

def find_nearest_zero_crossing(y: np.ndarray, idx: int, search_range: int = 100) -> int:
    """
    Findet den nächsten Nulldurchgang in der Nähe von idx
    
    Args:
        y: Audio-Signal
        idx: Gewünschter Index
        search_range: Suchbereich in Samples
    
    Returns:
        Index des nächsten Nulldurchgangs
    """
    start = max(0, idx - search_range)
    end = min(len(y), idx + search_range)
    
    # Finde Vorzeichenwechsel
    segment = y[start:end]
    sign_changes = np.where(np.diff(np.sign(segment)))[0]
    
    if len(sign_changes) == 0:
        return idx  # Kein Nulldurchgang gefunden, nutze Original-Index
    
    # Wähle den nächsten zum ursprünglichen Index
    abs_distances = np.abs(sign_changes + start - idx)
    nearest_idx = sign_changes[np.argmin(abs_distances)] + start
    
    return nearest_idx

def align_segments_to_zero_crossings(y: np.ndarray, sr: int,
                                      segments: List[Tuple[float, float]],
                                      search_range_ms: float = 2.0) -> List[Tuple[float, float]]:
    """
    Richtet Segment-Grenzen an Nulldurchgängen aus
    """
    search_samples = int(search_range_ms * sr / 1000.0)
    aligned_segments = []
    
    for start_t, end_t in segments:
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        
        # Finde Nulldurchgänge
        new_start_idx = find_nearest_zero_crossing(y, start_idx, search_samples)
        new_end_idx = find_nearest_zero_crossing(y, end_idx, search_samples)
        
        # Konvertiere zurück zu Zeit
        new_start_t = new_start_idx / sr
        new_end_t = new_end_idx / sr
        
        aligned_segments.append((new_start_t, new_end_t))
    
    return aligned_segments

def reconstruct_signals_optimized(y: np.ndarray, sr: int, 
                                  segments: List[Tuple[float, float]], 
                                  labels: np.ndarray, out_dir: str, fmt: str,
                                  fade_ms: float = 3.0,
                                  align_to_zero: bool = True) -> Dict[str, str]:
    """
    BESTE LÖSUNG: Kombiniert Zero-Crossing Alignment + Fade
    
    Args:
        align_to_zero: Wenn True, werden Grenzen an Nulldurchgängen ausgerichtet
        fade_ms: Fade-Länge (kann kürzer sein bei Zero-Crossing Alignment)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Optional: Richte Segmente an Nulldurchgängen aus
    if align_to_zero:
        segments = align_segments_to_zero_crossings(y, sr, segments)
    
    fade_samples = int(fade_ms * sr / 1000.0)
    fade_in = (1 - np.cos(np.linspace(0, np.pi, fade_samples))) / 2
    fade_out = (1 + np.cos(np.linspace(0, np.pi, fade_samples))) / 2
    
    unique_labels = sorted(set(labels))
    exported = {}
    
    for label in unique_labels:
        name = chr(ord('A') + label)
        reconstructed = np.zeros_like(y)
        
        for (start_t, end_t), seg_label in zip(segments, labels):
            if seg_label == label:
                start_idx = int(start_t * sr)
                end_idx = int(end_t * sr)
                segment = y[start_idx:end_idx].copy()
                
                # Kurzer Fade (da wir bereits bei Nulldurchgang sind)
                fade_len = min(fade_samples, len(segment) // 4)
                if fade_len > 0:
                    segment[:fade_len] *= fade_in[:fade_len]
                    segment[-fade_len:] *= fade_out[-fade_len:]
                
                reconstructed[start_idx:end_idx] = segment
        
        out_path = os.path.join(out_dir, f"signal_{name}.{fmt}")
        save_audio(reconstructed, sr, out_path, fmt)
        exported[name] = out_path
    
    return exported

# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()

    files_to_process = audio_files

    for a in files_to_process:

        cfg.INPUT_FILE = a
        base_name = os.path.splitext(os.path.basename(cfg.INPUT_FILE))[0]
        cfg.OUT_DIR = "out/out_v5.8.1_shorted/" + str(base_name) + "/"  

    # Load[1/6]
        y, sr = load_audio(cfg.INPUT_FILE)
        
    # Features[2/6]
        features = compute_stft_features(y, sr, cfg.WINDOW_MS, cfg.HOP_MS,cfg.MIN_AMPLITUDE_THRESHOLD)
        
    # Change Detection[3/6]

        change_score, threshold = detect_changes(cfg.WEIGHT_CENTROID, cfg.WEIGHT_RMS, cfg.WEIGHT_Rolloff,cfg.WEIGHT_ZCR, cfg.WEIGHT_FLUX, cfg.WEIGHT_BANDWIDTH,features, cfg.CHANGE_THRESHOLD_PERCENTILE)
        min_segment_s = cfg.MIN_SEGMENT_MS / 1000.0
        boundaries = find_boundaries(change_score, threshold, features['times'],min_segment_s, features['valid_frames'])
        
    # Segmentation[4/6]
        segments = create_segments(boundaries,features,deadzone_ms=cfg.DEADZONE_MS,min_segment_ms=cfg.MIN_SEGMENT_MS,verbose=cfg.VERBOSE)
        
        segment_features = extract_segment_features(y, sr, segments,cfg.MIN_AMPLITUDE_THRESHOLD)
        
    # Clustering[5/6]
        labels = cluster_segments(segment_features, cfg.NUM_CLUSTERS)
        
    # Reconstruction reconstruct_signals_optimized
        exported = reconstruct_signals_optimized(y, sr, segments, labels, cfg.OUT_DIR, cfg.EXPORT_FORMAT)
   
if __name__ == "__main__":
    
    
    
    audio_files = [

    r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_30_1k_8k_rand.mp3",


    ]


    main()