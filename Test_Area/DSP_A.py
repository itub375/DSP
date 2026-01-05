"""
# DSP v6.1.2 - Multiple Denoising Methods
# =============================================
# Based on v6.0.3 crossfade, add post-processing denoising:
#   A. Low-pass Filter: Remove high-frequency noise (Recommended)
#   B. Spike Detection: Detect and smooth abnormal spikes
#   C. Edge Trimming: Cut off unstable parts at the beginning and end of each segment
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pydub import AudioSegment
import time
from scipy import signal as scipy_signal


@dataclass
class Config:
    INPUT_FILE: str = r"D:/DSP-master/Inputsignals/rand/interleaved_pod_1k_60sec_rand.mp3"
    OUT_DIR: str = "out/out_vA"

    WINDOW_MS: float = 2.0
    HOP_MS: float = 0.5
    MIN_AMPLITUDE_THRESHOLD: float = 0.01

    CHANGE_THRESHOLD_PERCENTILE: float = 85.0
    MIN_SEGMENT_MS: float = 5.0

    WEIGHT_CENTROID: float = 2.0
    WEIGHT_RMS: float = 2.0
    WEIGHT_Rolloff: float = 1.5
    WEIGHT_ZCR: float = 1.5
    WEIGHT_FLUX: float = 5.0
    WEIGHT_BANDWIDTH: float = 1.0

    CLUSTER_WEIGHT_CENTROID: float = 2.0
    CLUSTER_WEIGHT_RMS: float = 5.0
    CLUSTER_WEIGHT_ROLLOFF: float = 2.0
    CLUSTER_WEIGHT_ZCR: float = 1.0
    CLUSTER_WEIGHT_FLUX: float = 1.0
    CLUSTER_WEIGHT_BANDWIDTH: float = 1.5

    EXPORT_FORMAT: str = "mp3"
    VERBOSE: bool = True
    NUM_CLUSTERS: int = 2

    # ===== Crossfade Parameters  新增 =====
    EDGE_MS: float = 1.0
    FADE_TYPE: str = 'hann'

    # ===== Denoising Parameters =====
    # Denoising Methods:
    #   'none'    - No processing
    #   'lowpass' - Low-pass filter (remove high-frequency noise)
    #   'median'  - Median filter (remove impulse noise)
    #   'median2' - Double median filter (stronger denoising)
    #   'notch'   - Notch filter (remove specific frequency)
    #   'bandpass'- Band-pass filter (only retain speech frequency band)
    #   'spike'   - Spike detection (smooth abnormal values)
    #   'median_notch'  - Median + Notch combination (Recommended!)
    #   'median_lowpass'- Median + Low-pass combination
    #   'full'    - All combinations (strongest denoising)
    DENOISE_METHOD: str = 'median_notch'

    # Low-pass filter parameters
    LOWPASS_CUTOFF_HZ: float = 8000

    # Median filter parameters (good for removing impulse noise)
    MEDIAN_KERNEL_MS: float = 0.15  # Increasing makes denoising stronger but may affect sound quality

    # Notch filter parameters (remove specific frequency)
    NOTCH_FREQ_HZ: float = 1000  # Frequency to be removed (sine wave frequency)
    NOTCH_Q: float = 20  # Q value, smaller means wider bandwidth and more thorough removal

    # Band-pass filter parameters (only retain speech)
    BANDPASS_LOW_HZ: float = 200   # Lower limit
    BANDPASS_HIGH_HZ: float = 6000  # Upper limit

    # Spike detection parameters
    SPIKE_THRESHOLD: float = 3.0
    SPIKE_SMOOTH_MS: float = 0.2

    # Edge trimming parameters
    TRIM_MS: float = 0.3

    ENABLE_ALTERNATING_FIX: bool = True
    CLASSIFY_MODE: str = 'rms_threshold'


def load_audio(path: str) -> Tuple[np.ndarray, int]:
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
    y = np.clip(y, -1.0, 1.0)
    int16 = (y * 32767.0).astype(np.int16)
    seg = AudioSegment(data=int16.tobytes(), sample_width=2, frame_rate=sr, channels=1)
    seg.export(path, format=fmt, bitrate="192k" if fmt == "mp3" else None)


def show_menu() -> Tuple[int, str]:   #新增交互式菜单
    print("\n" + "=" * 50)
    print("   DSP v6.1.2 - Multi Denoise")
    print("=" * 50)

    print("\nPlease select the number of audio sources:")
    print("  [2] 2-channel interleaving (A-B-A-B...)")
    print("  [3] 3-channel interleaving (A-B-C-A-B-C...)")

    while True:
        choice = input("\nPlease enter (2/3): ").strip()
        if choice in ['2', '3']:
            num_clusters = int(choice)
            break
        print("  ❌ Please enter 2 or 3")

    print("\nPlease select classification mode:")
    print("  [1] RMS threshold classification (Recommended for sine wave + speech)")
    print("  [2] K-means clustering (General purpose)")

    while True:
        mode_choice = input("\nPlease enter (1/2): ").strip()
        if mode_choice == '1':
            classify_mode = 'rms_threshold'   #新增双分类模式
            break
        elif mode_choice == '2':
            classify_mode = 'kmeans'
            break
        print("  ❌ Please enter 1 or 2")

    return num_clusters, classify_mode


def apply_mode_optimization(cfg: Config, num_clusters: int, classify_mode: str):
    cfg.NUM_CLUSTERS = num_clusters
    cfg.CLASSIFY_MODE = classify_mode

    print(f"\n  [Config]")
    print(f"    - Sources: {num_clusters}")
    print(f"    - Classify: {classify_mode}")
    print(f"    - Crossfade: {cfg.EDGE_MS}ms")
    print(f"    - Denoise: {cfg.DENOISE_METHOD}")

    if num_clusters == 2:
        cfg.ENABLE_ALTERNATING_FIX = True
    else:
        cfg.ENABLE_ALTERNATING_FIX = False
        cfg.WEIGHT_CENTROID = 3.0
        cfg.WEIGHT_FLUX = 6.0


def compute_features(y: np.ndarray, sr: int, window_ms: float, hop_ms: float,
                     min_amplitude: float = 0.0) -> Dict:
    window_samples = int(window_ms * sr / 1000)
    hop_samples = int(hop_ms * sr / 1000)

    if window_samples % 2 != 0:
        window_samples += 1

    window = np.hanning(window_samples)
    n_frames = 1 + (len(y) - window_samples) // hop_samples

    centroids = np.zeros(n_frames)
    rms_values = np.zeros(n_frames)
    rolloffs = np.zeros(n_frames)
    zcr_values = np.zeros(n_frames)
    flux_values = np.zeros(n_frames)
    bandwidth_values = np.zeros(n_frames)
    valid_frames = np.ones(n_frames, dtype=bool)

    freqs = np.fft.rfftfreq(window_samples, 1 / sr)
    prev_spec = None

    for i in range(n_frames):
        start = i * hop_samples
        frame = y[start:start + window_samples]

        frame_rms = np.sqrt(np.mean(frame ** 2))

        if frame_rms < min_amplitude:
            valid_frames[i] = False
            rms_values[i] = frame_rms
            continue

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
        'valid_frames': valid_frames
    }


def detect_boundaries(features: Dict, cfg: Config):
    centroid = features['centroid']
    rms = features['rms']
    rolloff = features['rolloff']
    zcr = features['zcr']
    flux = features['flux']
    bandwidth = features['bandwidth']
    valid_frames = features['valid_frames']
    times = features['times']

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

    wc, wrms, wroll, wzcr, wflux, wbw = (cfg.WEIGHT_CENTROID, cfg.WEIGHT_RMS,
                                         cfg.WEIGHT_Rolloff, cfg.WEIGHT_ZCR,
                                         cfg.WEIGHT_FLUX, cfg.WEIGHT_BANDWIDTH)

    combined_change = (wc * cent_change + wrms * rms_change + wroll * roll_change +
                       wzcr * zcr_change + wflux * flux_change + wbw * bw_change
                       ) / (wc + wrms + wroll + wzcr + wflux + wbw)

    combined_change[~valid_frames] = 0.0

    kernel = np.ones(5) / 5
    combined_change = np.convolve(combined_change, kernel, mode='same')

    valid_changes = combined_change[valid_frames]
    threshold = np.percentile(valid_changes, cfg.CHANGE_THRESHOLD_PERCENTILE) if len(valid_changes) > 0 else 0.0

    min_segment_s = cfg.MIN_SEGMENT_MS / 1000.0
    candidates = []
    for i in range(1, len(combined_change) - 1):
        if (valid_frames[i] and combined_change[i] > threshold and
                combined_change[i] >= combined_change[i - 1] and
                combined_change[i] >= combined_change[i + 1]):
            candidates.append(times[i])

    if len(candidates) < 2:
        return [0.0, times[-1]], combined_change, threshold

    boundaries = [0.0]
    for t in candidates:
        if t - boundaries[-1] >= min_segment_s:
            boundaries.append(t)
    boundaries.append(times[-1])

    return boundaries, combined_change, threshold


def create_segments(boundaries: List[float], features: Dict,
                    min_segment_ms: float = 0.0) -> List[Tuple[float, float]]:
    segments = []
    times = features['times']
    valid_frames = features['valid_frames']

    n = len(boundaries)
    for i in range(n - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        if min_segment_ms > 0 and (end - start) * 1000.0 < min_segment_ms:
            continue

        start_idx = np.searchsorted(times, start)
        end_idx = np.searchsorted(times, end)
        segment_valid = valid_frames[start_idx:end_idx]

        if len(segment_valid) > 0:
            if np.sum(segment_valid) / len(segment_valid) < 0.3:
                continue

        segments.append((start, end))
    return segments


def extract_segment_features(y: np.ndarray, sr: int,
                             segments: List[Tuple[float, float]],
                             min_amplitude: float = 0.0) -> np.ndarray:
    features = []

    for start_t, end_t in segments:
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        segment = y[start_idx:end_idx]

        if len(segment) < 10:
            features.append([0, 0, 0, 0, 0, 0])
            continue

        segment_rms = np.sqrt(np.mean(segment ** 2))
        if segment_rms < min_amplitude:
            features.append([0, 0, 0, 0, 0, 0])
            continue

        rms = segment_rms
        zcr = np.sum(np.abs(np.diff(np.sign(segment)))) / (2 * len(segment))

        window = np.hanning(len(segment))
        spec = np.abs(np.fft.rfft(segment * window))
        freqs = np.fft.rfftfreq(len(segment), 1 / sr)
        spec_power = spec ** 2

        if spec_power.sum() > 1e-10:
            centroid = np.sum(freqs * spec_power) / spec_power.sum()
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec_power) / spec_power.sum())
            flux = np.std(spec_power)
        else:
            centroid, rolloff, bandwidth, flux = 0, 0, 0, 0

        features.append([centroid, rms, rolloff, zcr, flux, bandwidth])

    return np.array(features)


def cluster_segments_kmeans(features: np.ndarray, n_clusters: int, cfg: Config,
                            verbose: bool = True) -> np.ndarray:
    if len(features) == 0:
        return np.array([])

    weights = np.array([
        cfg.CLUSTER_WEIGHT_CENTROID,
        cfg.CLUSTER_WEIGHT_RMS,
        cfg.CLUSTER_WEIGHT_ROLLOFF,
        cfg.CLUSTER_WEIGHT_ZCR,
        cfg.CLUSTER_WEIGHT_FLUX,
        cfg.CLUSTER_WEIGHT_BANDWIDTH
    ])
    weights = weights / weights.sum()

    features_norm = features.copy()
    for i in range(features.shape[1]):
        col = features[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 0:
            features_norm[:, i] = (col - col_min) / (col_max - col_min)

    features_weighted = features_norm * np.sqrt(weights)

    np.random.seed(42)
    n_samples = len(features_weighted)

    if n_clusters >= n_samples:
        return np.arange(n_samples)

    centers = [features_weighted[np.random.randint(n_samples)]]
    for _ in range(n_clusters - 1):
        distances = np.min([np.sum((features_weighted - c) ** 2, axis=1) for c in centers], axis=0)
        probs = distances / (distances.sum() + 1e-10)
        centers.append(features_weighted[np.random.choice(n_samples, p=probs)])
    centers = np.array(centers)

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(100):
        distances = np.sum((features_weighted[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = features_weighted[mask].mean(axis=0)

    rms_means = [features[labels == k, 1].mean() for k in range(n_clusters)]
    sorted_order = np.argsort(rms_means)
    label_map = {old: new for new, old in enumerate(sorted_order)}
    labels = np.array([label_map[l] for l in labels])

    if verbose:
        print(f"\n  K-means:")
        for k in range(n_clusters):
            count = np.sum(labels == k)
            ratio = count / len(labels) * 100
            mean_rms = features[labels == k, 1].mean()
            print(f"    {chr(ord('A') + k)}: {count} seg ({ratio:.1f}%), RMS={mean_rms:.4f}")

    return labels


def classify_by_rms_threshold(features: np.ndarray, segments: List[Tuple[float, float]],
                               n_clusters: int, verbose: bool = True) -> np.ndarray:
    if len(features) == 0:
        return np.array([])

    rms_values = features[:, 1]

    if verbose:
        print(f"\n  RMS: min={rms_values.min():.4f}, max={rms_values.max():.4f}, "
              f"median={np.median(rms_values):.4f}")

    if n_clusters == 2:
        sorted_rms = np.sort(rms_values)
        best_threshold = np.median(rms_values)
        best_variance = 0

        p10, p90 = np.percentile(rms_values, [10, 90])
        search_range = sorted_rms[(sorted_rms >= p10) & (sorted_rms <= p90)]

        for i in range(len(search_range) - 1):
            threshold = (search_range[i] + search_range[i + 1]) / 2
            low_group = rms_values[rms_values <= threshold]
            high_group = rms_values[rms_values > threshold]

            if len(low_group) > 0 and len(high_group) > 0:
                n1, n2 = len(low_group), len(high_group)
                m1, m2 = low_group.mean(), high_group.mean()
                between_variance = n1 * n2 * (m1 - m2) ** 2 / (n1 + n2) ** 2

                if between_variance > best_variance:
                    best_variance = between_variance
                    best_threshold = threshold

        if verbose:
            print(f"  Otsu: {best_threshold:.4f}")

        labels = (rms_values > best_threshold).astype(int)

    else:
        p33, p67 = np.percentile(rms_values, [33, 67])
        labels = np.zeros(len(rms_values), dtype=int)
        labels[rms_values > p33] = 1
        labels[rms_values > p67] = 2

    if verbose:
        print(f"\n  Result:")
        for k in range(n_clusters):
            count = np.sum(labels == k)
            ratio = count / len(labels) * 100
            mean_rms = rms_values[labels == k].mean() if count > 0 else 0
            print(f"    {chr(ord('A') + k)}: {count} seg ({ratio:.1f}%), RMS={mean_rms:.4f}")

    return labels


def classify_segments(features: np.ndarray, segments: List[Tuple[float, float]],
                      n_clusters: int, cfg: Config) -> np.ndarray:
    if cfg.CLASSIFY_MODE == 'rms_threshold':
        return classify_by_rms_threshold(features, segments, n_clusters, cfg.VERBOSE)
    else:
        return cluster_segments_kmeans(features, n_clusters, cfg, cfg.VERBOSE)


def fix_alternating_pattern(labels: np.ndarray, segment_features: np.ndarray,
                            cfg: Config, verbose: bool = True) -> np.ndarray:
    if len(labels) < 3:
        return labels

    labels = labels.copy()
    fix_count = 0

    rms_values = segment_features[:, 1]

    centers = {}
    for k in range(2):
        mask = labels == k
        if np.sum(mask) > 0:
            centers[k] = rms_values[mask].mean()

    if len(centers) < 2:
        return labels

    i = 1
    while i < len(labels) - 1:
        prev_label = labels[i - 1]
        curr_label = labels[i]
        next_label = labels[i + 1]

        if prev_label == curr_label == next_label:
            curr_rms = rms_values[i]
            dist_to_0 = abs(curr_rms - centers[0])
            dist_to_1 = abs(curr_rms - centers[1])

            if labels[i] == 0 and dist_to_1 < dist_to_0 * 0.7:
                labels[i] = 1
                fix_count += 1
            elif labels[i] == 1 and dist_to_0 < dist_to_1 * 0.7:
                labels[i] = 0
                fix_count += 1

        elif prev_label == curr_label and curr_label != next_label:
            prev_rms = rms_values[i - 1]
            other_label = 1 - curr_label
            dist_prev_to_other = abs(prev_rms - centers[other_label])
            dist_prev_to_curr = abs(prev_rms - centers[curr_label])

            if dist_prev_to_other < dist_prev_to_curr * 0.6:
                labels[i - 1] = other_label
                fix_count += 1

        i += 1

    if verbose and fix_count > 0:
        print(f"    [Alt fix] {fix_count} seg")

    return labels


# ============================================================================
# Crossfade 新增功能
# ============================================================================

def create_fade_curve(length: int, fade_type: str, direction: str) -> np.ndarray:
    if length <= 0:
        return np.array([])

    t = np.linspace(0, 1, length)

    if fade_type == 'hann':
        if direction == 'in':
            curve = 0.5 * (1 - np.cos(np.pi * t))
        else:
            curve = 0.5 * (1 + np.cos(np.pi * t))
    else:
        if direction == 'in':
            curve = t
        else:
            curve = 1 - t

    return curve


def crossfade_segments(seg1: np.ndarray, seg2: np.ndarray,
                       crossfade_samples: int, fade_type: str = 'hann') -> np.ndarray:
    if crossfade_samples <= 0 or len(seg1) < crossfade_samples or len(seg2) < crossfade_samples:
        return np.concatenate([seg1, seg2])

    fade_out = create_fade_curve(crossfade_samples, fade_type, 'out')
    fade_in = create_fade_curve(crossfade_samples, fade_type, 'in')

    seg1_end = seg1[-crossfade_samples:] * fade_out
    seg2_start = seg2[:crossfade_samples] * fade_in
    crossfade_region = seg1_end + seg2_start

    result = np.concatenate([
        seg1[:-crossfade_samples],
        crossfade_region,
        seg2[crossfade_samples:]
    ])

    return result


# ============================================================================
# Denoising Methods
# ============================================================================

def denoise_lowpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Low-pass Filter: Remove high-frequency noise"""
    nyquist = sr / 2
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)

    b, a = scipy_signal.butter(4, normalized_cutoff, btype='low')
    y_filtered = scipy_signal.filtfilt(b, a, y)

    return y_filtered.astype(np.float32)


def denoise_median(y: np.ndarray, sr: int, kernel_ms: float) -> np.ndarray:
    """Median Filter: Remove impulse noise (good for ticking sounds)"""
    kernel_size = int(kernel_ms * sr / 1000)
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be an odd number

    y_filtered = scipy_signal.medfilt(y, kernel_size)

    return y_filtered.astype(np.float32)


def denoise_notch(y: np.ndarray, sr: int, freq_hz: float, Q: float) -> np.ndarray:
    """Notch Filter: Remove specific frequency (e.g., residual 1kHz sine wave)"""
    nyquist = sr / 2
    normalized_freq = freq_hz / nyquist

    if normalized_freq >= 1.0:
        return y

    b, a = scipy_signal.iirnotch(normalized_freq, Q)
    y_filtered = scipy_signal.filtfilt(b, a, y)

    return y_filtered.astype(np.float32)


def denoise_bandpass(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Band-pass Filter: Only retain speech frequency band"""
    nyquist = sr / 2
    low = max(low_hz / nyquist, 0.01)
    high = min(high_hz / nyquist, 0.99)

    b, a = scipy_signal.butter(4, [low, high], btype='band')
    y_filtered = scipy_signal.filtfilt(b, a, y)

    return y_filtered.astype(np.float32)


def denoise_spike(y: np.ndarray, sr: int, threshold: float, smooth_ms: float) -> np.ndarray:
    """Spike Detection: Smooth abnormal spikes"""
    smooth_samples = max(int(smooth_ms * sr / 1000), 3)

    kernel = np.ones(smooth_samples) / smooth_samples
    local_rms = np.sqrt(np.convolve(y**2, kernel, mode='same'))

    global_median = np.median(local_rms)

    spike_mask = np.abs(y) > threshold * global_median

    if not spike_mask.any():
        return y

    y_smooth = y.copy()

    from scipy.ndimage import binary_dilation
    spike_mask = binary_dilation(spike_mask, iterations=smooth_samples//2)

    for i in range(len(y)):
        if spike_mask[i]:
            start = max(0, i - smooth_samples)
            end = min(len(y), i + smooth_samples)
            neighbors = y[start:end][~spike_mask[start:end]]
            if len(neighbors) > 0:
                y_smooth[i] = np.mean(neighbors)
            else:
                y_smooth[i] = 0

    return y_smooth


def apply_denoise(y: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    """Apply denoising processing"""
    method = cfg.DENOISE_METHOD.lower()

    if method == 'none':
        return y

    result = y.copy()

    if method == 'lowpass':
        result = denoise_lowpass(result, sr, cfg.LOWPASS_CUTOFF_HZ)
        print(f"      Applied lowpass (cutoff={cfg.LOWPASS_CUTOFF_HZ}Hz)")

    elif method == 'median':
        result = denoise_median(result, sr, cfg.MEDIAN_KERNEL_MS)
        print(f"      Applied median filter (kernel={cfg.MEDIAN_KERNEL_MS}ms)")

    elif method == 'median2':
        # Double median filter, stronger denoising
        result = denoise_median(result, sr, cfg.MEDIAN_KERNEL_MS)
        result = denoise_median(result, sr, cfg.MEDIAN_KERNEL_MS * 1.5)
        print(f"      Applied double median filter")

    elif method == 'notch':
        result = denoise_notch(result, sr, cfg.NOTCH_FREQ_HZ, cfg.NOTCH_Q)
        print(f"      Applied notch filter (freq={cfg.NOTCH_FREQ_HZ}Hz, Q={cfg.NOTCH_Q})")

    elif method == 'bandpass':
        result = denoise_bandpass(result, sr, cfg.BANDPASS_LOW_HZ, cfg.BANDPASS_HIGH_HZ)
        print(f"      Applied bandpass ({cfg.BANDPASS_LOW_HZ}-{cfg.BANDPASS_HIGH_HZ}Hz)")

    elif method == 'spike':
        result = denoise_spike(result, sr, cfg.SPIKE_THRESHOLD, cfg.SPIKE_SMOOTH_MS)
        print(f"      Applied spike removal (threshold={cfg.SPIKE_THRESHOLD}x)")

    elif method == 'median_notch':
        # Combination: Median + Notch
        result = denoise_median(result, sr, cfg.MEDIAN_KERNEL_MS)
        result = denoise_notch(result, sr, cfg.NOTCH_FREQ_HZ, cfg.NOTCH_Q)
        print(f"      Applied median + notch")

    elif method == 'median_lowpass':
        # Combination: Median + Low-pass
        result = denoise_median(result, sr, cfg.MEDIAN_KERNEL_MS)
        result = denoise_lowpass(result, sr, cfg.LOWPASS_CUTOFF_HZ)
        print(f"      Applied median + lowpass")

    elif method == 'full':
        # Full combination: Median → Notch → Low-pass
        result = denoise_median(result, sr, cfg.MEDIAN_KERNEL_MS)
        result = denoise_median(result, sr, cfg.MEDIAN_KERNEL_MS)  # Twice
        result = denoise_notch(result, sr, cfg.NOTCH_FREQ_HZ, cfg.NOTCH_Q)
        result = denoise_lowpass(result, sr, cfg.LOWPASS_CUTOFF_HZ)
        print(f"      Applied full denoise chain")

    elif method == 'all':
        result = denoise_median(result, sr, cfg.MEDIAN_KERNEL_MS)
        result = denoise_notch(result, sr, cfg.NOTCH_FREQ_HZ, cfg.NOTCH_Q)
        result = denoise_lowpass(result, sr, cfg.LOWPASS_CUTOFF_HZ)
        print(f"      Applied all filters")

    return result


# ============================================================================
# Signal Reconstruction
# ============================================================================

def reconstruct_signals(y: np.ndarray, sr: int,
                        segments: List[Tuple[float, float]],
                        labels: np.ndarray,
                        out_dir: str, fmt: str,
                        cfg: Config) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    unique_labels = sorted(set(labels))
    exported = {}

    crossfade_samples = max(int(cfg.EDGE_MS * sr / 1000), 2)
    trim_samples = int(cfg.TRIM_MS * sr / 1000) if cfg.DENOISE_METHOD in ['trim', 'all'] else 0

    total_samples = len(y)

    print(f"\n  Reconstruction:")
    print(f"    Crossfade: {crossfade_samples} samples ({cfg.EDGE_MS}ms)")
    print(f"    Denoise: {cfg.DENOISE_METHOD}")

    for label in unique_labels:
        name = chr(ord('A') + label)

        segment_data_list = []
        input_samples = 0

        for (start_t, end_t), seg_label in zip(segments, labels):
            if seg_label == label:
                start_idx = int(start_t * sr)
                end_idx = int(end_t * sr)
                segment = y[start_idx:end_idx].copy()

                # Edge trimming (if enabled)
                if trim_samples > 0 and len(segment) > 2 * trim_samples + 10:
                    segment = segment[trim_samples:-trim_samples]

                if len(segment) > 0:
                    segment_data_list.append(segment)
                    input_samples += len(segment)

        if len(segment_data_list) == 0:
            reconstructed = np.zeros(1000)
        elif len(segment_data_list) == 1:
            reconstructed = segment_data_list[0]
        else:
            # Crossfade concatenation
            reconstructed = segment_data_list[0]
            for i in range(1, len(segment_data_list)):
                reconstructed = crossfade_segments(
                    reconstructed,
                    segment_data_list[i],
                    crossfade_samples,
                    cfg.FADE_TYPE
                )

        # Apply denoising
        if cfg.DENOISE_METHOD != 'none':
            print(f"    {name}: applying denoise...")
            reconstructed = apply_denoise(reconstructed, sr, cfg)

        input_duration = input_samples / sr
        output_duration = len(reconstructed) / sr
        ratio = input_samples / total_samples * 100

        num_joins = len(segment_data_list) - 1
        lost_samples = num_joins * crossfade_samples
        lost_sec = lost_samples / sr
        speed_ratio = input_duration / output_duration if output_duration > 0 else 1

        print(f"    {name}: {len(segment_data_list)} seg, "
              f"in {input_duration:.2f}s ({ratio:.1f}%), "
              f"out {output_duration:.2f}s, "
              f"speed {speed_ratio:.2f}x")

        out_path = os.path.join(out_dir, f"signal_{name}.{fmt}")
        save_audio(reconstructed, sr, out_path, fmt)
        exported[name] = out_path

    return exported


def plot_results(y: np.ndarray, sr: int, features: Dict,
                 change_score: np.ndarray, threshold: float,
                 boundaries: List[float], segments: List[Tuple[float, float]],
                 labels: np.ndarray, n_clusters: int,
                 save_path: str, max_seconds: float = None):
    t = np.arange(len(y)) / sr

    if max_seconds:
        plot_mask = t <= max_seconds
        t = t[plot_mask]
        y_plot = y[plot_mask]
    else:
        max_seconds = t[-1]
        y_plot = y

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, y_plot, 'k-', linewidth=0.5, alpha=0.7)

    colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
    plotted_labels = set()
    for (start_t, end_t), label in zip(segments, labels):
        if start_t < max_seconds:
            show_label = label not in plotted_labels
            ax.axvspan(start_t, min(end_t, max_seconds), alpha=0.3,
                       color=colors[label % len(colors)],
                       label=f'Source {chr(ord("A") + label)}' if show_label else '')
            if show_label:
                plotted_labels.add(label)

    ax.set_ylabel('Amplitude')
    ax.set_title(f'v6.1.2 Multi Denoise')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    feat_t = features['times']
    mask = feat_t <= max_seconds if max_seconds else np.ones(len(feat_t), dtype=bool)
    feat_t_plot = feat_t[mask]

    ax = axes[1]
    ax.plot(feat_t_plot, features['centroid'][mask], 'b-', linewidth=1, alpha=0.7)
    ax.set_ylabel('Freq (Hz)')
    ax.set_title('Spectral Centroid')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(feat_t_plot, features['rms'][mask], 'g-', linewidth=1, alpha=0.7)
    ax.set_ylabel('RMS')
    ax.set_title('RMS Energy')
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(feat_t_plot, change_score[mask], 'purple', linewidth=1)
    ax.axhline(threshold, color='orange', linestyle=':', linewidth=2)
    ax.set_ylabel('Change')
    ax.set_xlabel('Time (s)')
    ax.set_title('Change Detection')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    num_clusters, classify_mode = show_menu()

    cfg = Config()
    apply_mode_optimization(cfg, num_clusters, classify_mode)

    for audio_path in audio_files:
        t1 = time.perf_counter()
        cfg.INPUT_FILE = audio_path
        base_name = os.path.splitext(os.path.basename(cfg.INPUT_FILE))[0]
        cfg.OUT_DIR = f"D:/python_project/DSP-master/out/v6.1.2/{base_name}/"

        print(f"\n{'=' * 50}")
        print(f"Processing: {base_name}")
        print(f"{'=' * 50}")

        print(f"\n[1/6] Load audio...")
        y, sr = load_audio(cfg.INPUT_FILE)
        duration = len(y) / sr
        print(f"  SR={sr}Hz, Duration={duration:.2f}s")

        print(f"\n[2/6] Extract features...")
        features = compute_features(y, sr, cfg.WINDOW_MS, cfg.HOP_MS, cfg.MIN_AMPLITUDE_THRESHOLD)
        print(f"  Frames: {len(features['times'])}")

        print(f"\n[3/6] Detect boundaries...")
        boundaries, change_score, threshold = detect_boundaries(features, cfg)
        print(f"  Boundaries: {len(boundaries) - 2}")

        print(f"\n[4/6] Create segments...")
        segments = create_segments(boundaries, features, cfg.MIN_SEGMENT_MS)
        print(f"  Segments: {len(segments)}")

        print(f"\n[5/6] Classify ({cfg.CLASSIFY_MODE})...")
        segment_features = extract_segment_features(y, sr, segments, cfg.MIN_AMPLITUDE_THRESHOLD)
        labels = classify_segments(segment_features, segments, cfg.NUM_CLUSTERS, cfg)

        if cfg.NUM_CLUSTERS == 2 and cfg.ENABLE_ALTERNATING_FIX:
            labels = fix_alternating_pattern(labels, segment_features, cfg, verbose=cfg.VERBOSE)

        print(f"\n[6/6] Reconstruct + Denoise...")
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        exported = reconstruct_signals(y, sr, segments, labels, cfg.OUT_DIR, cfg.EXPORT_FORMAT, cfg)

        print(f"\n  Output:")
        for name, path in sorted(exported.items()):
            print(f"    {name}: {path}")

        save_path_plt = os.path.join(cfg.OUT_DIR, f"plt_{base_name}.png")
        plot_results(y, sr, features, change_score, threshold, boundaries,
                     segments, labels, cfg.NUM_CLUSTERS, save_path_plt, max_seconds=3.0)

        print(f"\nDone! {time.perf_counter() - t1:.2f}s")

    print("\nAll done!")


if __name__ == "__main__":
    audio_files = [
        # r"D:/python_project/DSP-master/Inputsignals/rand/interleaved_20kHz_pod_1k_60sec_rand.mp3",
        r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_1k_GOD_30sec_rand.mp3",
    ]
    main()