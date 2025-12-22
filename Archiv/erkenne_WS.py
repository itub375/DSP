import numpy as np
import librosa
import matplotlib.pyplot as plt

# ===== Parameter anpassen =====
input_file     = "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/interleaved_50ms.mp3"  # dein Testsignal
frame_duration = 0.005   # Sekunden pro Frame (: 5 ms)
percentile     = 80     # wie “streng” der Schwellwert ist (70–90 testen)
min_gap_s      = 0.01   # Minimalabstand zwischen Wechseln (z.B. 30 ms)

# Dauer eines „echten“ Interleave-Segments (z.B. 50 ms)
true_segment_duration = 0.05   # 0.05 s = 50 ms


def compute_features(y, sr, frame_duration):
    """
    Berechnet Energie und spektralen Schwerpunkt pro Frame.

    Rückgabe:
      feats_norm: z-normalisierte Features [Energie, Centroid] pro Frame
      frame_length, hop_length: Längen in Samples
    """
    frame_length = int(frame_duration * sr)
    hop_length = frame_length

    window = np.hanning(frame_length)

    energies = []
    centroids = []

    for start in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[start:start + frame_length]
        frame_w = frame * window

        # Energie
        energy = np.mean(frame_w ** 2)

        # FFT
        spec = np.fft.rfft(frame_w)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(frame_length, 1 / sr)

        if mag.sum() == 0:
            centroid = 0.0
        else:
            centroid = (freqs * mag).sum() / mag.sum()

        energies.append(energy)
        centroids.append(centroid)

    feats = np.column_stack([energies, centroids])  # shape: (n_frames, 2)

    # z-Normierung je Feature
    mean = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-9
    feats_norm = (feats - mean) / std

    return feats_norm, frame_length, hop_length

def _peak_pick_from_diffs(diffs, frame_duration, percentile=80, min_gap_s=0.01):
    """
    Hilfsfunktion:
    - Threshold aus Perzentil
    - Peak-Picking (lokale Maxima)
    - Minimalabstand in Frames erzwingen

    Rückgabe:
      peaks: Frame-Indizes (beziehen sich auf diffs, also Übergang i -> i+1)
      threshold: verwendeter Schwellwert
    """
    # Schwellwert aus Perzentil
    threshold = np.percentile(diffs, percentile)

    # Kandidaten: alles über Schwellwert
    candidate_idxs = np.where(diffs >= threshold)[0]

    # Peak-Picking: nur lokale Maxima behalten
    peaks = []
    for idx in candidate_idxs:
        left  = diffs[idx - 1] if idx > 0 else -np.inf
        right = diffs[idx + 1] if idx < len(diffs) - 1 else -np.inf
        if diffs[idx] >= left and diffs[idx] >= right:
            peaks.append(idx)

    peaks = np.array(peaks, dtype=int)

    # Minimalabstand in Frames erzwingen
    if min_gap_s is not None and min_gap_s > 0 and len(peaks) > 0:
        min_gap_frames = int(np.ceil(min_gap_s / frame_duration))
        filtered = []
        last = -10**9
        for p in peaks:
            if p - last >= min_gap_frames:
                filtered.append(p)
                last = p
        peaks = np.array(filtered, dtype=int)

    return peaks, threshold

def detect_change_points(y, sr, frame_duration, percentile=80, min_gap_s=0.03):
    """
    Ursprüngliche Wechselstellen-Erkennung:
    verwendet den euklidischen Abstand zwischen
    [normierter Energie, normierter Centroid] (2D-Feature).
    """
    feats_norm, frame_length, hop_length = compute_features(y, sr, frame_duration)

    # Abstand zwischen aufeinanderfolgenden Frames im 2D-Feature-Raum
    diffs = np.linalg.norm(feats_norm[1:] - feats_norm[:-1], axis=1)

    peaks, threshold = _peak_pick_from_diffs(
        diffs,
        frame_duration=frame_duration,
        percentile=percentile,
        min_gap_s=min_gap_s
    )

    # Peaks liegen zwischen Frame i und i+1 → Grenze bei Frame (i+1)
    change_frame_indices = peaks + 1
    change_times = change_frame_indices * hop_length / sr

    return change_times, change_frame_indices, diffs, threshold


def detect_change_points_centroid_only(y, sr, frame_duration, percentile=80, min_gap_s=0.03):
    """
    Gegenprobe:
    - pro Frame nur den (z-normalisierten) spektralen Schwerpunkt betrachten
    - Differenz des Centroids zwischen Frame i und i+1
    - große Sprünge → Wechselstellen
    """
    feats_norm, frame_length, hop_length = compute_features(y, sr, frame_duration)

    # Nur die zweite Spalte: normierter Centroid
    centroids_norm = feats_norm[:, 1]

    # Absoluter Unterschied zwischen aufeinanderfolgenden Centroids
    centroid_diffs = np.abs(centroids_norm[1:] - centroids_norm[:-1])

    peaks, threshold = _peak_pick_from_diffs(
        centroid_diffs,
        frame_duration=frame_duration,
        percentile=percentile,
        min_gap_s=min_gap_s
    )

    change_frame_indices = peaks + 1
    change_times = change_frame_indices * hop_length / sr

    return change_times, change_frame_indices, centroid_diffs, threshold


def main():
    # 1) Signal laden
    y, sr = librosa.load(input_file, sr=None, mono=True)

    # 2a) Wechselstellen über Energie + Centroid (2D-Feature)
    change_times_all, change_frames_all, diffs_all, threshold_all = detect_change_points(
        y, sr,
        frame_duration=frame_duration,
        percentile=percentile,
        min_gap_s=min_gap_s
    )

    # 2b) Gegenprobe: Wechselstellen nur über Centroid-Differenz
    change_times_cent, change_frames_cent, diffs_cent, threshold_cent = detect_change_points_centroid_only(
        y, sr,
        frame_duration=frame_duration,
        percentile=percentile,
        min_gap_s=min_gap_s
    )

    print("Gefundene Wechselstellen (multi-Feature, s):")
    for t_c in change_times_all:
        print(f"{t_c:.3f}")
    print("\nGefundene Wechselstellen (Centroid-only, s):")
    for t_c in change_times_cent:
        print(f"{t_c:.3f}")

    # 3) Theoretische Segmentgrenzen alle true_segment_duration Sekunden
    total_duration = len(y) / sr
    segment_boundaries = np.arange(0, total_duration, true_segment_duration)

    plot(change_times_all,change_times_cent,segment_boundaries,diffs_all,diffs_cent,threshold_all,threshold_cent,sr,y)


def plot(change_times_all,change_times_cent,segment_boundaries,diffs_all,diffs_cent,threshold_all,threshold_cent,sr,y):    

    # 4) Plot 1: Signal + Wechselstellen + 50 ms-Grenzen
    t = np.arange(len(y)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, linewidth=0.7, label="Signal")

    # Vertikale Linien: rot = ursprüngliche Methode, grün = Gegenprobe
    for ct in change_times_all:
        plt.axvline(ct, color="red", alpha=0, linewidth=1.5)
    for ct in change_times_cent:
        plt.axvline(ct, color="green", alpha=1, linewidth=1.0)

    # 50 ms-„Soll“-Grenzen (theoretische Interleave-Grenzen)
    for bt in segment_boundaries:
        plt.axvline(bt, color="purple", alpha=0.2, linewidth=2.0)

    # Dummy-Linien für Legende
    plt.plot([], [], color="purple",  label=f"50 ms-Segmente ({true_segment_duration*1000:.0f} ms)")
    plt.plot([], [], color="red",   label="Wechsel (Energie + Centroid)")
    plt.plot([], [], color="green", label="Wechsel (Centroid-only)")

    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.title("Interleavtes Signal mit markierten Wechselstellen und 50 ms-Grenzen")
    plt.legend()
    plt.tight_layout()
    plt.show()

    '''
    # 5) Plot 2 – Change-Score (Vergleich beider Verfahren) + 50 ms-Grenzen
    frame_times = (np.arange(len(diffs_all)) + 1) * frame_duration

    plt.figure(figsize=(10, 4))
    plt.plot(frame_times, diffs_all,  label="Feature-Abstand (E+Centroid)")
    plt.plot(frame_times, diffs_cent, label="Centroid-Differenz (Gegenprobe)")

    plt.axhline(threshold_all,  linestyle="--", label="Threshold E+Centroid")
    plt.axhline(threshold_cent, linestyle=":",  label="Threshold Centroid-only")

    for ct in change_times_all:
        plt.axvline(ct, color="red", alpha=0.3)
    for ct in change_times_cent:
        plt.axvline(ct, color="green", alpha=0.3)

    # 50 ms-Grenzen in Score-Plot
    for bt in segment_boundaries:
        plt.axvline(bt, color="blue", alpha=0.2, linewidth=0.8)

    plt.xlabel("Zeit [s]")
    plt.ylabel("Score")
    plt.title("Change-Score zwischen Frames (mit 50 ms-Segmentgrenzen)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''


if __name__ == "__main__":
    main()
