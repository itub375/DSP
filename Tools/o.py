import numpy as np
import librosa
import matplotlib.pyplot as plt
import time  # für Zeitmessung

# ===== Parameter anpassen =====
input_file     = "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/interleaved_50ms.mp3"  # dein Testsignal
frame_duration = 0.001   # Sekunden pro Frame
percentile     = 90      # wie “streng” der Schwellwert ist (70–90 testen)
min_gap_s      = 0.01    # Minimalabstand zwischen Wechseln (z.B. 10 ms)

# Dauer eines „echten“ Interleave-Segments (z.B. 50 ms)
true_segment_duration = 0.05   # 0.05 s = 50 ms


def compute_centroids(y, sr, frame_duration):
    """
    Berechnet pro Frame den spektralen Schwerpunkt.

    Rückgabe:
      centroids_norm : z-normalisierte Centroids pro Frame
      centroids_hz   : Centroids in Hz pro Frame
      frame_length, hop_length : Längen in Samples
    """
    frame_length = int(frame_duration * sr)
    hop_length = frame_length

    window = np.hanning(frame_length)

    centroids = []

    for start in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[start:start + frame_length]
        frame_w = frame * window

        # FFT
        spec = np.fft.rfft(frame_w)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(frame_length, 1 / sr)

        if mag.sum() == 0:
            centroid = 0.0
        else:
            centroid = (freqs * mag).sum() / mag.sum()

        centroids.append(centroid)

    centroids_hz = np.asarray(centroids)

    # z-Normierung
    mean = centroids_hz.mean()
    std = centroids_hz.std() + 1e-9
    centroids_norm = (centroids_hz - mean) / std

    return centroids_norm, centroids_hz, frame_length, hop_length


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


def detect_change_points_centroid_only(y, sr, frame_duration, percentile=80, min_gap_s=0.03):
    """
    Wechselstellen-Erkennung nur über spektralen Schwerpunkt:
    - pro Frame den z-normalisierten Centroid
    - Differenz des Centroids zwischen Frame i und i+1
    - große Sprünge → Wechselstellen
    """
    centroids_norm, centroids_hz, frame_length, hop_length = compute_centroids(y, sr, frame_duration)

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

    return change_times, change_frame_indices, centroid_diffs, threshold, centroids_hz


def assign_segments_to_sources(centroids_hz, frame_duration, change_frame_indices,
                               cluster_threshold_hz=500.0):
    """
    Ordnet die zwischen den Wechselstellen liegenden Abschnitte (Segmente)
    verschiedenen Quell-Signalen A, B, C, ... zu.

    Basis: mittlerer spektraler Schwerpunkt pro Segment.
    """
    n_frames = len(centroids_hz)

    # Frame-Grenzen -> Segment-Grenzen in Frames
    boundaries = np.concatenate(([0], change_frame_indices, [n_frames]))

    segments = []
    seg_centroids = []

    for seg_idx in range(len(boundaries) - 1):
        start_f = int(boundaries[seg_idx])
        end_f   = int(boundaries[seg_idx + 1])

        if end_f <= start_f:
            continue  # Sicherheitscheck

        seg_c = float(centroids_hz[start_f:end_f].mean())
        seg_centroids.append(seg_c)

        segments.append({
            "segment_index": seg_idx,
            "start_frame": start_f,
            "end_frame": end_f,
            "start_time": start_f * frame_duration,
            "end_time":   end_f   * frame_duration,
            "mean_centroid_hz": seg_c,
        })

    seg_centroids = np.asarray(seg_centroids)

    # --- Clustering der Segmente nach Schwerpunkt ---
    cluster_means = []   # in Hz
    cluster_counts = []  # wie viele Segmente bisher in jedem Cluster
    cluster_ids = []     # pro Segment: Index in cluster_means

    for seg_c in seg_centroids:
        if len(cluster_means) == 0:
            # erster Cluster -> A
            cluster_means.append(seg_c)
            cluster_counts.append(1)
            cluster_ids.append(0)
            continue

        dists = np.abs(np.asarray(cluster_means) - seg_c)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        if best_dist <= cluster_threshold_hz:
            # zu existierender Quelle hinzufügen
            cluster_ids.append(best_idx)
            cluster_counts[best_idx] += 1
            # laufenden Mittelwert aktualisieren
            n = cluster_counts[best_idx]
            cluster_means[best_idx] = cluster_means[best_idx] + (seg_c - cluster_means[best_idx]) / n
        else:
            # neue Quelle (neuer Buchstabe)
            cluster_means.append(seg_c)
            cluster_counts.append(1)
            cluster_ids.append(len(cluster_means) - 1)

    # Quellen-Buchstaben zuweisen
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_per_cluster = {}
    for cid in sorted(set(cluster_ids)):
        if cid < len(letters):
            label_per_cluster[cid] = letters[cid]
        else:
            label_per_cluster[cid] = f"Cluster_{cid}"

    # Labels in Segment-Struktur schreiben
    for seg, cid in zip(segments, cluster_ids):
        seg["cluster_id"] = cid
        seg["label"] = label_per_cluster[cid]

    return segments, cluster_means


def plot(change_times_cent, segment_boundaries, sr, y):
    # Plot: Signal + Wechselstellen + 50 ms-Grenzen
    t = np.arange(len(y)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, linewidth=0.7, label="Signal")

    # Vertikale Linien: Wechselstellen (Centroid-only)
    for ct in change_times_cent:
        plt.axvline(ct, color="green", alpha=1, linewidth=1.0)

    # 50 ms-„Soll“-Grenzen (theoretische Interleave-Grenzen)
    for bt in segment_boundaries:
        plt.axvline(bt, color="purple", alpha=0.2, linewidth=2.0)

    # Dummy-Linien für Legende
    plt.plot([], [], color="purple",  label=f"50 ms-Segmente ({true_segment_duration*1000:.0f} ms)")
    plt.plot([], [], color="green", label="Wechsel (Centroid-only)")

    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.title("Interleavtes Signal mit markierten Wechselstellen und 50 ms-Grenzen")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # === Zeitmessung: Start Gesamt ===
    t0 = time.perf_counter()

    # 1) Signal laden
    y, sr = librosa.load(input_file, sr=None, mono=True)

    # 2) Wechselstellen nur über Centroid-Differenz
    (change_times_cent,
     change_frames_cent,
     diffs_cent,
     threshold_cent,
     centroids_hz) = detect_change_points_centroid_only(
        y, sr,
        frame_duration=frame_duration,
        percentile=percentile,
        min_gap_s=min_gap_s
    )

    print("\nGefundene Wechselstellen (Centroid-only, s):")
    for t_c in change_times_cent:
        print(f"{t_c:.3f}")

    # 3) Theoretische Segmentgrenzen alle true_segment_duration Sekunden
    total_duration = len(y) / sr
    segment_boundaries = np.arange(0, total_duration, true_segment_duration)

    # 4) Segmente den Quell-Signalen A, B, C, ... zuordnen
    segments, cluster_means = assign_segments_to_sources(
        centroids_hz=centroids_hz,
        frame_duration=frame_duration,
        change_frame_indices=change_frames_cent,
        cluster_threshold_hz=500.0  # hier ggf. anpassen
    )

    print("\nSegment-Zuordnung (basierend auf spektralem Schwerpunkt):")
    for seg in segments:
        print(
            f"Segment {seg['segment_index']:2d}: "
            f"{seg['start_time']:.3f}s - {seg['end_time']:.3f}s | "
            f"Centroid ≈ {seg['mean_centroid_hz']:.0f} Hz -> Signal {seg['label']}"
        )

    print("\nMittlere Schwerpunkte der gefundenen Quellen:")
    for cid, mean_c in enumerate(cluster_means):
        label = "?"
        for seg in segments:
            if seg["cluster_id"] == cid:
                label = seg["label"]
                break
        print(f"Quelle {label}: ca. {mean_c:.0f} Hz")

    # === Anzahl erkannter Signale ausgeben ===
    n_signale = len(cluster_means)
    print(f"\nAnzahl erkannter unterschiedlicher Signale (Cluster): {n_signale}")

    # === Zeitmessung: Ende Rechenanteil (ohne Plot) ===
    t_before_plot = time.perf_counter()

    # 5) Plotten
    plot(change_times_cent, segment_boundaries, sr, y)

    # === Zeitmessung: Ende Gesamt ===
    t_end = time.perf_counter()

    gesamt_zeit = t_end - t0
    rechen_zeit = t_before_plot - t0
    plot_zeit   = t_end - t_before_plot

    print(f"\nZeitmessung:")
    print(f"  Gesamtzeit (inkl. Plot):       {gesamt_zeit:.4f} s")
    print(f"  Rechenzeit (ohne Plot):        {rechen_zeit:.4f} s")
    print(f"  Plotzeit (inkl. plt.show()):   {plot_zeit:.4f} s")


if __name__ == "__main__":
    main()
