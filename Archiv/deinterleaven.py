import numpy as np
import librosa
import matplotlib.pyplot as plt
import time  # für Zeitmessung
from pydub import AudioSegment   # <--- NEU


# ===== Parameter anpassen =====
input_file     = "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/interleaved_50ms.mp3"  # dein Testsignal
frame_duration = 0.005   # Sekunden pro Frame (Experimentell: 5 ms funktioniert)
percentile     = 80     # wie “streng” der Schwellwert ist (70–90 testen)
min_gap_s      = 0.009   # Minimalabstand zwischen Wechseln (z.B. 9 ms da min länge 10 ms sind)

# Dauer eines „echten“ Interleave-Segments (z.B. 50 ms)
true_segment_duration = 0.05   # 0.05 s = 50 ms


def compute_features(y, sr, frame_duration):
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
    Gegenprobe:
    - pro Frame nur den (z-normalisierten) spektralen Schwerpunkt betrachten
    - Differenz des Centroids zwischen Frame i und i+1
    - große Sprünge → Wechselstellen
    """
    centroids_norm, centroids_hz, frame_length, hop_length = compute_features(y, sr, frame_duration)


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

    return change_times, change_frame_indices, centroid_diffs, threshold,centroids_hz

def detect_change_points_energy_only(y, sr, frame_duration, percentile=80, min_gap_s=0.03):
    """
    Wechselstellen-Erkennung nur über Rahmenenergie:
    - pro Frame Energie berechnen
    - z-normalisieren
    - Differenz der Energien zwischen i und i+1
    - große Sprünge → Wechselstellen
    """
    frame_length = int(frame_duration * sr)
    hop_length = frame_length

    window = np.hanning(frame_length)

    energies = []
    for start in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[start:start + frame_length]
        frame_w = frame * window

        energy = np.mean(frame_w ** 2)
        energies.append(energy)

    energies = np.asarray(energies)

    # z-Normierung
    mean = energies.mean()
    std = energies.std() + 1e-9
    energies_norm = (energies - mean) / std

    # Absoluter Unterschied zwischen aufeinanderfolgenden Energien
    energy_diffs = np.abs(energies_norm[1:] - energies_norm[:-1])

    peaks, threshold = _peak_pick_from_diffs(
        energy_diffs,
        frame_duration=frame_duration,
        percentile=percentile,
        min_gap_s=min_gap_s
    )

    change_frame_indices = peaks + 1
    change_times = change_frame_indices * hop_length / sr

    return change_times, change_frame_indices, energy_diffs, threshold, energies

def detect_change_points_amplitude_jump(y, sr, percentile=95, min_gap_s=0.01):
    """
    Detektiert abrupte Sprünge im Zeitsignal:
    - |y[n] - y[n-1]| als 'Sprungmaß'
    - Peak-Picking + Minimalabstand
    """
    # Differenzen zwischen aufeinanderfolgenden Samples
    diffs = np.abs(y[1:] - y[:-1])

    # Wir tun so, als wäre jedes Sample ein "Frame" mit Dauer 1/sr
    frame_duration_samples = 1.0 / sr

    peaks, threshold = _peak_pick_from_diffs(
        diffs,
        frame_duration=frame_duration_samples,
        percentile=percentile,
        min_gap_s=min_gap_s,
    )

    change_sample_indices = peaks + 1
    change_times = change_sample_indices / sr

    return change_times, change_sample_indices, diffs, threshold

def detect_change_points_shape_change(y, sr, frame_duration, percentile=80, min_gap_s=0.03):
    """
    Detektiert Formänderungen des Signals:
    - Wir betrachten aufeinanderfolgende Frames
    - berechnen die (normierte) Korrelation zwischen ihnen
    - große 'Unähnlichkeit' (1 - corr) -> mögliche Wechselstelle
    """
    frame_length = int(frame_duration * sr)
    hop_length = frame_length

    window = np.hanning(frame_length)

    prev_frame = None
    diffs = []

    for start in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[start:start + frame_length] * window

        if prev_frame is not None:
            num = np.dot(prev_frame, frame)
            den = (np.linalg.norm(prev_frame) * np.linalg.norm(frame) + 1e-9)
            corr = num / den
            # Je kleiner corr, desto unähnlicher -> diff = 1 - corr
            diffs.append(1.0 - corr)

        prev_frame = frame

    diffs = np.asarray(diffs)

    if len(diffs) == 0:
        return np.array([]), np.array([]), diffs, 0.0

    peaks, threshold = _peak_pick_from_diffs(
        diffs,
        frame_duration=frame_duration,
        percentile=percentile,
        min_gap_s=min_gap_s,
    )

    # Achtung: diffs[k] gehört zur Übergangsstelle zwischen Frame k und k+1
    change_frame_indices = peaks + 1
    change_times = change_frame_indices * hop_length / sr

    return change_times, change_frame_indices, diffs, threshold

def find_joint_change_points(
    change_times_cent,
    change_times_energy,
    change_times_jump,
    change_times_shape,
    max_diff_s=0.01
):
    """
    Fasst Wechselstellen von 4 Detektoren zusammen und gibt nach Priorität
    gruppierte Zeitpunkte zurück.

    Priorität = Anzahl verschiedener Methoden - 1:
      0 -> nur 1 Methode hat dort einen Wechsel gefunden
      1 -> 2 Methoden stimmen überein
      2 -> 3 Methoden stimmen überein
      3 -> alle 4 Methoden stimmen überein

    Rückgabe:
      prio0_times, prio1_times, prio2_times, prio3_times  (jeweils np.array)
    """
    all_lists = [
        np.asarray(change_times_cent),
        np.asarray(change_times_energy),
        np.asarray(change_times_jump),
        np.asarray(change_times_shape),
    ]

    # Alle Punkte als (time, method_id) sammeln
    points = []
    for method_id, arr in enumerate(all_lists):
        for t in arr:
            points.append((float(t), method_id))

    if not points:
        empty = np.array([])
        return empty, empty, empty, empty

    # Nach Zeit sortieren
    points.sort(key=lambda x: x[0])

    # Cluster bilden: alle Punkte, die max_diff_s vom ersten im Cluster entfernt sind
    clusters = []
    current_cluster = [points[0]]
    cluster_start_time = points[0][0]

    for t, mid in points[1:]:
        if t - cluster_start_time <= max_diff_s:
            current_cluster.append((t, mid))
        else:
            clusters.append(current_cluster)
            current_cluster = [(t, mid)]
            cluster_start_time = t
    clusters.append(current_cluster)

    # Nach Priorität einsortieren
    prio_lists = {0: [], 1: [], 2: [], 3: []}

    for cluster in clusters:
        times = [t for t, _ in cluster]
        methods = set(mid for _, mid in cluster)
        n_methods = len(methods)
        if n_methods == 0:
            continue
        prio = n_methods - 1  # 1 Methode -> 0, 4 Methoden -> 3
        prio = max(0, min(3, prio))

        t_mean = float(np.mean(times))
        prio_lists[prio].append(t_mean)

    prio0_times = np.asarray(prio_lists[0])
    prio1_times = np.asarray(prio_lists[1])
    prio2_times = np.asarray(prio_lists[2])
    prio3_times = np.asarray(prio_lists[3])

    return prio0_times, prio1_times, prio2_times, prio3_times





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

def reconstruct_source(y, sr, segments, label="A"):
    """
    Baut aus allen Segmenten mit dem gegebenen Label (z.B. 'A')
    ein neues Signal auf, indem die entsprechenden Zeitabschnitte
    aus y hintereinander geklebt werden.
    """
    parts = []

    for seg in segments:
        if seg["label"] != label:
            continue

        # Zeiten -> Sample-Indizes
        start_sample = int(round(seg["start_time"] * sr))
        end_sample   = int(round(seg["end_time"]   * sr))

        # Sicherheitscheck
        start_sample = max(0, min(start_sample, len(y)))
        end_sample   = max(0, min(end_sample,   len(y)))

        if end_sample > start_sample:
            parts.append(y[start_sample:end_sample])

    if not parts:
        return np.array([], dtype=y.dtype)

    return np.concatenate(parts)

def reconstruct(y,sr,segments,label):
    # === Rekonstruktion eines deinterleavten Signals (z.B. A) ===
    signal = reconstruct_source(y, sr, segments, label)
    print(f"\nRekonstruierte Länge von Signal "+ label+": {len(signal)} Samples "
          f"≈ {len(signal) / sr:.3f} s")

    if len(signal) > 0:
        # --- Als MP3 speichern ---
        max_val = np.max(np.abs(signal)) + 1e-9
        sig_norm = signal / max_val          # auf [-1,1] normieren
        audio_int16 = (sig_norm * 32767).astype(np.int16)

        audio_seg = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,   # 2 Byte = 16 Bit
            channels=1
        )
        audio_seg.export("deinterleaved_"+ label+".mp3", format="mp3")
        print("Deinterleavtes Signal "+ label+" als 'deinterleaved_"+ label+".mp3' gespeichert.")

        # --- Deinterleavtes Signal A plotten ---
        t_A = np.arange(len(signal)) / sr
        plt.figure(figsize=(10, 3))
        plt.plot(t_A, signal, linewidth=0.8)
        plt.xlabel("Zeit [s]")
        plt.ylabel("Amplitude")
        plt.title("Deinterleavtes Signal "+ label+"")
        plt.tight_layout()
        plt.show()
    else:
        print("Hinweis: Es wurden keine Segmente mit Label '"+ label+"' gefunden.")


def plot(change_times_cent, change_times_energy,change_times_jump,change_times_shape,
         segment_boundaries, diffs_cent, threshold_cent, sr, y, segments):
    # Plot 1: Signal + Wechselstellen + 50 ms-Grenzen
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y, linewidth=0.7, label="Signal")

    # Vertikale Linien: Wechselstellen (Centroid-only, Energie-only, gemeinsam)
    for ct in change_times_cent:
        ax.axvline(ct, color="green", alpha=1, linewidth=3.0)

    for ct in change_times_energy:
        ax.axvline(ct, color="pink", alpha=1, linewidth=3.0)

    for ct in change_times_jump:
        ax.axvline(ct, color="blue", alpha=1, linewidth=2.0)

    for ct in change_times_shape:
        ax.axvline(ct, color="orange", alpha=1, linewidth=1.0)



    # 50 ms-„Soll“-Grenzen (theoretische Interleave-Grenzen)
    for bt in segment_boundaries:
        ax.axvline(bt, color="purple", alpha=0.2, linewidth=2.0)

    # Zwischen den Wechselpunkten: Buchstaben A, B, C, ...
    # y-Position für Text: etwas unterhalb des oberen Randes
    y_min, y_max = ax.get_ylim()
    y_text = y_min + 0.8 * (y_max - y_min)

    for seg in segments:
        x_mid = 0.5 * (seg["start_time"] + seg["end_time"])
        ax.text(
            x_mid,
            y_text,
            seg["label"],
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6)
        )

    # Dummy-Linien für Legende
    ax.plot([], [], color="purple", label=f"50 ms-Segmente ({true_segment_duration*1000:.0f} ms)")
    ax.plot([], [], color="green",  label="Wechsel (Centroid-only)")
    ax.plot([], [], color="Pink",    label="Wechsel (Energie-only)")
    ax.plot([], [], color="blue",  label="Wechsel (Amplitude-Sprung)")
    ax.plot([], [], color="orange", label="Wechsel (Formänderung)")


    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Interleavtes Signal mit Wechselstellen, 50 ms-Grenzen und Signal-Labels")
    ax.legend()
    fig.tight_layout()
    plt.show()
    
def plot_priority_changes(y, sr, prio1_times, prio2_times, prio3_times):
    """
    Plottet das Signal und nur die Wechselstellen mit Priorität 1–3:
      Prio 1 -> rot
      Prio 2 -> orange
      Prio 3 -> grün
    """
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y, linewidth=0.7, label="Signal")

    # Prio 1: rot
    for ct in prio1_times:
        ax.axvline(ct, color="red", alpha=0.9, linewidth=1.2)

    # Prio 2: orange
    for ct in prio2_times:
        ax.axvline(ct, color="orange", alpha=0.9, linewidth=1.5)

    # Prio 3: grün
    for ct in prio3_times:
        ax.axvline(ct, color="green", alpha=0.9, linewidth=1.8)

    # Legende (Dummy-Linien)
    ax.plot([], [], color="red",    label="Wechsel Prio 1 (2 Methoden)")
    ax.plot([], [], color="orange", label="Wechsel Prio 2 (3 Methoden)")
    ax.plot([], [], color="green",  label="Wechsel Prio 3 (4 Methoden)")

    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal mit priorisierten Wechselstellen (Prio 1–3)")
    ax.legend()
    fig.tight_layout()
    plt.show()


def main():
    # === Zeitmessung: Start Gesamt ===
    t0 = time.perf_counter()

    # 1) Signal laden
    y, sr = librosa.load(input_file, sr=None, mono=True)

###############################################################################################################################################
    # 2b) Wechselstellen nach Schwerpunkt
    change_times_cent, change_frames_cent, diffs_cent, threshold_cent, centroids_hz = \
        detect_change_points_centroid_only(
            y, sr,
            frame_duration=frame_duration,
            percentile=percentile,
            min_gap_s=min_gap_s
        )

    # 2c) Wechselstellen nach Energie (falls du die Methode schon drin hast)
    (change_times_energy,
     change_frames_energy,
     diffs_energy,
     threshold_energy,
     energies) = detect_change_points_energy_only(
        y, sr,
        frame_duration=frame_duration,
        percentile=percentile,
        min_gap_s=min_gap_s
    )

    # 2d) Sample-Sprünge
    change_times_jump, change_samples_jump, diffs_jump, threshold_jump = \
        detect_change_points_amplitude_jump(
            y, sr,
            percentile=98,      # meist etwas strenger
            min_gap_s=0.005     # z.B. 5 ms Mindestabstand
        )

    # 2e) Formänderungen
    change_times_shape, change_frames_shape, diffs_shape, threshold_shape = \
        detect_change_points_shape_change(
            y, sr,
            frame_duration=frame_duration,
            percentile=percentile,
            min_gap_s=min_gap_s
        )

###############################################################################################################################################
    
    # === Priorisierte Wechselstellen aus allen 4 Detektoren ===
    prio0_times, prio1_times, prio2_times, prio3_times = find_joint_change_points(
        change_times_cent,
        change_times_energy,
        change_times_jump,
        change_times_shape,
        max_diff_s=0.005  # 10 ms
    )

    print(f"\nWechselstellen nach Priorität:")
    print(f"  Prio 0 (nur 1 Methode): {len(prio0_times)}")
    print(f"  Prio 1 (2 Methoden):    {len(prio1_times)}")
    print(f"  Prio 2 (3 Methoden):    {len(prio2_times)}")
    print(f"  Prio 3 (4 Methoden):    {len(prio3_times)}")


    # 3) Theoretische Segmentgrenzen alle true_segment_duration Sekunden
    total_duration = len(y) / sr
    segment_boundaries = np.arange(0, total_duration, true_segment_duration)



#--------------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------------


    # === Zeitmessung: Ende Rechenanteil (ohne Plot) ===
    t_before_plot = time.perf_counter()





    plot(
        change_times_cent,
        change_times_energy,
        change_times_jump,
        change_times_shape, 
        segment_boundaries, 
        diffs_cent, 
        threshold_cent, 
        sr, 
        y, 
        segments
    )


    # Zweiter Plot: Nur priorisierte Wechselstellen
    plot_priority_changes(y, sr, prio1_times, prio2_times, prio3_times)




    # === Zeitmessung: reconstruct ===
    t_recon = time.perf_counter()

    reconstruct(y,sr,segments,"A")
    reconstruct(y,sr,segments,"B")
    reconstruct(y,sr,segments,"C")

    # === Zeitmessung: Ende Gesamt ===
    t_end = time.perf_counter()

    gesamt_zeit = t_end - t0
    rechen_zeit = t_before_plot - t0
    plot_zeit   = t_end - t_before_plot
    reconstruct_zeit = t_end-t_recon

    print(f"\nZeitmessung:")
    print(f"  Gesamtzeit (inkl. Plot):       {gesamt_zeit:.4f} s")
    print(f"  Rechenzeit (ohne Plot):        {rechen_zeit:.4f} s")
    print(f"  Plotzeit (inkl. plt.show()):   {plot_zeit:.4f} s")
    print(f"  rekonstruktionszeit (ohne Plot):        {reconstruct_zeit:.4f} s")
    print(f"  Deinterleavenzeit:        {(reconstruct_zeit+rechen_zeit):.4f} s")




    
if __name__ == "__main__":
    main()
