import numpy as np
import librosa
import matplotlib.pyplot as plt
import time  # für Zeitmessung
from pydub import AudioSegment   # <--- NEU


#
#Dieser code ist Experimentell und nicht echtzeitfähig
#
#Die eingegeben Werte sind experimentell eingestellt
#
#



# ===== KONFIGURATION / DREHSCHRAUBEN =====

# Pfad zur interleavten MP3-Datei
INPUT_FILE = "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/interleaved_1k_8k_vio_rand.mp3"

# Länge eines Analyse-Frames in Sekunden ( 0.001 = 1 ms/ Achtung für signale unter 1000 Hz wird 1 ms evtl. zu knapp für eine richtige erkennung)
FRAME_DURATION_S = 0.002

# Perzentil für Schwellenwerte bei Centroid- und Energie-Detektion
# höherer Wert -> weniger, aber stärkere Wechselstellen
PERCENTILE_FEATURE = 80

# Minimalabstand zwischen Wechselstellen innerhalb EINER Methode (Sekunden)
MIN_GAP_S = 0.01

# „Soll“-Segmentlänge der Interleaves (nur für Referenzlinien im Plot)
TRUE_SEGMENT_DURATION_S = 0.05  # 0.05 s = 50 ms

# Spezielle Einstellungen für die Energie-Detektion
ENERGY_PERCENTILE = PERCENTILE_FEATURE     # ggf. separat anpassen
ENERGY_MIN_GAP_S  = MIN_GAP_S

# Einstellungen für die Formänderungs-Detektion (ShapeChange)
SHAPE_PERCENTILE = PERCENTILE_FEATURE
SHAPE_MIN_GAP_S  = MIN_GAP_S

# Einstellungen für Amplitudensprünge (Sample-Differenzen)
JUMP_PERCENTILE = 98       # hoch -> nur sehr starke Sprünge
JUMP_MIN_GAP_S  = 0.005    # Minimalabstand zwischen Sprung-Wechselstellen

# Zeitfenster, in dem verschiedene Methoden als „gleicher“ Wechsel zählen (Sekunden)
JOINT_MAX_DIFF_S = 0.002   # z.B. 5 ms

# Clustering der Segmente nach spektralem Schwerpunkt
CLUSTER_THRESHOLD_HZ     = 500.0  # max. erlaubter Abstand zwischen Cluster-Mittelpunkten
MIN_SEGMENTS_PER_CLUSTER = 3      # min. Anzahl Segmente, damit ein Cluster als „Signal“ zählt

# Welche Prioritäten für Segmentgrenzen genutzt werden:
# False: nur Prio 2+3; True: Prio 1–3
USE_PRIO1_IN_BOUNDARIES = True

#--------------Generelle Berechnungen für die Methoden--------------------------------------------------------
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

def compute_frame_features(y, sr, frame_duration):
    L = int(frame_duration * sr)
    H = L
    win = np.hanning(L)

    feats = []  # Liste von Vektoren pro Frame
    for start in range(0, len(y) - L + 1, H):
        f = y[start:start+L] * win

        # Grundgrößen
        spec = np.fft.rfft(f)
        mag  = np.abs(spec) + 1e-12
        freqs = np.fft.rfftfreq(L, 1/sr)

        energy = np.mean(f**2)
        centroid = (freqs * mag).sum() / mag.sum()
        # Spektralbandbreite um den Schwerpunkt
        bandwidth = np.sqrt(((freqs - centroid)**2 * mag).sum() / mag.sum())
        # 85%-Rolloff
        cumsum = np.cumsum(mag)
        rolloff = freqs[np.searchsorted(cumsum, 0.85 * cumsum[-1])]
        # Flatness (Geometrisch/Arithmetisch)
        flatness = np.exp(np.mean(np.log(mag))) / np.mean(mag)
        # Zero-Crossing-Rate
        zcr = np.mean(np.signbit(f[1:]) != np.signbit(f[:-1]))

        # Grobe MFCC-Proxy (2–4 grobe Melbänder, ohne Librosa-MFCC):
        # (Optional – schnell & simpel)
        n_bands = 4
        edges = np.linspace(0, mag.size-1, n_bands+1, dtype=int)
        melbands = [mag[edges[i]:edges[i+1]].mean() for i in range(n_bands)]

        vec = [energy, centroid, bandwidth, rolloff, flatness, zcr] + melbands
        feats.append(vec)

    F = np.asarray(feats, dtype=float)
    # z-Normierung spaltenweise
    mu = F.mean(axis=0)
    sd = F.std(axis=0) + 1e-9
    Fz = (F - mu) / sd
    return Fz  # Shape: (n_frames, d)

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


#---------------methoden zur erkennung: Spektralschwerpunkt, Energie, Amplitudensprünge, Formänderung, Überlappung der change_points----------------------------------------------------------------
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

    - Priorität richtet sich NUR nach der Anzahl VERSCHIEDENER Methoden im Fenster.
    - Mehrere Treffer derselben Methode innerhalb des Fensters zählen nur als 1.
    - Joint-Zeit = Mittelwert über je einen repräsentativen Zeitpunkt pro Methode.

    Priorität = Anzahl verschiedener Methoden - 1:
      0 -> nur 1 Methode hat dort einen Wechsel gefunden
      1 -> 2 Methoden stimmen überein
      2 -> 3 Methoden stimmen überein
      3 -> alle 4 Methoden stimmen überein

    Rückgabe:
      prio0_times, prio1_times, prio2_times, prio3_times  (np.array)
      prio0_methods, prio1_methods, prio2_methods, prio3_methods  (Liste von Listen mit method_ids)
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
        empty_arr = np.array([])
        empty_list = []
        return (empty_arr, empty_arr, empty_arr, empty_arr,
                empty_list, empty_list, empty_list, empty_list)

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
    prio_times = {0: [], 1: [], 2: [], 3: []}
    prio_methods = {0: [], 1: [], 2: [], 3: []}

    for cluster in clusters:
        # Zeiten pro Methode sammeln
        method_to_times = {}
        for t, mid in cluster:
            method_to_times.setdefault(mid, []).append(t)

        method_set = set(method_to_times.keys())
        n_methods = len(method_set)
        if n_methods == 0:
            continue

        # NEU: Fälle mit genau 2 Methoden, die NUR Energy + AmplitudeJump sind,
        # werden komplett ignoriert.
        # (Energy = 1, AmplitudeJump = 2)
        if n_methods == 2 and method_set == {1, 2}:
            continue

        prio = n_methods - 1
        if prio < 0:
            continue
        prio = min(3, prio)

        # Für jede Methode einen repräsentativen Zeitpunkt (z.B. Mittelwert)
        rep_times = [float(np.mean(ts)) for ts in method_to_times.values()]
        t_mean = float(np.mean(rep_times))

        prio_times[prio].append(t_mean)
        prio_methods[prio].append(sorted(method_set))


    prio0_times = np.asarray(prio_times[0])
    prio1_times = np.asarray(prio_times[1])
    prio2_times = np.asarray(prio_times[2])
    prio3_times = np.asarray(prio_times[3])

    prio0_methods = prio_methods[0]
    prio1_methods = prio_methods[1]
    prio2_methods = prio_methods[2]
    prio3_methods = prio_methods[3]

    return (prio0_times, prio1_times, prio2_times, prio3_times,
            prio0_methods, prio1_methods, prio2_methods, prio3_methods)


#-----------Zuordnung und neu zusammensetzen-----------------------------------------------------------------------
def assign_segments_to_sources(centroids_hz,
                               frame_duration,
                               change_frame_indices,
                               cluster_threshold_hz=CLUSTER_THRESHOLD_HZ,
                               min_segments_per_cluster=MIN_SEGMENTS_PER_CLUSTER,
                               frame_feats=None):  # <— NEU
    import numpy as np

    n_frames = len(centroids_hz)
    boundaries = np.concatenate(([0], change_frame_indices, [n_frames]))

    segments = []
    seg_centroids = []
    seg_featvecs = []  # <— NEU: Segment-Featurevektoren

    for seg_idx in range(len(boundaries) - 1):
        s = int(boundaries[seg_idx])
        e = int(boundaries[seg_idx + 1])
        if e <= s:
            continue

        # 1D-Referenz: mittlerer Spektralschwerpunkt (nur für Output/Labeling)
        seg_c = float(centroids_hz[s:e].mean())
        seg_centroids.append(seg_c)

        # Mehrmerkmals-Vektor (z-normierte Frame-Features werden gemittelt)
        if frame_feats is not None:
            seg_featvecs.append(frame_feats[s:e].mean(axis=0))
        else:
            seg_featvecs.append(np.array([seg_c], dtype=float))  # Fallback: 1D

        segments.append({
            "segment_index": seg_idx,
            "start_frame": s, "end_frame": e,
            "start_time": s * frame_duration, "end_time": e * frame_duration,
            "mean_centroid_hz": seg_c,
        })

    if len(segments) == 0:
        return segments, np.array([])

    seg_centroids = np.asarray(seg_centroids, dtype=float)
    seg_featvecs = np.vstack(seg_featvecs)

    # --- Clustering (hierarchisches Mergen) im passenden Raum ---
    use_features = frame_feats is not None
    clusters = [[i] for i in range(len(segments))]
    cluster_vecs = [seg_featvecs[i].copy() for i in range(len(segments))]

    if use_features:
        MERGE_THRESH = 1.8  # Richtwert im z-normierten Feature-Raum (1.5–2.5 feintunen)
        def dist(a, b): return float(np.linalg.norm(a - b))
    else:
        MERGE_THRESH = float(cluster_threshold_hz)  # wie bisher im 1D-Centroid-Raum
        def dist(a, b): return abs(float(a[0] - b[0]))

    while True:
        if len(clusters) <= 1:
            break
        min_d = np.inf
        pair = None
        for a in range(len(clusters) - 1):
            for b in range(a + 1, len(clusters)):
                d = dist(cluster_vecs[a], cluster_vecs[b])
                if d < min_d:
                    min_d = d
                    pair = (a, b)
        if min_d > MERGE_THRESH or pair is None:
            break

        a, b = pair
        merged_idx = clusters[a] + clusters[b]
        merged_vec = seg_featvecs[merged_idx].mean(axis=0)

        for idx in sorted([a, b], reverse=True):
            del clusters[idx]
            del cluster_vecs[idx]
        clusters.append(merged_idx)
        cluster_vecs.append(merged_vec)

    # Kleine Cluster an große anhängen (nächstgelegene Distanz)
    sizes = np.array([len(c) for c in clusters])
    big_mask = sizes >= min_segments_per_cluster
    if not np.any(big_mask):
        big_mask[:] = True
    big_idx = np.where(big_mask)[0]
    small_idx = np.where(~big_mask)[0]

    for sidx in small_idx:
        dists = np.array([dist(cluster_vecs[sidx], cluster_vecs[b]) for b in big_idx])
        nb = big_idx[int(np.argmin(dists))]
        clusters[nb].extend(clusters[sidx])
        cluster_vecs[nb] = seg_featvecs[clusters[nb]].mean(axis=0)

    clusters = [clusters[i] for i in big_idx]

    # Labeling bleibt über den mittleren Centroid (A, B, C …)
    cluster_centroid_means = np.array([seg_centroids[c].mean() for c in clusters])
    order = np.argsort(cluster_centroid_means)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_per_cluster = {cid: (letters[r] if r < len(letters) else f"Cluster_{r}")
                         for r, cid in enumerate(order)}

    # Mapping in Segmente
    cluster_ids = np.empty(len(segments), dtype=int)
    for cid, idx_list in enumerate(clusters):
        for i in idx_list:
            cluster_ids[i] = cid
    for seg, cid in zip(segments, cluster_ids):
        seg["cluster_id"] = cid
        seg["label"] = label_per_cluster[cid]

    return segments, cluster_centroid_means

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

def segmente_zuordnen(centroids_hz, prio1_times, prio2_times, prio3_times, change_frames_cent, frame_feats):
    #    Grenzen sind jetzt die Wechselstellen mit Priorität 1–3
    n_frames = len(centroids_hz)

    
    if USE_PRIO1_IN_BOUNDARIES:
        # Prio 1–3 verwenden
        prio_times_all = np.concatenate([prio1_times, prio2_times, prio3_times]) if (
            len(prio1_times) + len(prio2_times) + len(prio3_times)
        ) > 0 else np.array([])
    else:
        # Nur Prio 2–3 verwenden
        prio_times_all = np.concatenate([prio2_times, prio3_times]) if (
            len(prio2_times) + len(prio3_times)
        ) > 0 else np.array([])

    frame_boundaries = []
    for t in prio_times_all:
        f_idx = int(round(t / FRAME_DURATION_S))
        if 0 < f_idx < n_frames:   # innerhalb gültiger Frame-Grenzen
            frame_boundaries.append(f_idx)

    if len(frame_boundaries) > 0:
        frame_boundaries = np.unique(frame_boundaries).astype(int)
    else:
        # Fallback: wenn gar keine prio 1–3 Wechsel gefunden wurden,
        # weiterhin die Centroid-Wechselstellen verwenden
        frame_boundaries = change_frames_cent

    segments, cluster_means = assign_segments_to_sources(
        centroids_hz=centroids_hz,
        frame_duration=FRAME_DURATION_S,
        change_frame_indices=frame_boundaries,
        cluster_threshold_hz=CLUSTER_THRESHOLD_HZ,
        min_segments_per_cluster=MIN_SEGMENTS_PER_CLUSTER,
        frame_feats=frame_feats  # --- NEU ---
    )

    print("\nSegment-Zuordnung (basierend auf priorisierten Wechselstellen):")
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

    return segments


#-----------Plot methoden---------------------------------------------------------------------

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

    # Dummy-Linien für Legende
    ax.plot([], [], color="purple", label=f"50 ms-Segmente ({TRUE_SEGMENT_DURATION_S*1000:.0f} ms)")
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
    
def plot_priority_changes(y, sr, prio1_times, prio2_times, prio3_times, segments):

    """
    Plottet das Signal und nur die Wechselstellen mit Priorität 1–3:
      Prio 1 -> rot
      Prio 2 -> orange
      Prio 3 -> grün
    """
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y, linewidth=0.7, label="Signal")

    if USE_PRIO1_IN_BOUNDARIES:
        # Prio 1: rot
        for ct in prio1_times:
            ax.axvline(ct, color="red", alpha=0.9, linewidth=1.2)

    # Prio 2: orange
    for ct in prio2_times:
        ax.axvline(ct, color="orange", alpha=0.9, linewidth=1.5)

    # Prio 3: grün
    for ct in prio3_times:
        ax.axvline(ct, color="green", alpha=0.9, linewidth=1.8)

    # Zwischen den priorisierten Wechselpunkten: Buchstaben A, B, C, ...
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


    # Legende (Dummy-Linien)
    if USE_PRIO1_IN_BOUNDARIES:
        ax.plot([], [], color="red",    label="Wechsel Prio 1 (2 Methoden)")
    ax.plot([], [], color="orange", label="Wechsel Prio 2 (3 Methoden)")
    ax.plot([], [], color="green",  label="Wechsel Prio 3 (4 Methoden)")

    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal mit priorisierten Wechselstellen (Prio 1–3)")
    #ax.legend()
    fig.tight_layout()
    plt.show()

def plot_close_priority_changes(y, sr,
                                prio1_times, prio2_times, prio3_times,
                                max_diff_s=0.01):
    """
    Plottet das Signal und alle Wechselstellen (Prio 1–3),
    die einen Nachbarn mit Abstand < max_diff_s Sekunden haben.
    """
    # Alle Prio-1..3-Punkte mit ihrer Prio einsammeln
    all_points = []
    for t in prio1_times:
        all_points.append((float(t), 1))
    for t in prio2_times:
        all_points.append((float(t), 2))
    for t in prio3_times:
        all_points.append((float(t), 3))

    if not all_points:
        print("\nKeine Wechselstellen (Prio 1–3) vorhanden.")
        return

    # Nach Zeit sortieren
    all_points.sort(key=lambda x: x[0])

    # Punkte finden, die einen Nachbarn < max_diff_s haben
    close_points = []
    n = len(all_points)
    for i, (t, p) in enumerate(all_points):
        is_close = False
        if i > 0 and abs(t - all_points[i - 1][0]) < max_diff_s:
            is_close = True
        if i < n - 1 and abs(all_points[i + 1][0] - t) < max_diff_s:
            is_close = True
        if is_close:
            close_points.append((t, p))

    if not close_points:
        print(f"\nKeine Prio-1–3-Wechselstellen mit Nachbarn < {max_diff_s*1000:.1f} ms gefunden.")
        return

    print(f"\nWechselstellen (Prio 1–3) mit Nachbarn < {max_diff_s*1000:.1f} ms:")
    for t, p in close_points:
        print(f"  t = {t:.4f} s, Prio {p}")

    # Plot
    t_axis = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_axis, y, linewidth=0.7, label="Signal")

    for t, p in close_points:
        if p == 1:
            color = "red"
        elif p == 2:
            color = "orange"
        else:  # p == 3
            color = "green"
        ax.axvline(t, color=color, alpha=0.9, linewidth=1.5)

    # Dummy-Linien für Legende
    ax.plot([], [], color="red",    label="Prio 1 (2 Methoden), Nachbar < 10 ms")
    ax.plot([], [], color="orange", label="Prio 2 (3 Methoden), Nachbar < 10 ms")
    ax.plot([], [], color="green",  label="Prio 3 (4 Methoden), Nachbar < 10 ms")

    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal mit nah beieinanderliegenden Wechselstellen (Prio 1–3)")
    ax.legend()
    fig.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------


def main():
    # === Zeitmessung: Start Gesamt ===
    t0 = time.perf_counter()

    # 1) Signal laden
    y, sr = librosa.load(INPUT_FILE, sr=None, mono=True)


###############################################################################################################################################
    # 2b) Wechselstellen nach Schwerpunkt
    change_times_cent, change_frames_cent, diffs_cent, threshold_cent, centroids_hz = \
        detect_change_points_centroid_only(
            y, sr,
            frame_duration=FRAME_DURATION_S,
            percentile=PERCENTILE_FEATURE,
            min_gap_s=MIN_GAP_S
        )


    # 2c) Wechselstellen nach Energie (falls du die Methode schon drin hast)
    (change_times_energy,
    change_frames_energy,
    diffs_energy,
    threshold_energy,
    energies) = detect_change_points_energy_only(
        y, sr,
        frame_duration=FRAME_DURATION_S,
        percentile=ENERGY_PERCENTILE,
        min_gap_s=ENERGY_MIN_GAP_S
    )


    # 2d) Sample-Sprünge
    change_times_jump, change_samples_jump, diffs_jump, threshold_jump = \
        detect_change_points_amplitude_jump(
            y, sr,
            percentile=JUMP_PERCENTILE,
            min_gap_s=JUMP_MIN_GAP_S
        )

    # 2e) Formänderungen
    change_times_shape, change_frames_shape, diffs_shape, threshold_shape = \
        detect_change_points_shape_change(
            y, sr,
            frame_duration=FRAME_DURATION_S,
            percentile=SHAPE_PERCENTILE,
            min_gap_s=SHAPE_MIN_GAP_S
        )



###############################################################################################################################################
    
    # === Priorisierte Wechselstellen aus allen 4 Detektoren ===
    (prio0_times, prio1_times, prio2_times, prio3_times,
    prio0_methods, prio1_methods, prio2_methods, prio3_methods) = find_joint_change_points(
        change_times_cent,
        change_times_energy,
        change_times_jump,
        change_times_shape,
        max_diff_s=JOINT_MAX_DIFF_S
    )

    method_names = {
        0: "Centroid",
        1: "Energy",
        2: "AmplitudeJump",
        3: "ShapeChange",
    }

    print(f"\nWechselstellen nach Priorität:")

    def _print_prio_info(prio_times, prio_methods, prio_label):
        print(f"\n{prio_label} – {len(prio_times)} Stellen:")
        for t, mids in zip(prio_times, prio_methods):
            names = ", ".join(method_names[m] for m in mids)
            print(f"  t = {t:.4f} s  ->  {names}")

    _print_prio_info(prio1_times, prio1_methods, "Prio 1 (2 Methoden)")
    _print_prio_info(prio2_times, prio2_methods, "Prio 2 (3 Methoden)")
    _print_prio_info(prio3_times, prio3_methods, "Prio 3 (4 Methoden)")


    # 3) Theoretische Segmentgrenzen alle true_segment_duration Sekunden
    total_duration = len(y) / sr
    segment_boundaries = np.arange(0, total_duration, TRUE_SEGMENT_DURATION_S)



#--------------------------------------------------------------------------------------
    # 4) Segmente den Quell-Signalen A, B, C, ... zuordnen

    # --- NEU: Mehrmerkmals-Frames ---
    frame_feats = compute_frame_features(y, sr, FRAME_DURATION_S)

    segments = segmente_zuordnen(centroids_hz, prio1_times, prio2_times, prio3_times, change_frames_cent, frame_feats)

#--------------------------------------------------------------------------------------

    # === Zeitmessung: Ende Rechenanteil (ohne Plot) ===
    t_before_plot = time.perf_counter()

#--------------------------------------------------------------------------------------
    # === Plots === 
    #plot(change_times_cent, change_times_energy, change_times_jump, change_times_shape, segment_boundaries,diffs_cent, threshold_cent, sr, y, segments)

    # Zweiter Plot: Nur priorisierte Wechselstellen + Segment-Labels A/B/C
    plot_priority_changes(y, sr, prio1_times, prio2_times, prio3_times, segments)

    # Dritter Plot: Prio-1–3-Wechselstellen mit Nachbarn < 10 ms
    #plot_close_priority_changes(y, sr, prio1_times, prio2_times, prio3_times, max_diff_s=0.010)
#--------------------------------------------------------------------------------------



    # === Zeitmessung: reconstruct ===
    t_recon = time.perf_counter()


#--------------------------------------------------------------------------------------
    # === Recontruieren aller erkannten Signale ===
    labels = sorted({seg["label"] for seg in segments})
    for lab in labels:
        reconstruct(y, sr, segments, lab)
#--------------------------------------------------------------------------------------


    # === Zeitmessung: Ende Gesamt ===
    t_end = time.perf_counter()

    # === berechnung Zeiten ===
    gesamt_zeit = t_end - t0
    rechen_zeit = t_before_plot - t0
    plot_zeit   = t_end - t_before_plot
    reconstruct_zeit = t_end-t_recon

    # === Ausgabe der Zeitmessungen ===
    print(f"\nZeitmessung:")
    print(f"  Gesamtzeit (inkl. Plot):       {gesamt_zeit:.4f} s")
    print(f"  Rechenzeit (ohne Plot):        {rechen_zeit:.4f} s")
    print(f"  Plotzeit (inkl. plt.show()):   {plot_zeit:.4f} s")
    print(f"  rekonstruktionszeit (ohne Plot):        {reconstruct_zeit:.4f} s")
    print(f"  Deinterleavenzeit:        {(reconstruct_zeit+rechen_zeit):.4f} s")


    
if __name__ == "__main__":
    main()
