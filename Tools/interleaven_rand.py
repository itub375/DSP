from pydub import AudioSegment
import random
import csv
import os
# ===== EINSTELLUNGEN =====
TARGET_DURATION_MS = 60_000    # Ziel-Länge pro Spur (z.B. 10_000 für 10 s)
MIN_BLOCK_MS = 10             # minimale Blocklänge in ms
MAX_BLOCK_MS = 50             # maximale Blocklänge in ms
OUTPUT_FILE = "Inputsignals/rand/interleaved_pod_1k_60sec_rand.mp3"
CSV_FILE = os.path.splitext(OUTPUT_FILE)[0] + "_wechselstellen.csv"
# HIER deine MP3-Dateien eintragen:
audio_files = [
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/sine_100Hz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/drum.mp3",
    "C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Raw_signals/sine_1kHz.mp3",
    "C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Raw_signals/Podcast_shorted.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/RAP_God.mp4",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/sine_20kHz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/violin.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/drum.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/jingle.mp3",
    # ...
]


def load_and_normalize(path, target_duration_ms=TARGET_DURATION_MS):
    """
    Lädt eine Audiodatei, konvertiert sie auf Mono & einheitliche Samplerate
    und bringt sie per Kürzen/Loopen genau auf target_duration_ms.
    """
    audio = AudioSegment.from_file(path)

    # Auf Mono und feste Sample-Rate setzen (damit alles zusammenpasst)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(44100)

    length = len(audio)

    if length > target_duration_ms:
        # Zu lang -> abschneiden
        audio = audio[:target_duration_ms]
    elif length < target_duration_ms:
        # Zu kurz -> wiederholen, bis die Ziel-Länge erreicht ist
        original = audio
        while len(audio) < target_duration_ms:
            remaining = target_duration_ms - len(audio)
            # nur so viel von original anhängen, wie noch fehlt
            audio += original[:remaining]

    return audio


def interleave_segments(
    audio_list,
    min_block_ms=MIN_BLOCK_MS,
    max_block_ms=MAX_BLOCK_MS,
    target_duration_ms=TARGET_DURATION_MS
):
    """
    N normalisierte Audios werden in Segmente zufälliger Länge (min_block_ms–max_block_ms) zerlegt
    und streng interleaved zusammengebaut: A1,B1,C1,...,A2,B2,C2,...

    Für jeden "Durchlauf" (A1,B1,C1,...) wird eine neue zufällige Segmentlänge gewählt,
    die für alle Spuren gleich ist.
    """
    if not audio_list:
        raise ValueError("audio_list ist leer – füge Dateien in audio_files hinzu!")

    output = AudioSegment.silent(duration=0)
    pos = 0  # aktuelle Position in den Eingangs-Signalen
    change_events = []
    current_time_ms = 0
    seg_index = 0
    prev_label = None


    while pos < target_duration_ms:
        remaining = target_duration_ms - pos
        if remaining <= 0:
            break

        # Zufällige Blocklänge wählen
        block_ms = random.randint(min_block_ms, max_block_ms)

        # Wenn der Block länger wäre als der Rest, auf Rest begrenzen
        if block_ms > remaining:
            block_ms = remaining

        start = pos
        end = pos + block_ms

        for src_idx, audio in enumerate(audio_list):
            # Label: A, B, C, ... (falls >26 Spuren -> S27, S28, ...)
            label = chr(ord('A') + src_idx) if src_idx < 26 else f"S{src_idx+1}"

            segment = audio[start:end]

            # Wechselstelle = Startzeit jedes Segments (außer beim allerersten)
            if seg_index > 0:
                change_events.append({
                    "change_ms": current_time_ms,
                    "change_s": current_time_ms / 1000.0,
                    "from": prev_label,
                    "to": label,
                    "segment_index": seg_index,
                    "block_ms": len(segment),
                    "source_pos_ms": start,
                })

            output += segment
            current_time_ms += len(segment)
            prev_label = label
            seg_index += 1

        pos += block_ms

    return output, change_events


def save_change_events_csv(events, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = ["change_ms", "change_s", "from", "to", "segment_index", "block_ms", "source_pos_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)

def main():
    # 1. Alle Audios laden und auf Ziel-Länge normalisieren
    prepared_audios = []
    for path in audio_files:
        print(f"Lade und normalisiere: {path}")
        prepared_audios.append(load_and_normalize(path))

    # 2. Interleaven mit zufälligen Blocklängen
    print("Interleave-Audio wird erstellt...")
    interleaved, change_events = interleave_segments(prepared_audios)

    # 3. Speichern
    print(f"Speichere Ergebnis als: {OUTPUT_FILE}")
    interleaved.export(OUTPUT_FILE, format="mp3")
    print("Fertig!")

    # 4. Wechselstellen als CSV speichern
    print(f"Speichere Wechselstellen als: {CSV_FILE}")
    save_change_events_csv(change_events, CSV_FILE)

if __name__ == "__main__":
    main()
