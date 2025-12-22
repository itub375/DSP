from pydub import AudioSegment
import random

# ===== EINSTELLUNGEN =====
TARGET_DURATION_MS = 30_000    # Ziel-Länge pro Spur (z.B. 10_000 für 10 s)
MIN_BLOCK_MS = 20             # minimale Blocklänge in ms
MAX_BLOCK_MS = 50             # maximale Blocklänge in ms
OUTPUT_FILE = "Inputsignals/rand/interleaved_pod_1k_30sec_rand.mp3"

# HIER deine MP3-Dateien eintragen:
audio_files = [
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/sine_100Hz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/drum.mp3",
    "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Raw_signals/sine_1kHz.mp3",
    "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Raw_signals/Podcast.mp3",
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

        for audio in audio_list:
            segment = audio[start:end]
            output += segment

        pos += block_ms

    return output


def main():
    # 1. Alle Audios laden und auf Ziel-Länge normalisieren
    prepared_audios = []
    for path in audio_files:
        print(f"Lade und normalisiere: {path}")
        prepared_audios.append(load_and_normalize(path))

    # 2. Interleaven mit zufälligen Blocklängen
    print("Interleave-Audio wird erstellt...")
    interleaved = interleave_segments(prepared_audios)

    # 3. Speichern
    print(f"Speichere Ergebnis als: {OUTPUT_FILE}")
    interleaved.export(OUTPUT_FILE, format="mp3")
    print("Fertig!")


if __name__ == "__main__":
    main()
