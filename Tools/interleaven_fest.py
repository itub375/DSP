from pydub import AudioSegment

# ===== EINSTELLUNGEN =====
TARGET_DURATION_MS = 5_000   # 10 Sekunden
BLOCK_MS = 50                 # Blocklänge 50 ms
OUTPUT_FILE = "Inputsignale/50ms/interleaved_1k_8k_vio_50ms.mp3"

# HIER deine MP3-Dateien eintragen:
audio_files = [
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Singale/sine_30Hz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Singale/drum.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Singale/sine_200Hz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Singale/sine_500Hz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Singale/sine_600Hz.mp3",
    "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/sine_1kHz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Singale/sine_4625kHz.mp3",
    "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/sine_8kHz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/sine_20kHz.mp3",
    "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/violin.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Singale/drum.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Singale/jingle.mp3",
    # "audio3.mp3",
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


def interleave_segments(audio_list, block_ms=BLOCK_MS, target_duration_ms=TARGET_DURATION_MS):
    """
    N normalisierte Audios werden in block_ms-Segmente zerlegt
    und streng interleaved zusammengebaut: A1,B1,C1,...,A2,B2,C2,...
    """
    if not audio_list:
        raise ValueError("audio_list ist leer – füge Dateien in audio_files hinzu!")

    blocks_per_file = target_duration_ms // block_ms
    output = AudioSegment.silent(duration=0)

    for i in range(blocks_per_file):
        start = i * block_ms
        end = start + block_ms

        for audio in audio_list:
            segment = audio[start:end]
            output += segment

    return output


def main():
    # 1. Alle Audios laden und auf 10 s normalisieren
    prepared_audios = []
    for path in audio_files:
        print(f"Lade und normalisiere: {path}")
        prepared_audios.append(load_and_normalize(path))

    # 2. Interleaven
    print("Interleave-Audio wird erstellt...")
    interleaved = interleave_segments(prepared_audios)

    # 3. Speichern
    print(f"Speichere Ergebnis als: {OUTPUT_FILE}")
    interleaved.export(OUTPUT_FILE, format="wav")
    print("Fertig!")


if __name__ == "__main__":
    main()
