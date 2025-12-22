from pydub import AudioSegment

# ==============================
# Parameter
# ==============================
CHUNK_MS = 20  # Dauer eines Segments in Millisekunden (10–50 ms sinnvoll)

TARGET_LENGTH_SEC = 20     # Ziel-Länge der Audios in Sekunden
TARGET_LENGTH_MS = TARGET_LENGTH_SEC * 1000

# Pfade zu deinen 3 MP3-Dateien
FILE_A = "sinus.mp3"  # z.B. Sprachaufnahme
FILE_B = "drum.mp3"  # z.B. Musik
FILE_C = "violin.mp3"  # z.B. weiteres Signal
OUTPUT_FILE = "interleaved_output.mp3"  # Ergebnisdatei


def load_and_normalize(path, target_frame_rate=48000, target_channels=1):
    """
    Lädt eine MP3-Datei, bringt sie auf gleiche Abtastrate & Kanalzahl
    und passt die Länge auf exakt TARGET_LENGTH_MS an.
    """
    audio = AudioSegment.from_file(path)  # erkennt MP3 automatisch
    audio = audio.set_frame_rate(target_frame_rate)
    audio = audio.set_channels(target_channels)
    audio = make_exact_length(audio, TARGET_LENGTH_MS)
    return audio

def make_exact_length(audio, target_ms):
    """
    Bringt ein AudioSegment auf exakt target_ms:
    - ist es länger -> hart auf target_ms schneiden
    - ist es kürzer -> loopen, bis >= target_ms, dann wieder auf target_ms schneiden
    """
    if len(audio) > target_ms:
        # Zu lang -> einfach abschneiden
        return audio[:target_ms]

    if len(audio) == target_ms:
        return audio

    if len(audio) == 0:
        raise ValueError("Audio-Datei ist leer oder konnte nicht geladen werden.")

    # Zu kurz -> loopen
    out = AudioSegment.silent(duration=0)
    while len(out) < target_ms:
        remaining = target_ms - len(out)
        if remaining >= len(audio):
            out += audio
        else:
            # Nur den Rest auffüllen
            out += audio[:remaining]

    # Sicherheit: auf exakt target_ms trimmen (falls minimal drüber)
    return out[:target_ms]


def chunk_audio(audio, chunk_ms):
    """
    Zerlegt ein AudioSegment in gleich lange Stücke mit Länge chunk_ms.
    """
    chunks = []
    for start in range(0, len(audio), chunk_ms):
        chunk = audio[start:start + chunk_ms]
        chunks.append(chunk)  # letztes, evtl. kürzeres Stück wird mitgenommen
    return chunks


def interleave_three_sources(chunks_a, chunks_b, chunks_c):
    """
    Interleaved die Chunks A, B, C streng:
    A1, B1, C1, A2, B2, C2, ...
    bis die kürzeste Liste zu Ende ist.
    (Da wir vorher alle exakt gleich lang gemacht haben, sollten die
     Längen eigentlich identisch sein.)
    """
    num_chunks = min(len(chunks_a), len(chunks_b), len(chunks_c))
    out = AudioSegment.silent(duration=0)

    for i in range(num_chunks):
        out += chunks_a[i]
        out += chunks_b[i]
        out += chunks_c[i]

    return out


def main():
    print("Lade, normalisiere und trimme/loope Dateien auf 20 Sekunden...")

    a = load_and_normalize(FILE_A)
    b = load_and_normalize(FILE_B)
    c = load_and_normalize(FILE_C)

    print(f"Längen (sollten jetzt ~{TARGET_LENGTH_SEC}s sein): "
          f"A={len(a)/1000:.3f}s, B={len(b)/1000:.3f}s, C={len(c)/1000:.3f}s")

    print(f"Zerlege in Chunks zu {CHUNK_MS} ms...")
    chunks_a = chunk_audio(a, CHUNK_MS)
    chunks_b = chunk_audio(b, CHUNK_MS)
    chunks_c = chunk_audio(c, CHUNK_MS)

    print("Interleave A, B, C...")
    result = interleave_three_sources(chunks_a, chunks_b, chunks_c)

    print(f"Exportiere Ergebnis nach: {OUTPUT_FILE}")
    result.export(OUTPUT_FILE, format="mp3", bitrate="192k")

    print("Fertig!")


if __name__ == "__main__":
    main()