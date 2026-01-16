from pydub import AudioSegment
import random
import csv
import os
# ===== EINSTELLUNGEN =====
TARGET_DURATION_MS = 30_000    # Ziel-Länge pro Spur (z.B. 10_000 für 10 s)
MIN_BLOCK_MS = 10             # minimale Blocklänge in ms
MAX_BLOCK_MS = 50             # maximale Blocklänge in ms
OUTPUT_FILE = "Inputsignals/rand/interleaved_RAP_Tag_30sec_rand.mp3"
CSV_FILE = os.path.splitext(OUTPUT_FILE)[0] + "_wechselstellen.csv"

FIRST_SIGNAL_MS = 200   # erste 100 ms sollen nur Signal A sein

# HIER deine MP3-Dateien eintragen:
audio_files = [
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/sine_100Hz.mp3",
    #"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Signale/drum.mp3",
    "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Raw_signals/TagesSchau (2).mp3",
    "C:/eigene Programme/VS_Code_Programme/HKA/DSP/Raw_signals/RAP_God.mp3",
    #"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Raw_signals/sine_20kHz.mp3",
    #"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Raw_signals/sine_30Hz.mp3",
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
    target_duration_ms=TARGET_DURATION_MS,
    first_signal_ms=FIRST_SIGNAL_MS
):
    """
    N normalisierte Audios werden in Segmente zufälliger Länge (min_block_ms–max_block_ms) zerlegt
    und streng interleaved zusammengebaut: A1,B1,C1,...,A2,B2,C2,...

    Zusätzlich: Die ersten `first_signal_ms` im Output sind garantiert nur Signal A.
    Damit die Gesamtdauer stimmt, werden die "verdrängten" ersten first_signal_ms von B,C,...
    am Ende angehängt.
    """
    if not audio_list:
        raise ValueError("audio_list ist leer – füge Dateien in audio_files hinzu!")

    output = AudioSegment.silent(duration=0)
    change_events = []
    current_time_ms = 0
    prev_label = None
    seg_index = 0

    n = len(audio_list)
    first_signal_ms = max(0, min(int(first_signal_ms), int(target_duration_ms)))

    # 1) Start: nur A für first_signal_ms
    if first_signal_ms > 0:
        a_head = audio_list[0][0:first_signal_ms]
        output += a_head
        current_time_ms += len(a_head)
        prev_label = "A"
        seg_index = 1

    # 2) Heads von B,C,... merken (die hängen wir am Ende an)
    tail_heads = []
    if first_signal_ms > 0:
        for src_idx in range(1, n):
            label = chr(ord('A') + src_idx) if src_idx < 26 else f"S{src_idx+1}"
            tail_heads.append((label, audio_list[src_idx][0:first_signal_ms]))

    # Ab hier lesen wir bei allen Quellen ab pos = first_signal_ms
    pos = first_signal_ms
    first_round = True

    while pos < target_duration_ms:
        remaining = target_duration_ms - pos
        if remaining <= 0:
            break

        block_ms = random.randint(min_block_ms, max_block_ms)
        if block_ms > remaining:
            block_ms = remaining

        start = pos
        end = pos + block_ms

        # Direkt nach dem A-Head soll als nächstes B kommen (damit Wechsel bei t=first_signal_ms passiert)
        if first_round and n > 1:
            order = list(range(1, n)) + [0]   # B,C,...,A
        else:
            order = list(range(n))            # A,B,C,...

        for src_idx in order:
            audio = audio_list[src_idx]
            label = chr(ord('A') + src_idx) if src_idx < 26 else f"S{src_idx+1}"

            segment = audio[start:end]

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
        first_round = False

    # 3) Am Ende: die "verdrängten" ersten first_signal_ms von B,C,... anhängen
    for label, seg in tail_heads:
        if len(seg) == 0:
            continue

        change_events.append({
            "change_ms": current_time_ms,
            "change_s": current_time_ms / 1000.0,
            "from": prev_label,
            "to": label,
            "segment_index": seg_index,
            "block_ms": len(seg),
            "source_pos_ms": 0,
        })

        output += seg
        current_time_ms += len(seg)
        prev_label = label
        seg_index += 1

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
