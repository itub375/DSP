import math
import numpy as np
from pydub import AudioSegment
from pathlib import Path

# ========================================
# KONFIGURATION - Hier anpassen!
# ========================================
INPUT_FILE: str = r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/rand/interleaved_1k_8k_20k_rand.mp3"
OUTPUT_FILE = r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/output_cut.mp3"

# Schwellwert als PROZENT der maximalen Amplitude (0-100%)
# 0% = absolut Null, 1% = sehr leise, 5% = noch leise, 10% = deutlich hörbar
SILENCE_THRESHOLD_PERCENT = 55  # Empfohlen: 0.1 - 2.0%

# Mindestlänge eines Stille-Bereichs, damit er entfernt wird (in ms)
MIN_SILENCE_DURATION_MS = 0.5  # Nur Bereiche >= diesem Wert werden entfernt

# Optional: Crossfade beim Zusammenfügen gegen Klicks (in ms)
CROSSFADE_MS = 0  # 0 = aus, 1-5 ms empfohlen bei hörbaren Klicks

# Debug-Modus: Zeigt detaillierte Informationen
DEBUG = True


# ========================================
# EILENES DATEIPFAD
# ========================================

def ask_input_file(default_path: str = "") -> str:
    prompt = 'Welche Datei soll bearbeitet werden?'
    if default_path:
        prompt += f' (Enter = Default: {default_path})'
    prompt += '\n> '

    path = input(prompt).strip()

    # Wenn leer -> Default nehmen
    if not path:
        path = default_path

    # Anführungszeichen entfernen (Windows Copy/Paste)
    path = path.strip().strip('"').strip("'")

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    return str(p)

# ========================================
# HAUPTFUNKTIONEN
# ========================================

def find_silence_regions(audio_samples, threshold_abs, sample_rate, min_duration_samples):
    """
    Findet alle Bereiche in den Audio-Samples, die als Stille gelten.
    
    Returns:
        List von (start_sample, end_sample) Tupeln
    """
    # Bei Stereo: Maximum über alle Kanäle
    if audio_samples.ndim > 1:
        amplitude = np.max(np.abs(audio_samples), axis=1)
    else:
        amplitude = np.abs(audio_samples)
    
    # Maske: True wo Amplitude <= Schwellwert (= Stille)
    is_silent = amplitude <= threshold_abs
    
    # Finde zusammenhängende Stille-Bereiche
    silence_regions = []
    in_silence = False
    start = 0
    
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            # Stille beginnt
            start = i
            in_silence = True
        elif not silent and in_silence:
            # Stille endet
            if i - start >= min_duration_samples:
                silence_regions.append((start, i))
            in_silence = False
    
    # Falls am Ende noch Stille ist
    if in_silence and len(is_silent) - start >= min_duration_samples:
        silence_regions.append((start, len(is_silent)))
    
    return silence_regions


def remove_silence(audio, silence_regions, sample_rate, crossfade_ms):
    """
    Entfernt Stille-Bereiche aus dem Audio und fügt die Teile zusammen.
    
    Returns:
        AudioSegment ohne Stille
    """
    if not silence_regions:
        return audio
    
    # Berechne Keep-Bereiche (Komplement der Stille)
    keep_regions = []
    n_samples = len(audio.get_array_of_samples()) // audio.channels
    
    current_pos = 0
    for silence_start, silence_end in silence_regions:
        if current_pos < silence_start:
            keep_regions.append((current_pos, silence_start))
        current_pos = silence_end
    
    # Letzter Bereich nach letzter Stille
    if current_pos < n_samples:
        keep_regions.append((current_pos, n_samples))
    
    # Zusammenfügen
    output = AudioSegment.empty()
    
    for i, (start_sample, end_sample) in enumerate(keep_regions):
        start_ms = int(round(start_sample * 1000.0 / sample_rate))
        end_ms = int(round(end_sample * 1000.0 / sample_rate))
        
        segment = audio[start_ms:end_ms]
        
        if i == 0:
            output = segment
        else:
            output = output.append(segment, crossfade=crossfade_ms)
    
    return output


def main():
    inputfile = ask_input_file(INPUT_FILE)

    print("=" * 60)
    print("AUDIO STILLE-ENTFERNER")
    print("=" * 60)
    
    # Audio laden
    print(f"\n📂 Lade Audio: {inputfile}")
    audio = AudioSegment.from_file(inputfile)
    
    # Audio-Eigenschaften
    sample_rate = audio.frame_rate
    channels = audio.channels
    duration_ms = len(audio)
    
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   Kanäle: {channels}")
    print(f"   Dauer: {duration_ms/1000:.2f} s ({duration_ms} ms)")
    
    # Samples als NumPy Array
    samples = np.array(audio.get_array_of_samples())
    
    # Bei Stereo: Reshape zu (N, channels)
    if channels > 1:
        samples = samples.reshape((-1, channels))
    
    # Maximale Amplitude ermitteln
    max_amplitude = np.max(np.abs(samples))
    
    # Schwellwert berechnen
    threshold_abs = max_amplitude * (SILENCE_THRESHOLD_PERCENT / 100.0)
    min_silence_samples = int(math.ceil((MIN_SILENCE_DURATION_MS / 1000.0) * sample_rate))
    
    print(f"\n🎚️  Schwellwert-Einstellungen:")
    print(f"   Maximale Amplitude: {max_amplitude}")
    print(f"   Stille-Schwellwert: {SILENCE_THRESHOLD_PERCENT}% = {threshold_abs:.1f}")
    print(f"   Min. Stille-Dauer: {MIN_SILENCE_DURATION_MS} ms = {min_silence_samples} samples")
    
    # Stille-Bereiche finden
    print(f"\n🔍 Suche Stille-Bereiche...")
    silence_regions = find_silence_regions(samples, threshold_abs, sample_rate, min_silence_samples)
    
    if not silence_regions:
        print("   ✓ Keine Stille-Bereiche gefunden (oder zu kurz)")
        print(f"\n💡 Tipp: Erhöhe SILENCE_THRESHOLD_PERCENT (aktuell: {SILENCE_THRESHOLD_PERCENT}%)")
        return
    
    # Statistik
    total_removed_ms = sum((end - start) * 1000.0 / sample_rate 
                          for start, end in silence_regions)
    removed_percent = (total_removed_ms / duration_ms) * 100
    
    print(f"   ✓ {len(silence_regions)} Stille-Bereiche gefunden")
    print(f"   ✓ Zu entfernen: {total_removed_ms:.2f} ms ({removed_percent:.1f}% der Gesamtdauer)")
    
    if DEBUG:
        print(f"\n📊 Details der gefundenen Stille-Bereiche:")
        for i, (start, end) in enumerate(silence_regions, 1):
            duration = (end - start) * 1000.0 / sample_rate
            time_pos = start * 1000.0 / sample_rate
            print(f"   #{i}: {time_pos:.0f} ms - {time_pos+duration:.0f} ms (Dauer: {duration:.2f} ms)")
            if i >= 10 and len(silence_regions) > 10:
                print(f"   ... und {len(silence_regions)-10} weitere")
                break
    
    # Stille entfernen
    print(f"\n✂️  Entferne Stille und füge zusammen...")
    output = remove_silence(audio, silence_regions, sample_rate, CROSSFADE_MS)
    
    # Exportieren
    print(f"\n💾 Exportiere: {OUTPUT_FILE}")
    output.export(OUTPUT_FILE, format="mp3")
    
    # Finale Statistik
    output_duration_ms = len(output)
    compression_ratio = (1 - output_duration_ms / duration_ms) * 100
    
    print(f"\n" + "=" * 60)
    print(f"✅ FERTIG!")
    print(f"=" * 60)
    print(f"Original-Länge:  {duration_ms/1000:.2f} s")
    print(f"Neue Länge:      {output_duration_ms/1000:.2f} s")
    print(f"Entfernt:        {total_removed_ms/1000:.2f} s ({compression_ratio:.1f}%)")
    print(f"=" * 60)
    
    if compression_ratio < 1:
        print(f"\n💡 Kaum Stille entfernt. Versuche:")
        print(f"   • SILENCE_THRESHOLD_PERCENT erhöhen (aktuell: {SILENCE_THRESHOLD_PERCENT}%)")
        print(f"   • MIN_SILENCE_DURATION_MS verringern (aktuell: {MIN_SILENCE_DURATION_MS} ms)")


if __name__ == "__main__":
    main()