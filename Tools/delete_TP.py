import math
import numpy as np
from pydub import AudioSegment
from scipy import signal
from pathlib import Path

# ========================================
# KONFIGURATION - Hier anpassen!
# ========================================
INPUT_FILE = r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/out/out_v5.8/interleaved_pod_1k_60sec_rand/Signal_A.mp3"
OUTPUT_FILE = r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/output_cut_TP.mp3"

# --- TIEFPASS-FILTER ---
ENABLE_LOWPASS = True                # True = Tiefpass aktivieren, False = deaktivieren
LOWPASS_CUTOFF_HZ = 20000             # Grenzfrequenz in Hz (z.B. 8000 = alle Frequenzen > 8kHz werden gedämpft)
LOWPASS_ORDER = 5                    # Filter-Ordnung (höher = steiler, 4-6 empfohlen)

# --- STILLE-ENTFERNUNG ---
ENABLE_SILENCE_REMOVAL = True        # True = Stille entfernen, False = nur Tiefpass

# Schwellwert als PROZENT der maximalen Amplitude (0-100%)
# 0% = absolut Null, 1% = sehr leise, 5% = noch leise, 10% = deutlich hörbar
SILENCE_THRESHOLD_PERCENT = 5      # Empfohlen: 0.1 - 2.0%

# Mindestlänge eines Stille-Bereichs, damit er entfernt wird (in ms)
MIN_SILENCE_DURATION_MS = 0.5        # Nur Bereiche >= diesem Wert werden entfernt

# Optional: Crossfade beim Zusammenfügen gegen Klicks (in ms)
CROSSFADE_MS = 0                     # 0 = aus, 1-5 ms empfohlen bei hörbaren Klicks

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
# TIEFPASS-FILTER
# ========================================

def apply_lowpass_filter(samples, sample_rate, cutoff_hz, order):
    """
    Wendet einen Butterworth-Tiefpass-Filter auf die Audio-Samples an.
    
    Args:
        samples: NumPy Array der Audio-Samples
        sample_rate: Sample-Rate in Hz
        cutoff_hz: Grenzfrequenz des Tiefpasses
        order: Filter-Ordnung
    
    Returns:
        Gefilterte Samples
    """
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    
    # Butterworth-Tiefpass designen
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Bei Stereo: Jeden Kanal einzeln filtern
    if samples.ndim > 1:
        filtered = np.zeros_like(samples)
        for ch in range(samples.shape[1]):
            filtered[:, ch] = signal.filtfilt(b, a, samples[:, ch])
    else:
        filtered = signal.filtfilt(b, a, samples)
    
    return filtered


def samples_to_audio(samples, sample_rate, channels):
    """
    Konvertiert NumPy-Samples zurück zu AudioSegment.
    """
    # Auf int16 clippen
    samples_int16 = np.clip(samples, -32768, 32767).astype(np.int16)
    
    # Bei Stereo: Flatten
    if samples_int16.ndim > 1:
        samples_int16 = samples_int16.flatten()
    
    # AudioSegment erstellen
    audio = AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit = 2 bytes
        channels=channels
    )
    
    return audio


# ========================================
# STILLE-ENTFERNUNG
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


# ========================================
# HAUPTFUNKTION
# ========================================

def main():

    inputfile = ask_input_file(INPUT_FILE)

    print("=" * 60)
    print("AUDIO-BEARBEITUNG: TIEFPASS + STILLE-ENTFERNUNG")
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
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Bei Stereo: Reshape zu (N, channels)
    if channels > 1:
        samples = samples.reshape((-1, channels))
    
    # ========================================
    # TIEFPASS-FILTER
    # ========================================
    if ENABLE_LOWPASS:
        print(f"\n🎛️  Tiefpass-Filter:")
        print(f"   Grenzfrequenz: {LOWPASS_CUTOFF_HZ} Hz")
        print(f"   Filter-Ordnung: {LOWPASS_ORDER}")
        print(f"   Verarbeite...")
        
        samples = apply_lowpass_filter(samples, sample_rate, LOWPASS_CUTOFF_HZ, LOWPASS_ORDER)
        
        # Audio neu erstellen aus gefilterten Samples
        audio = samples_to_audio(samples, sample_rate, channels)
        
        print(f"   ✓ Tiefpass angewendet")
    else:
        print(f"\n⏭️  Tiefpass-Filter: Deaktiviert")
    
    # ========================================
    # STILLE-ENTFERNUNG
    # ========================================
    if ENABLE_SILENCE_REMOVAL:
        # Maximale Amplitude ermitteln (nach Filter!)
        max_amplitude = np.max(np.abs(samples))
        
        # Schwellwert berechnen
        threshold_abs = max_amplitude * (SILENCE_THRESHOLD_PERCENT / 100.0)
        min_silence_samples = int(math.ceil((MIN_SILENCE_DURATION_MS / 1000.0) * sample_rate))
        
        print(f"\n🎚️  Stille-Entfernung:")
        print(f"   Maximale Amplitude: {max_amplitude:.1f}")
        print(f"   Stille-Schwellwert: {SILENCE_THRESHOLD_PERCENT}% = {threshold_abs:.1f}")
        print(f"   Min. Stille-Dauer: {MIN_SILENCE_DURATION_MS} ms = {min_silence_samples} samples")
        
        # Stille-Bereiche finden
        print(f"   Suche Stille-Bereiche...")
        silence_regions = find_silence_regions(samples, threshold_abs, sample_rate, min_silence_samples)
        
        if not silence_regions:
            print("   ✓ Keine Stille-Bereiche gefunden (oder zu kurz)")
            print(f"\n💡 Tipp: Erhöhe SILENCE_THRESHOLD_PERCENT (aktuell: {SILENCE_THRESHOLD_PERCENT}%)")
        else:
            # Statistik
            total_removed_ms = sum((end - start) * 1000.0 / sample_rate 
                                  for start, end in silence_regions)
            removed_percent = (total_removed_ms / duration_ms) * 100
            
            print(f"   ✓ {len(silence_regions)} Stille-Bereiche gefunden")
            print(f"   ✓ Zu entfernen: {total_removed_ms:.2f} ms ({removed_percent:.1f}% der Gesamtdauer)")
            
            if DEBUG and silence_regions:
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
            audio = remove_silence(audio, silence_regions, sample_rate, CROSSFADE_MS)
    else:
        print(f"\n⏭️  Stille-Entfernung: Deaktiviert")
    
    # ========================================
    # EXPORT
    # ========================================
    print(f"\n💾 Exportiere: {OUTPUT_FILE}")
    audio.export(OUTPUT_FILE, format="mp3")
    
    # Finale Statistik
    output_duration_ms = len(audio)
    
    print(f"\n" + "=" * 60)
    print(f"✅ FERTIG!")
    print(f"=" * 60)
    print(f"Original-Länge:  {duration_ms/1000:.2f} s")
    print(f"Neue Länge:      {output_duration_ms/1000:.2f} s")
    
    if ENABLE_SILENCE_REMOVAL and 'total_removed_ms' in locals():
        compression_ratio = (1 - output_duration_ms / duration_ms) * 100
        print(f"Entfernt:        {total_removed_ms/1000:.2f} s ({compression_ratio:.1f}%)")
    
    print(f"=" * 60)
    
    if ENABLE_SILENCE_REMOVAL and (not silence_regions or ('compression_ratio' in locals() and compression_ratio < 1)):
        print(f"\n💡 Kaum Stille entfernt. Versuche:")
        print(f"   • SILENCE_THRESHOLD_PERCENT erhöhen (aktuell: {SILENCE_THRESHOLD_PERCENT}%)")
        print(f"   • MIN_SILENCE_DURATION_MS verringern (aktuell: {MIN_SILENCE_DURATION_MS} ms)")


if __name__ == "__main__":
    main()