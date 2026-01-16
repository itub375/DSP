import numpy as np
import librosa
import csv
from pathlib import Path


def extract_audio_features(audio_file, block_size_ms=50):
    """
    Extrahiert spektrale Features aus einer Audiodatei.
    
    Parameters:
    -----------
    audio_file : str
        Pfad zur MP3-Datei
    block_size_ms : float
        Blockgröße in Millisekunden (Standard: 50ms)
    """
    
    # Audio laden
    print(f"Lade Audiodatei: {audio_file}")
    y, sr = librosa.load(audio_file, sr=None)
    
    # Blockgröße von ms in Samples umrechnen
    hop_length = int(sr * block_size_ms / 1000)
    n_fft = hop_length * 4  # FFT-Größe (üblicherweise größer als hop_length)
    
    print(f"Samplerate: {sr} Hz")
    print(f"Blockgröße: {block_size_ms} ms ({hop_length} Samples)")
    print(f"FFT-Größe: {n_fft} Samples")
    
    # STFT berechnen
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    
    # Features berechnen
    print("Berechne Features...")
    
    # 1. Spektral Centroid (in Hz)
    spectral_centroid = librosa.feature.spectral_centroid(
        S=magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    
    # 2. Power (in dB)
    power_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    power_per_frame = np.mean(power_db, axis=0)
    
    # 3. Energy (RMS Energy)
    energy = librosa.feature.rms(S=magnitude, frame_length=n_fft, hop_length=hop_length)[0]
    
    # 4. Peak Frequenz (in Hz) - Frequenz mit höchster Magnitude pro Frame
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peak_freq = freqs[np.argmax(magnitude, axis=0)]
    
    # 5. Spectral Flux
    flux = np.sqrt(np.sum(np.diff(magnitude, axis=1)**2, axis=0))
    flux = np.concatenate([[0], flux])  # Ersten Wert auf 0 setzen
    
    # 6. Spectral Bandwidth (in Hz)
    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]
    
    # 7. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(
        S=magnitude, hop_length=hop_length
    )[0]
    
    # 8. Spectral Rolloff (in Hz)
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )[0]
    
    # Zeitstempel berechnen (in Sekunden)
    times = librosa.frames_to_time(
        np.arange(len(spectral_centroid)), sr=sr, hop_length=hop_length
    )
    
    # Statistiken berechnen
    stats = {
        'Spektral Centroid (Hz)': {
            'min': np.min(spectral_centroid),
            'max': np.max(spectral_centroid),
            'mean': np.mean(spectral_centroid)
        },
        'Power (dB)': {
            'min': np.min(power_per_frame),
            'max': np.max(power_per_frame),
            'mean': np.mean(power_per_frame)
        },
        'Energy': {
            'min': np.min(energy),
            'max': np.max(energy),
            'mean': np.mean(energy)
        },
        'Peak Frequenz (Hz)': {
            'min': np.min(peak_freq),
            'max': np.max(peak_freq),
            'mean': np.mean(peak_freq)
        },
        'Spectral Flux': {
            'min': np.min(flux),
            'max': np.max(flux),
            'mean': np.mean(flux)
        },
        'Bandbreite (Hz)': {
            'min': np.min(bandwidth),
            'max': np.max(bandwidth),
            'mean': np.mean(bandwidth)
        },
        'Spectral Flatness': {
            'min': np.min(flatness),
            'max': np.max(flatness),
            'mean': np.mean(flatness)
        },
        'Spectral Rolloff (Hz)': {
            'min': np.min(rolloff),
            'max': np.max(rolloff),
            'mean': np.mean(rolloff)
        }
    }
    
    # Statistiken ausgeben
    print("\n=== Statistiken ===")
    for feature, values in stats.items():
        print(f"\n{feature}:")
        print(f"  Minimum: {values['min']:.2f}")
        print(f"  Maximum: {values['max']:.2f}")
        print(f"  Durchschnitt: {values['mean']:.2f}")
    
    # CSV-Datei erstellen
    output_file = "C:/eigene Programme/VS_Code_Programme/HKA/DSP/CSV/"+ Path(audio_file).stem + "_CSV.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header schreiben
        writer.writerow([
            'Zeit (s)',
            'Spektral Centroid (Hz)',
            'Power (dB)',
            'Energy',
            'Peak Frequenz (Hz)',
            'Spectral Flux',
            'Bandbreite (Hz)',
            'Spectral Flatness',
            'Spectral Rolloff (Hz)'
        ])
        
        # Daten schreiben
        for i in range(len(times)):
            writer.writerow([
                f"{times[i]:.3f}",
                f"{spectral_centroid[i]:.2f}",
                f"{power_per_frame[i]:.2f}",
                f"{energy[i]:.6f}",
                f"{peak_freq[i]:.2f}",
                f"{flux[i]:.6f}",
                f"{bandwidth[i]:.2f}",
                f"{flatness[i]:.6f}",
                f"{rolloff[i]:.2f}"
            ])
        
        # Leerzeile
        writer.writerow([])
        
        # Statistiken schreiben
        writer.writerow(['=== STATISTIKEN ==='])
        writer.writerow([])
        
        writer.writerow(['Feature', 'Minimum', 'Maximum', 'Durchschnitt'])
        for feature, values in stats.items():
            writer.writerow([
                feature,
                f"{values['min']:.2f}",
                f"{values['max']:.2f}",
                f"{values['mean']:.2f}"
            ])
    
    print(f"\nErgebnisse gespeichert in: {output_file}")
    print(f"Anzahl verarbeiteter Blöcke: {len(times)}")


# Beispielaufruf
if __name__ == "__main__":
    # Pfad zur MP3-Datei anpassen
    audio_files_folder = r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Raw_signals/"               
    p = Path(audio_files_folder)
    files_to_process = [str(f) for f in p.iterdir() if f.is_file()]

    # Blockgröße in ms (kann einfach angepasst werden)
    block_size_ms = 50
    
    # Features extrahieren
    for a in files_to_process:
        extract_audio_features(a, block_size_ms)
        