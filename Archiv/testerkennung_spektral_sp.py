import numpy as np
import librosa

def berechne_spektralen_schwerpunkt(
    datei_pfad: str,
    frame_size: int = 2048,
    hop_length: int = 512
):
    """
    Berechnet den spektralen Schwerpunkt (spectral centroid) eines Audiosignals.

    Parameter:
        datei_pfad : Pfad zur Audio-Datei (z.B. .wav, .mp3)
        frame_size: FFT-Fenstergröße
        hop_length: Schrittweite zwischen den Frames

    Rückgabe:
        sr        : Samplingrate
        zeiten    : Zeitpunkte jedes Frames (in Sekunden)
        centroids : Spektraler Schwerpunkt pro Frame (in Hz)
        centroid_mean : gemittelter Schwerpunkt über das ganze Signal (in Hz)
    """
    # Audio laden (mono)
    y, sr = librosa.load(datei_pfad, sr=None, mono=True)

    # Betrag des Spektrums via STFT
    S = np.abs(librosa.stft(y, n_fft=frame_size, hop_length=hop_length))

    # Frequenzachsenwerte der FFT-Bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_size)

    # Spektraler Schwerpunkt pro Frame:
    #   C = Sum( f_k * |X_k| ) / Sum( |X_k| )
    # S Form: (freq_bins, frames)
    # freqs[:, None] broadcastet die Frequenzen über die Spalten (Frames)
    spektralgewicht = S.sum(axis=0)
    # Schutz vor Division durch 0
    spektralgewicht[spektralgewicht == 0] = 1e-12

    centroids = (freqs[:, None] * S).sum(axis=0) / spektralgewicht

    # Zeitachse zu jedem Frame
    frames = np.arange(len(centroids))
    zeiten = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # Mittelwert über das ganze Signal
    centroid_mean = float(np.mean(centroids))

    return sr, zeiten, centroids, centroid_mean


if __name__ == "__main__":
    # Beispielaufruf
    pfad = "drum.mp3"  # hier Pfad zur Datei einsetzen
    sr, t, c, c_mean = berechne_spektralen_schwerpunkt(pfad)

    print(f"Samplingrate: {sr} Hz")
    print(f"Anzahl Frames: {len(c)}")
    print(f"Durchschnittlicher spektraler Schwerpunkt: {c_mean:.2f} Hz")

    # Beispiel: ersten 10 Werte anzeigen
    for ti, ci in list(zip(t, c))[:10]:
        print(f"t = {ti:6.3f} s, Spektraler Schwerpunkt = {ci:8.2f} Hz")
