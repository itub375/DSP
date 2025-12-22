import numpy as np
import librosa
import matplotlib.pyplot as plt

def plot_spektrum_mit_schwerpunkt(
    datei_pfad: str,
    verwende_hanning: bool = True
):
    """
    Lädt eine Audiodatei, berechnet das (globale) Amplitudenspektrum
    und zeichnet den spektralen Schwerpunkt als vertikale Linie ein.
    """

    # 1. Audio laden (mono, Original-Samplingrate)
    y, sr = librosa.load(datei_pfad, sr=None, mono=True)

    # 2. Optional Fensterung zur Reduktion von Leakage
    if verwende_hanning:
        window = np.hanning(len(y))
        y_win = y * window
    else:
        y_win = y

    # 3. FFT (nur positive Frequenzen)
    X = np.fft.rfft(y_win)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(y_win), d=1.0/sr)

    # 4. Spektralen Schwerpunkt berechnen:
    #    C = Sum( f_k * |X_k| ) / Sum( |X_k| )
    mag_sum = np.sum(mag)
    if mag_sum == 0:
        centroid = 0.0
    else:
        centroid = np.sum(freqs * mag) / mag_sum

    # 5. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, mag, label="Spektrum |X(f)|")

    # Schwerpunkt als vertikale Linie
    plt.axvline(centroid, linestyle='--', linewidth=2,
                label=f"Schwerpunkt ≈ {centroid:.1f} Hz")

    # Beschriftung direkt am Plot
    ymax = mag.max() if mag.size > 0 else 1
    plt.text(centroid, ymax * 0.9, f"{centroid:.0f} Hz",
             rotation=90, va="top", ha="right")

    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Betrag |X(f)|")
    plt.title("Amplitude-Spektrum mit spektralem Schwerpunkt")
    plt.xlim(0, sr / 2)      # nur 0–Nyquist zeigen
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return centroid, sr


if __name__ == "__main__":
    pfad = "drum.mp3"  # Pfad anpassen
    centroid, sr = plot_spektrum_mit_schwerpunkt(pfad)
    print(f"Spektraler Schwerpunkt: {centroid:.2f} Hz (Samplingrate: {sr} Hz)")
