import librosa
import numpy as np
import matplotlib.pyplot as plt

# === Einstellungen ===
audio_path = "C:/eigene Programme/VS_Code_Programme/HKA/DSP/out_v5.7/interleaved_pod_1k_30sec_rand/0/Signal_A.mp3"  # Pfad zur MP3-Datei

# Grenzfrequenzen für den Plot [Hz]
# Setze f_min oder f_max auf None, wenn du die jeweilige Grenze nicht einschränken willst
f_min = 0        # z.B. 0 Hz
f_max = 30000     # z.B. 8000 Hz (oder None für Nyquist-Grenze)

# === MP3 laden ===
y, sr = librosa.load(audio_path, sr=None, mono=True)

print(f"Abtastrate: {sr} Hz")
dauer = len(y) / sr
print(f"Dauer: {dauer:.2f} s")

# === Zeitachse ===
t = np.linspace(0, dauer, len(y))

# === FFT ===
N = len(y)
Y = np.fft.rfft(y)
freqs = np.fft.rfftfreq(N, d=1/sr)
magnitude = np.abs(Y)

# === Frequenzgrenzen anpassen ===
# Standard: 0 bis Nyquist
plot_f_min = 0 if f_min is None else max(0, f_min)
plot_f_max = (sr / 2) if f_max is None else min(f_max, sr / 2)

# Indexbereich für gewünschten Frequenzbereich
mask = (freqs >= plot_f_min) & (freqs <= plot_f_max)

# === Plotten ===
plt.figure(figsize=(12, 6))

# --- 1) Zeitbereich ---
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.title("Signal im Zeitbereich")
plt.xlabel("Zeit [s]")
plt.ylabel("Amplitude")
plt.grid(True)

# --- 2) Frequenzbereich ---
plt.subplot(2, 1, 2)
plt.plot(freqs[mask], magnitude[mask])
plt.title("Signal im Frequenzbereich (Betragsspektrum)")
plt.xlabel("Frequenz [Hz]")
plt.ylabel("|X(f)|")
plt.grid(True)

plt.tight_layout()
plt.show()
