"""


ÄNDERUNG ZU V5.2

Ok Claude hier eine schwierigere aufgabe. Ich habe diesen code welcher ein Interleavetes mp3 einliest und deinterleaven soll. jetzt habe ich folgende punkte die angepasst werden sollen.
1. Der code soll echtzeitfähig werden. d.h. er soll ein signal durch den aux anschluss einlesen und über die lautsprecher des PCs gefiltert nur eines der erkannten signale ausgeben. dabei gillt Folgende bedingung für das eingangssignal : Es ist nicht bekannt wie viele originalsignale im eingangssignal vorhanden sind, es ist gegeben das die segmente des interleavten 10 - 50 ms lang sein können, es können leere segmente kommen bei denen kein signal oder ein extrem kleines anliegen diese sollen ignoriert werden.
2. Der Code soll eine erkennung bekommen wann ein signal beginnt. also die ganze bearbeitung soll erst starten wenn erkannt wird das ein signal anliegt. denn es ist nicht bekannt wann das signal startet es muss nocht direkt zum start des Programms anliegen.
3. Ich habe gesagt Echtzeitfähig das heist nicht in der selben ms, damit der code zeit hat die signale zu erkennen, zu sortieren und zu bearbeiten hat er einen puffer von 600 ms.
4. Es können störgerausche auftauchen welche zu beginn gefiltert werden sollen dafür soll ein bandpass filter welcher nur Frequenzen von 20Hz bis 20kHz durchlassen soll realisiert werden. Das sollte dem Hörbaren bereich des menschen entsprechen. 
5. Optional: wenn es den code einfacher macht kann man eine bedingung einbauen welche ein Limit für die interleaveten signale angibt. also das zbsp. max 3 signale im Interleaveten signal vorhanden sein dürfen.

"""


"""
Echtzeit Audio Deinterleaver
Liest Audio vom AUX-Eingang, erkennt interleaved Signale und gibt nur eines gefiltert aus

Anforderungen:
- 600ms Latenz-Puffer für Verarbeitung
- 20Hz - 20kHz Bandpass Filter
- Automatische Signal-Detektion
- Segmente: 10-50ms
- Max 3 Signale (optional)
"""

import numpy as np
import sounddevice as sd
from scipy import signal
from collections import deque
from dataclasses import dataclass
import time
from typing import Optional, List, Tuple

@dataclass
class RealtimeConfig:
    # Audio Setup
    SAMPLE_RATE: int = 44100  # Hz
    BLOCK_SIZE: int = 512     # Samples pro Callback (~ 11.6ms bei 44.1kHz)
    
    # Latenz Puffer
    LATENCY_MS: float = 600.0  # Gesamtlatenz
    
    # Signal Detection
    SIGNAL_THRESHOLD_DB: float = -40.0  # dB für Signal-Detektion
    SILENCE_TIMEOUT_MS: float = 200.0   # Zeit bis Signal als beendet gilt
    
    # Bandpass Filter
    HIGHPASS_FREQ: float = 20.0   # Hz
    LOWPASS_FREQ: float = 20000.0 # Hz
    FILTER_ORDER: int = 4
    
    # Segmentierung
    MIN_SEGMENT_MS: float = 10.0  # Minimale Segmentlänge
    MAX_SEGMENT_MS: float = 50.0  # Maximale Segmentlänge
    SEGMENT_DETECT_WINDOW_MS: float = 5.0  # Fenster für Change-Detection
    
    # Clustering
    MAX_SIGNALS: int = 3  # Maximale Anzahl zu trennender Signale
    
    # Welches Signal ausgeben? (0 = niedrigste Frequenz, 1 = mittlere, 2 = höchste)
    OUTPUT_SIGNAL_INDEX: int = 0
    
    # Debug
    VERBOSE: bool = True


class BandpassFilter:
    """Butterworth Bandpass Filter für Echtzeit-Verarbeitung"""
    
    def __init__(self, lowcut: float, highcut: float, fs: float, order: int = 4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        
        # Design Filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        self.sos = signal.butter(order, [low, high], btype='band', output='sos')
        
        # Filter State für kontinuierliche Verarbeitung
        self.zi = signal.sosfilt_zi(self.sos)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Filtert Audio-Block und erhält State"""
        filtered, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        return filtered


class RingBuffer:
    """Ring-Buffer für Audio-Daten mit Latenz-Management"""
    
    def __init__(self, size_samples: int):
        self.buffer = np.zeros(size_samples, dtype=np.float32)
        self.size = size_samples
        self.write_pos = 0
        self.read_pos = 0
        self.filled = 0
    
    def write(self, data: np.ndarray):
        """Schreibt Daten in Buffer"""
        n = len(data)
        
        # Wrap around wenn nötig
        if self.write_pos + n > self.size:
            first_part = self.size - self.write_pos
            self.buffer[self.write_pos:] = data[:first_part]
            self.buffer[:n - first_part] = data[first_part:]
        else:
            self.buffer[self.write_pos:self.write_pos + n] = data
        
        self.write_pos = (self.write_pos + n) % self.size
        self.filled = min(self.filled + n, self.size)
    
    def read(self, n_samples: int) -> np.ndarray:
        """Liest n Samples vom Read-Pointer"""
        if self.filled < n_samples:
            return np.zeros(n_samples, dtype=np.float32)
        
        if self.read_pos + n_samples > self.size:
            first_part = self.size - self.read_pos
            result = np.concatenate([
                self.buffer[self.read_pos:],
                self.buffer[:n_samples - first_part]
            ])
        else:
            result = self.buffer[self.read_pos:self.read_pos + n_samples].copy()
        
        self.read_pos = (self.read_pos + n_samples) % self.size
        self.filled -= n_samples
        
        return result
    
    def get_latest_window(self, window_samples: int) -> np.ndarray:
        """Holt die letzten N Samples ohne Read-Pointer zu bewegen"""
        if self.filled < window_samples:
            return np.zeros(window_samples, dtype=np.float32)
        
        end_pos = self.write_pos
        start_pos = (end_pos - window_samples) % self.size
        
        if start_pos < end_pos:
            return self.buffer[start_pos:end_pos].copy()
        else:
            return np.concatenate([
                self.buffer[start_pos:],
                self.buffer[:end_pos]
            ])


class SegmentDetector:
    """Erkennt Segment-Wechsel in Echtzeit"""
    
    def __init__(self, sr: int, window_ms: float = 5.0):
        self.sr = sr
        self.window_samples = int(window_ms * sr / 1000)
        
        # History für Feature-Tracking
        self.centroid_history = deque(maxlen=10)
        self.rms_history = deque(maxlen=10)
        
        self.last_change_time = 0
        self.current_segment_start = 0
    
    def compute_features(self, audio_chunk: np.ndarray) -> Tuple[float, float]:
        """Berechnet Spectral Centroid und RMS für Chunk"""
        
        if len(audio_chunk) < 10:
            return 0.0, 0.0
        
        # RMS
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Spectral Centroid
        window = np.hanning(len(audio_chunk))
        windowed = audio_chunk * window
        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sr)
        
        spec_power = spec ** 2
        
        if spec_power.sum() > 1e-10:
            centroid = np.sum(freqs * spec_power) / spec_power.sum()
        else:
            centroid = 0.0
        
        return centroid, rms
    
    def detect_change(self, centroid: float, rms: float, 
                     min_segment_ms: float = 10.0) -> bool:
        """Erkennt ob sich das Signal signifikant geändert hat"""
        
        current_time = time.time()
        
        # Zu kurze Segmente vermeiden
        if (current_time - self.last_change_time) * 1000 < min_segment_ms:
            self.centroid_history.append(centroid)
            self.rms_history.append(rms)
            return False
        
        if len(self.centroid_history) < 3:
            self.centroid_history.append(centroid)
            self.rms_history.append(rms)
            return False
        
        # Berechne Änderung
        prev_centroid = np.median(list(self.centroid_history)[-3:])
        prev_rms = np.median(list(self.rms_history)[-3:])
        
        # Relative Änderung
        centroid_change = abs(centroid - prev_centroid) / max(prev_centroid, 100)
        rms_change = abs(rms - prev_rms) / max(prev_rms, 0.001)
        
        # Threshold
        is_change = centroid_change > 0.3 or rms_change > 0.5
        
        self.centroid_history.append(centroid)
        self.rms_history.append(rms)
        
        if is_change:
            self.last_change_time = current_time
        
        return is_change


class SignalClassifier:
    """Klassifiziert Segmente zu Signalen basierend auf Frequenz"""
    
    def __init__(self, max_signals: int = 3):
        self.max_signals = max_signals
        self.signal_profiles = []  # Liste von (centroid_mean, rms_mean)
        self.last_update = time.time()
        self.adaptation_rate = 0.1  # Wie schnell sich Profile anpassen
    
    def classify_segment(self, centroid: float, rms: float) -> int:
        """Ordnet Segment einem Signal zu (0, 1, oder 2)"""
        
        # Zu leise = ignorieren
        if rms < 0.001:
            return -1
        
        if len(self.signal_profiles) == 0:
            # Erstes Signal
            self.signal_profiles.append([centroid, rms])
            return 0
        
        # Finde nächstes Signal
        distances = []
        for i, (c_mean, r_mean) in enumerate(self.signal_profiles):
            # Gewichtete Distanz (Centroid wichtiger)
            c_dist = abs(centroid - c_mean) / max(c_mean, 100)
            r_dist = abs(rms - r_mean) / max(r_mean, 0.01)
            dist = 0.8 * c_dist + 0.2 * r_dist
            distances.append(dist)
        
        min_dist = min(distances)
        best_signal = distances.index(min_dist)
        
        # Wenn zu weit weg und Platz für neues Signal
        if min_dist > 0.4 and len(self.signal_profiles) < self.max_signals:
            # Neues Signal
            self.signal_profiles.append([centroid, rms])
            # Sortiere nach Centroid
            self.signal_profiles.sort(key=lambda x: x[0])
            return self.signal_profiles.index([centroid, rms])
        
        # Update Profil leicht (adaptiv)
        c_mean, r_mean = self.signal_profiles[best_signal]
        self.signal_profiles[best_signal] = [
            c_mean * (1 - self.adaptation_rate) + centroid * self.adaptation_rate,
            r_mean * (1 - self.adaptation_rate) + rms * self.adaptation_rate
        ]
        
        return best_signal


class RealtimeDeinterleaver:
    """Hauptklasse für Echtzeit-Deinterleaving"""
    
    def __init__(self, config: RealtimeConfig):
        self.cfg = config
        
        # Audio Setup
        self.sr = config.SAMPLE_RATE
        self.block_size = config.BLOCK_SIZE
        
        # Bandpass Filter
        self.bandpass = BandpassFilter(
            config.HIGHPASS_FREQ,
            config.LOWPASS_FREQ,
            self.sr,
            config.FILTER_ORDER
        )
        
        # Ring Buffer (600ms Latenz)
        buffer_samples = int(config.LATENCY_MS * self.sr / 1000)
        self.input_buffer = RingBuffer(buffer_samples)
        
        # Signal Detection
        self.signal_detected = False
        self.silence_counter = 0
        self.silence_threshold_blocks = int(
            config.SILENCE_TIMEOUT_MS / (config.BLOCK_SIZE / self.sr * 1000)
        )
        
        # Segmentierung
        self.segment_detector = SegmentDetector(self.sr, config.SEGMENT_DETECT_WINDOW_MS)
        self.classifier = SignalClassifier(config.MAX_SIGNALS)
        
        # Verarbeitungs-State
        self.current_signal = -1  # Welches Signal gerade läuft
        self.processing_window_samples = int(50 * self.sr / 1000)  # 50ms Window
        
        # Stats
        self.blocks_processed = 0
        self.last_stats_time = time.time()
        
        if config.VERBOSE:
            print("="*60)
            print("Echtzeit Audio Deinterleaver")
            print("="*60)
            print(f"Sample Rate: {self.sr} Hz")
            print(f"Block Size: {self.block_size} samples (~{self.block_size/self.sr*1000:.1f}ms)")
            print(f"Latenz: {config.LATENCY_MS}ms")
            print(f"Bandpass: {config.HIGHPASS_FREQ}-{config.LOWPASS_FREQ} Hz")
            print(f"Output Signal: #{config.OUTPUT_SIGNAL_INDEX}")
            print("="*60)
            print("Warte auf Eingangssignal...")
    
    def db_to_linear(self, db: float) -> float:
        """Konvertiert dB zu linearem Wert"""
        return 10 ** (db / 20)
    
    def detect_signal_start(self, audio_block: np.ndarray) -> bool:
        """Erkennt ob Signal anliegt"""
        rms = np.sqrt(np.mean(audio_block ** 2))
        threshold = self.db_to_linear(self.cfg.SIGNAL_THRESHOLD_DB)
        
        return rms > threshold
    
    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        """Verarbeitet einen Audio-Block"""
        
        # 1. Bandpass Filter
        filtered = self.bandpass.process(audio_block)
        
        # 2. Signal Detection
        if not self.signal_detected:
            if self.detect_signal_start(filtered):
                self.signal_detected = True
                if self.cfg.VERBOSE:
                    print("\n[SIGNAL DETECTED] Starte Verarbeitung...")
                self.input_buffer.write(filtered)
                return np.zeros_like(filtered)
            else:
                return np.zeros_like(filtered)
        
        # 3. Schreibe in Buffer
        self.input_buffer.write(filtered)
        
        # 4. Prüfe ob Buffer gefüllt ist
        if self.input_buffer.filled < self.processing_window_samples:
            return np.zeros_like(filtered)
        
        # 5. Hole Analyse-Window
        analysis_window = self.input_buffer.get_latest_window(self.processing_window_samples)
        
        # 6. Feature Extraction
        centroid, rms = self.segment_detector.compute_features(analysis_window)
        
        # 7. Segment Change Detection
        is_change = self.segment_detector.detect_change(centroid, rms, self.cfg.MIN_SEGMENT_MS)
        
        if is_change:
            # Klassifiziere neues Segment
            self.current_signal = self.classifier.classify_segment(centroid, rms)
            
            if self.cfg.VERBOSE and self.blocks_processed % 100 == 0:
                print(f"Signal #{self.current_signal} | Freq: {centroid:.0f}Hz | RMS: {rms:.4f}")
        
        # 8. Hole Output aus Buffer (mit Latenz)
        output = self.input_buffer.read(len(filtered))
        
        # 9. Filtere nur gewünschtes Signal
        if self.current_signal != self.cfg.OUTPUT_SIGNAL_INDEX:
            # Andere Signale stumm
            output = np.zeros_like(output)
        
        # 10. Silence Detection
        if rms < 0.001:
            self.silence_counter += 1
            if self.silence_counter > self.silence_threshold_blocks:
                if self.cfg.VERBOSE:
                    print("\n[SILENCE DETECTED] Stoppe Verarbeitung...")
                self.signal_detected = False
                self.silence_counter = 0
                self.classifier = SignalClassifier(self.cfg.MAX_SIGNALS)
                return np.zeros_like(output)
        else:
            self.silence_counter = 0
        
        self.blocks_processed += 1
        
        # Stats
        if self.cfg.VERBOSE and time.time() - self.last_stats_time > 5.0:
            print(f"[STATS] Blocks: {self.blocks_processed} | Signale: {len(self.classifier.signal_profiles)}")
            self.last_stats_time = time.time()
        
        return output
    
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Audio Callback für sounddevice"""
        if status:
            print(f"[WARNING] {status}")
        
        # Verarbeite
        input_audio = indata[:, 0]  # Mono
        output_audio = self.process_block(input_audio)
        
        # Output
        outdata[:, 0] = output_audio
        outdata[:, 1] = output_audio  # Stereo copy
    
    def start(self):
        """Startet Echtzeit-Verarbeitung"""
        try:
            with sd.Stream(
                samplerate=self.sr,
                blocksize=self.block_size,
                channels=2,
                callback=self.audio_callback
            ):
                print("\n[RUNNING] Drücke Ctrl+C zum Beenden\n")
                while True:
                    sd.sleep(1000)
        
        except KeyboardInterrupt:
            print("\n\n[STOPPED] Verarbeitung beendet")
        except Exception as e:
            print(f"\n[ERROR] {e}")


def main():
    # Konfiguration
    config = RealtimeConfig()
    
    # Optional: Überschreibe einzelne Parameter
    # config.OUTPUT_SIGNAL_INDEX = 1  # Gibt mittleres Signal aus
    # config.VERBOSE = False
    # config.MAX_SIGNALS = 2
    
    # Erstelle und starte Deinterleaver
    deinterleaver = RealtimeDeinterleaver(config)
    deinterleaver.start()


if __name__ == "__main__":
    main()