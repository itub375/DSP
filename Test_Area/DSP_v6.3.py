"""


ÄNDERUNG ZU V6.2
Es gibt jetzt einen Oszilator Plot zum anschauen der Echtzeitdaten
"""

"""
Echtzeit Audio Deinterleaver mit Live Oszilloskop
Liest Audio vom AUX-Eingang, erkennt interleaved Signale und gibt nur eines gefiltert aus

Anforderungen:
- 600ms Latenz-Puffer für Verarbeitung
- 20Hz - 20kHz Bandpass Filter
- Automatische Signal-Detektion
- Segmente: 10-50ms
- Max 3 Signale (optional)
- Live Oszilloskop Visualisierung
"""

import numpy as np
import sounddevice as sd
from scipy import signal
from collections import deque
from dataclasses import dataclass
import time
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

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
    
    # Visualisierung
    ENABLE_PLOT: bool = True
    PLOT_WINDOW_MS: float = 200.0  # Zeitfenster im Oszilloskop
    PLOT_UPDATE_FPS: int = 30      # Update-Rate des Plots
    
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


class OscilloscopeVisualizer:
    """Live Oszilloskop Visualisierung"""
    
    def __init__(self, sr: int, window_ms: float = 200.0, update_fps: int = 30):
        self.sr = sr
        self.window_samples = int(window_ms * sr / 1000)
        self.update_interval = 1000 / update_fps  # ms
        
        # Data Buffers
        self.input_buffer = deque(maxlen=self.window_samples)
        self.output_buffer = deque(maxlen=self.window_samples)
        
        # Metadata
        self.current_signal = -1
        self.detected_signals = 0
        self.current_freq = 0.0
        self.signal_active = False
        
        # Thread-safe lock
        self.lock = threading.Lock()
        
        # Matplotlib Setup
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('🎵 Echtzeit Audio Deinterleaver - Oszilloskop', 
                         fontsize=16, fontweight='bold')
        
        # Time axis
        self.time_axis = np.linspace(0, window_ms, self.window_samples)
        
        # Plot 1: Input Signal
        self.line_input, = self.ax1.plot(self.time_axis, np.zeros(self.window_samples), 
                                         'cyan', linewidth=1.5, label='Eingangssignal')
        self.ax1.set_ylim(-1.0, 1.0)
        self.ax1.set_ylabel('Amplitude', fontsize=11)
        self.ax1.set_title('📥 Eingangssignal (gefiltert 20Hz-20kHz)', 
                          fontsize=12, loc='left')
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        self.ax1.legend(loc='upper right')
        
        # Plot 2: Output Signal
        self.line_output, = self.ax2.plot(self.time_axis, np.zeros(self.window_samples), 
                                          'lime', linewidth=1.5, label='Ausgangssignal')
        self.ax2.set_ylim(-1.0, 1.0)
        self.ax2.set_ylabel('Amplitude', fontsize=11)
        self.ax2.set_title('📤 Ausgangssignal (gefiltertes Signal)', 
                          fontsize=12, loc='left')
        self.ax2.grid(True, alpha=0.3, linestyle='--')
        self.ax2.legend(loc='upper right')
        
        # Plot 3: Frequency Spectrum
        self.freq_axis = np.fft.rfftfreq(self.window_samples, 1/sr)
        self.line_spectrum, = self.ax3.plot(self.freq_axis, 
                                            np.zeros(len(self.freq_axis)), 
                                            'yellow', linewidth=1.5)
        self.ax3.set_xlim(0, 5000)  # Zeige bis 5kHz
        self.ax3.set_ylim(0, 0.1)
        self.ax3.set_xlabel('Frequenz (Hz)', fontsize=11)
        self.ax3.set_ylabel('Magnitude', fontsize=11)
        self.ax3.set_title('🎼 Frequenzspektrum (Ausgang)', fontsize=12, loc='left')
        self.ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Info Text
        self.info_text = self.fig.text(0.02, 0.02, '', fontsize=10, 
                                       family='monospace', color='white',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        # Animation
        self.ani = FuncAnimation(self.fig, self.update_plot, 
                                interval=self.update_interval, 
                                blit=False, cache_frame_data=False)
    
    def update_data(self, input_data: np.ndarray, output_data: np.ndarray,
                   current_signal: int, detected_signals: int, 
                   current_freq: float, signal_active: bool):
        """Thread-safe Update der Plot-Daten"""
        with self.lock:
            self.input_buffer.extend(input_data)
            self.output_buffer.extend(output_data)
            self.current_signal = current_signal
            self.detected_signals = detected_signals
            self.current_freq = current_freq
            self.signal_active = signal_active
    
    def update_plot(self, frame):
        """Update Plot (wird von Animation aufgerufen)"""
        with self.lock:
            # Hole Daten
            if len(self.input_buffer) == 0:
                return
            
            input_data = np.array(list(self.input_buffer))
            output_data = np.array(list(self.output_buffer))
            
            # Fülle mit Nullen wenn zu kurz
            if len(input_data) < self.window_samples:
                input_data = np.pad(input_data, 
                                   (self.window_samples - len(input_data), 0))
            if len(output_data) < self.window_samples:
                output_data = np.pad(output_data, 
                                    (self.window_samples - len(output_data), 0))
            
            # Update Input
            self.line_input.set_ydata(input_data)
            
            # Update Output
            self.line_output.set_ydata(output_data)
            
            # Update Spectrum
            if np.any(output_data):
                window = np.hanning(len(output_data))
                spectrum = np.abs(np.fft.rfft(output_data * window))
                spectrum = spectrum / (len(output_data) / 2)
                
                # Smooth spectrum
                if len(spectrum) > 10:
                    kernel_size = 5
                    kernel = np.ones(kernel_size) / kernel_size
                    spectrum = np.convolve(spectrum, kernel, mode='same')
                
                self.line_spectrum.set_ydata(spectrum)
                
                # Auto-scale Y
                max_val = np.max(spectrum[:int(5000 * len(spectrum) / (self.sr/2))])
                if max_val > 0:
                    self.ax3.set_ylim(0, max_val * 1.2)
            
            # Update Info Text
            status = "🟢 AKTIV" if self.signal_active else "🔴 WARTE"
            signal_name = chr(ord('A') + self.current_signal) if self.current_signal >= 0 else "---"
            
            info = f"""
STATUS: {status}
Erkannte Signale: {self.detected_signals}
Aktuelles Segment: Signal {signal_name}
Frequenz: {self.current_freq:.0f} Hz
Output Signal: #{self.current_signal if self.current_signal >= 0 else '---'}
            """.strip()
            
            self.info_text.set_text(info)
    
    def show(self):
        """Zeigt Plot (blockierend)"""
        plt.show()


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
        self.current_signal = -1
        self.current_freq = 0.0
        self.processing_window_samples = int(50 * self.sr / 1000)
        
        # Visualisierung
        self.visualizer = None
        if config.ENABLE_PLOT:
            self.visualizer = OscilloscopeVisualizer(
                self.sr, 
                config.PLOT_WINDOW_MS, 
                config.PLOT_UPDATE_FPS
            )
        
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
            print(f"Visualisierung: {'AN' if config.ENABLE_PLOT else 'AUS'}")
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
                
                # Update Visualizer
                if self.visualizer:
                    self.visualizer.update_data(filtered, np.zeros_like(filtered),
                                              -1, 0, 0.0, True)
                
                return np.zeros_like(filtered)
            else:
                # Update Visualizer (nur Input)
                if self.visualizer:
                    self.visualizer.update_data(filtered, np.zeros_like(filtered),
                                              -1, 0, 0.0, False)
                return np.zeros_like(filtered)
        
        # 3. Schreibe in Buffer
        self.input_buffer.write(filtered)
        
        # 4. Prüfe ob Buffer gefüllt ist
        if self.input_buffer.filled < self.processing_window_samples:
            if self.visualizer:
                self.visualizer.update_data(filtered, np.zeros_like(filtered),
                                          -1, 0, 0.0, True)
            return np.zeros_like(filtered)
        
        # 5. Hole Analyse-Window
        analysis_window = self.input_buffer.get_latest_window(self.processing_window_samples)
        
        # 6. Feature Extraction
        centroid, rms = self.segment_detector.compute_features(analysis_window)
        self.current_freq = centroid
        
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
            output = np.zeros_like(output)
        
        # 10. Update Visualizer
        if self.visualizer:
            self.visualizer.update_data(
                filtered, output,
                self.current_signal, 
                len(self.classifier.signal_profiles),
                self.current_freq,
                True
            )
        
        # 11. Silence Detection
        if rms < 0.001:
            self.silence_counter += 1
            if self.silence_counter > self.silence_threshold_blocks:
                if self.cfg.VERBOSE:
                    print("\n[SILENCE DETECTED] Stoppe Verarbeitung...")
                self.signal_detected = False
                self.silence_counter = 0
                self.classifier = SignalClassifier(self.cfg.MAX_SIGNALS)
                
                if self.visualizer:
                    self.visualizer.update_data(filtered, np.zeros_like(output),
                                              -1, 0, 0.0, False)
                
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
        
        # Starte Audio in separatem Thread
        audio_thread = threading.Thread(target=self._run_audio, daemon=True)
        audio_thread.start()
        
        # Zeige Plot (blockiert Main Thread)
        if self.visualizer:
            print("\n[PLOT] Oszilloskop wird gestartet...")
            self.visualizer.show()
        else:
            # Warte endlos
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n[STOPPED] Verarbeitung beendet")
    
    def _run_audio(self):
        """Läuft in separatem Thread"""
        try:
            with sd.Stream(
                samplerate=self.sr,
                blocksize=self.block_size,
                channels=2,
                callback=self.audio_callback
            ):
                print("\n[RUNNING] Audio-Stream aktiv")
                print("Schließe das Plot-Fenster zum Beenden\n")
                while True:
                    sd.sleep(1000)
        
        except Exception as e:
            print(f"\n[ERROR] {e}")


def main():
    # Konfiguration
    config = RealtimeConfig()
    
    # Optional: Parameter überschreiben
    # config.OUTPUT_SIGNAL_INDEX = 1  # Mittleres Signal
    # config.VERBOSE = False
    # config.ENABLE_PLOT = False  # Plot deaktivieren
    # config.PLOT_WINDOW_MS = 300.0  # Längeres Zeitfenster
    
    # Erstelle und starte Deinterleaver
    deinterleaver = RealtimeDeinterleaver(config)
    deinterleaver.start()


if __name__ == "__main__":
    main()