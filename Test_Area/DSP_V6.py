"""
Echtzeit Audio-Signal-Filter für Interleaved Signale
Filtert live ein gewähltes Signal (A, B, oder C) heraus

Unterschied zu V5.2

das system SOLL jetzt echtzeit fähig und kann entweder eine Audio datei einlesen oder direkt vom Aux

IST NICHT AUSREICHEND GETESTET
"""
import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple
import time

@dataclass
class Config:
    # Audio
    SAMPLE_RATE: int = 44100
    BLOCK_SIZE: int = 512  # Samples pro Block (~11ms @ 44.1kHz)
    BUFFER_MS: float = 200.0  # Latenz-Puffer
    
    # Feature Extraction
    WINDOW_MS: float = 20.0  # Feature-Fenster
    
    # Signal Classification
    MIN_CONFIDENCE: float = 0.6  # Mindest-Konfidenz für Signal-Klassifikation
    SMOOTHING_FRAMES: int = 3  # Glättung über N Frames
    
    # Signal Definitions (müssen trainiert/kalibriert werden)
    # Format: (min_freq, max_freq, label)
    SIGNAL_PROFILES: list = None  # Wird automatisch kalibriert
    
    # Output
    TARGET_SIGNAL: str = "C"  # Welches Signal soll ausgegeben werden? "A", "B", oder "C"
    FADE_MS: float = 5.0  # Crossfade bei Signal-Wechsel
    
    # Calibration
    CALIBRATION_SECONDS: float = 2.0  # Sekunden für Auto-Kalibrierung
    
    VERBOSE: bool = True

class SignalClassifier:
    """Klassifiziert Audio-Frames in Echtzeit"""
    
    def __init__(self, sr: int, window_samples: int):
        self.sr = sr
        self.window_samples = window_samples
        self.window = np.hanning(window_samples)
        self.freqs = np.fft.rfftfreq(window_samples, 1/sr)
        
        # Signal-Profile (werden während Kalibrierung gesetzt)
        self.profiles = []  # Liste von (min_freq, max_freq, label)
        self.calibrated = False
        
        # History für Smoothing
        self.recent_labels = deque(maxlen=5)
        self.recent_centroids = deque(maxlen=50)
    
    def extract_features(self, frame: np.ndarray) -> dict:
        """Extrahiert Features aus einem Frame"""
        if len(frame) < self.window_samples:
            frame = np.pad(frame, (0, self.window_samples - len(frame)))
        elif len(frame) > self.window_samples:
            frame = frame[:self.window_samples]
        
        # Windowing
        windowed = frame * self.window
        
        # FFT
        spec = np.abs(np.fft.rfft(windowed))
        spec_power = spec ** 2
        
        # Features
        rms = np.sqrt(np.mean(frame ** 2))
        
        if spec_power.sum() > 1e-10:
            centroid = np.sum(self.freqs * spec_power) / spec_power.sum()
            
            # Rolloff
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = self.freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Bandwidth
            bandwidth = np.sqrt(np.sum(((self.freqs - centroid) ** 2) * spec_power) / spec_power.sum())
        else:
            centroid = 0
            rolloff = 0
            bandwidth = 0
        
        return {
            'centroid': centroid,
            'rms': rms,
            'rolloff': rolloff,
            'bandwidth': bandwidth
        }
    
    def calibrate(self, calibration_data: list):
        """
        Kalibriert den Classifier basierend auf gesammelten Features
        calibration_data: Liste von feature dicts
        """
        if len(calibration_data) < 10:
            print("Nicht genug Daten für Kalibrierung!")
            return False
        
        # Extrahiere Centroids
        centroids = np.array([d['centroid'] for d in calibration_data])
        centroids = centroids[centroids > 100]  # Filtere sehr niedrige Werte
        
        if len(centroids) < 10:
            return False
        
        # K-Means Clustering auf Centroids
        n_clusters = 3  # Annahme: 3 Signale
        
        # Einfaches K-Means
        np.random.seed(42)
        
        # Init: min, median, max
        sorted_cents = np.sort(centroids)
        centers = np.array([
            sorted_cents[len(sorted_cents)//4],
            sorted_cents[len(sorted_cents)//2],
            sorted_cents[3*len(sorted_cents)//4]
        ])
        
        for _ in range(20):
            # Assign
            distances = np.abs(centroids[:, None] - centers[None, :])
            labels = np.argmin(distances, axis=1)
            
            # Update
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    centers[k] = centroids[mask].mean()
        
        # Erstelle Profile mit Toleranz
        self.profiles = []
        for k in range(n_clusters):
            mask = labels == k
            cluster_cents = centroids[mask]
            
            min_freq = cluster_cents.min() * 0.7  # 30% Toleranz nach unten
            max_freq = cluster_cents.max() * 1.3  # 30% Toleranz nach oben
            
            self.profiles.append((min_freq, max_freq, chr(ord('A') + k)))
        
        # Sortiere nach Frequenz
        self.profiles.sort(key=lambda x: (x[0] + x[1]) / 2)
        
        # Re-label
        for i, (min_f, max_f, _) in enumerate(self.profiles):
            self.profiles[i] = (min_f, max_f, chr(ord('A') + i))
        
        self.calibrated = True
        
        print("\n[KALIBRIERUNG ABGESCHLOSSEN]")
        for min_f, max_f, label in self.profiles:
            print(f"  Signal {label}: {min_f:.0f} - {max_f:.0f} Hz")
        
        return True
    
    def classify(self, features: dict, min_confidence: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Klassifiziert Features zu einem Signal-Label
        Returns: (label, confidence) oder (None, 0.0)
        """
        if not self.calibrated:
            return None, 0.0
        
        centroid = features['centroid']
        rms = features['rms']
        
        # Zu leise = Stille
        if rms < 0.001:
            return None, 0.0
        
        # Finde passendes Profil
        best_label = None
        best_confidence = 0.0
        
        for min_freq, max_freq, label in self.profiles:
            if min_freq <= centroid <= max_freq:
                # Confidence basierend auf Position im Range
                center = (min_freq + max_freq) / 2
                range_width = max_freq - min_freq
                distance = abs(centroid - center)
                confidence = 1.0 - (distance / (range_width / 2))
                confidence = max(0.0, min(1.0, confidence))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label
        
        # Smoothing über letzte Frames
        if best_label and best_confidence >= min_confidence:
            self.recent_labels.append(best_label)
            self.recent_centroids.append(centroid)
            
            # Mehrheitsvotum
            if len(self.recent_labels) >= 3:
                labels_list = list(self.recent_labels)
                most_common = max(set(labels_list), key=labels_list.count)
                count = labels_list.count(most_common)
                smoothed_confidence = count / len(labels_list)
                
                return most_common, smoothed_confidence
        
        return best_label, best_confidence

class RealtimeAudioFilter:
    """Echtzeit Audio-Filter für interleaved Signale"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.running = False
        self.calibrating = False
        
        # Berechne Puffer-Größe
        buffer_samples = int(cfg.BUFFER_MS * cfg.SAMPLE_RATE / 1000)
        self.buffer = deque(maxlen=buffer_samples)
        
        # Feature Extraction
        window_samples = int(cfg.WINDOW_MS * cfg.SAMPLE_RATE / 1000)
        self.classifier = SignalClassifier(cfg.SAMPLE_RATE, window_samples)
        
        # Calibration Buffer
        self.calibration_buffer = []
        self.calibration_start = None
        
        # State
        self.current_signal = None
        self.fade_counter = 0
        self.fade_samples = int(cfg.FADE_MS * cfg.SAMPLE_RATE / 1000)
        
        # Stats
        self.blocks_processed = 0
        self.signal_changes = 0
        
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Callback für Audio-Stream"""
        if status:
            print(f"Status: {status}")
        
        # Input ist (frames, channels), wir brauchen mono
        audio_in = indata[:, 0] if indata.ndim > 1 else indata
        
        # Füge zu Buffer hinzu
        self.buffer.extend(audio_in)
        
        # Feature Extraction auf aktuellem Block
        features = self.classifier.extract_features(audio_in)
        
        # Kalibrierungsmodus
        if self.calibrating:
            self.calibration_buffer.append(features)
            
            # Prüfe ob Kalibrierung fertig
            elapsed = time.time() - self.calibration_start
            if elapsed >= self.cfg.CALIBRATION_SECONDS:
                self.classifier.calibrate(self.calibration_buffer)
                self.calibrating = False
                print("\n[LIVE] Starte Audio-Filterung...")
            
            # Während Kalibrierung: Pass-through
            outdata[:] = indata
            return
        
        # Klassifiziere Signal
        label, confidence = self.classifier.classify(features, self.cfg.MIN_CONFIDENCE)
        
        # Entscheide ob wir Signal wechseln
        if label and label != self.current_signal:
            self.current_signal = label
            self.signal_changes += 1
            self.fade_counter = 0
        
        # Output generieren
        if self.current_signal == self.cfg.TARGET_SIGNAL and confidence > self.cfg.MIN_CONFIDENCE:
            # Pass-through mit optional Fade
            if self.fade_counter < self.fade_samples:
                # Fade in
                fade = self.fade_counter / self.fade_samples
                outdata[:, 0] = audio_in * fade
                self.fade_counter += len(audio_in)
            else:
                outdata[:, 0] = audio_in
        else:
            # Mute (anderes Signal oder zu geringe Confidence)
            if self.fade_counter < self.fade_samples and self.current_signal == self.cfg.TARGET_SIGNAL:
                # Fade out
                fade = 1.0 - (self.fade_counter / self.fade_samples)
                outdata[:, 0] = audio_in * fade
                self.fade_counter += len(audio_in)
            else:
                outdata[:, 0] = 0
        
        self.blocks_processed += 1
        
        # Status-Output (alle 100 Blocks)
        if self.cfg.VERBOSE and self.blocks_processed % 100 == 0:
            print(f"\rSignal: {self.current_signal or '?'} | "
                  f"Centroid: {features['centroid']:.0f} Hz | "
                  f"Confidence: {confidence:.2f} | "
                  f"Wechsel: {self.signal_changes}", end='')
    
    def start(self):
        """Startet Echtzeit-Verarbeitung"""
        print("="*60)
        print("Echtzeit Audio-Signal-Filter")
        print("="*60)
        print(f"\nTarget-Signal: {self.cfg.TARGET_SIGNAL}")
        print(f"Sample Rate: {self.cfg.SAMPLE_RATE} Hz")
        print(f"Block Size: {self.cfg.BLOCK_SIZE} samples")
        print(f"Latenz: {self.cfg.BUFFER_MS} ms")
        
        # Starte Kalibrierung
        print(f"\n[KALIBRIERUNG] Erfasse {self.cfg.CALIBRATION_SECONDS}s Audio...")
        print("→ Stelle sicher, dass alle Signale im Input vorkommen!")
        
        self.calibrating = True
        self.calibration_start = time.time()
        
        # Starte Stream
        self.running = True
        
        try:
            with sd.Stream(
                samplerate=self.cfg.SAMPLE_RATE,
                blocksize=self.cfg.BLOCK_SIZE,
                channels=1,
                callback=self.audio_callback
            ):
                print(f"\n{'='*60}")
                print("Stream läuft... Drücke Ctrl+C zum Beenden")
                print(f"{'='*60}\n")
                
                while self.running:
                    sd.sleep(100)
                    
        except KeyboardInterrupt:
            print("\n\n[STOP] Beende Stream...")
        except Exception as e:
            print(f"\n\nFehler: {e}")
        finally:
            self.running = False
            
        print(f"\n{'='*60}")
        print(f"Blocks verarbeitet: {self.blocks_processed}")
        print(f"Signal-Wechsel: {self.signal_changes}")
        print(f"{'='*60}\n")

# ============================================================================
# DATEI-BASIERTE SIMULATION (Alternative)
# ============================================================================

class FileBasedRealtimeFilter:
    """
    Simuliert Echtzeit-Verarbeitung mit einer Audio-Datei
    Nützlich zum Testen ohne Mikrofon
    """
    
    def __init__(self, cfg: Config, input_file: str):
        self.cfg = cfg
        self.input_file = input_file
        
        # Audio laden
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_channels(1).set_frame_rate(cfg.SAMPLE_RATE)
        samples = np.array(audio.get_array_of_samples())
        
        if audio.sample_width == 2:
            self.audio_data = samples.astype(np.float32) / 32768.0
        else:
            self.audio_data = samples.astype(np.float32) / 2147483648.0
        
        self.sr = cfg.SAMPLE_RATE
        
        # Classifier
        window_samples = int(cfg.WINDOW_MS * cfg.SAMPLE_RATE / 1000)
        self.classifier = SignalClassifier(cfg.SAMPLE_RATE, window_samples)
        
        print(f"Geladen: {input_file}")
        print(f"Dauer: {len(self.audio_data) / self.sr:.2f}s")
    
    def process(self):
        """Verarbeitet die gesamte Datei"""
        
        # Kalibrierung
        print("\n[1/3] Kalibrierung...")
        calib_samples = int(self.cfg.CALIBRATION_SECONDS * self.sr)
        calib_audio = self.audio_data[:calib_samples]
        
        # Extrahiere Features für Kalibrierung
        block_size = self.cfg.BLOCK_SIZE
        calib_features = []
        
        for i in range(0, len(calib_audio), block_size):
            block = calib_audio[i:i+block_size]
            if len(block) < block_size:
                break
            feats = self.classifier.extract_features(block)
            calib_features.append(feats)
        
        if not self.classifier.calibrate(calib_features):
            print("Kalibrierung fehlgeschlagen!")
            return None
        
        # Verarbeite Rest
        print("\n[2/3] Verarbeite Audio...")
        output = np.zeros_like(self.audio_data)
        
        current_signal = None
        fade_counter = 0
        fade_samples = int(self.cfg.FADE_MS * self.sr / 1000)
        
        for i in range(0, len(self.audio_data), block_size):
            block = self.audio_data[i:i+block_size]
            if len(block) < block_size:
                break
            
            # Klassifiziere
            features = self.classifier.extract_features(block)
            label, confidence = self.classifier.classify(features, self.cfg.MIN_CONFIDENCE)
            
            if label and label != current_signal:
                current_signal = label
                fade_counter = 0
            
            # Output
            if current_signal == self.cfg.TARGET_SIGNAL and confidence > self.cfg.MIN_CONFIDENCE:
                if fade_counter < fade_samples:
                    fade = fade_counter / fade_samples
                    output[i:i+len(block)] = block * fade
                    fade_counter += len(block)
                else:
                    output[i:i+len(block)] = block
            else:
                if fade_counter < fade_samples and current_signal == self.cfg.TARGET_SIGNAL:
                    fade = 1.0 - (fade_counter / fade_samples)
                    output[i:i+len(block)] = block * fade
                    fade_counter += len(block)
                else:
                    output[i:i+len(block)] = 0
            
            # Progress
            if i % (self.sr * 2) == 0:
                progress = i / len(self.audio_data) * 100
                print(f"\rFortschritt: {progress:.1f}% | Signal: {current_signal or '?'}", end='')
        
        print("\n")
        return output
    
    def save_output(self, output: np.ndarray, output_path: str):
        """Speichert gefilterte Audio"""
        print(f"[3/3] Speichere: {output_path}")
        
        from pydub import AudioSegment
        
        output = np.clip(output, -1.0, 1.0)
        int16 = (output * 32767.0).astype(np.int16)
        
        seg = AudioSegment(
            data=int16.tobytes(),
            sample_width=2,
            frame_rate=self.sr,
            channels=1
        )
        
        seg.export(output_path, format="mp3", bitrate="192k")
        print(f"✓ Gespeichert: {output_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()
    
    # Wähle Modus
    print("Echtzeit Audio-Filter")
    print("=" * 60)
    print("\nModus wählen:")
    print("  [1] Live (Mikrofon → Lautsprecher)")
    print("  [2] Datei (MP3 → gefilterte MP3)")
    
    mode = input("\nModus (1 oder 2): ").strip()
    
    if mode == "1":
        # Live-Modus
        filter_obj = RealtimeAudioFilter(cfg)
        filter_obj.start()
        
    elif mode == "2":
        # Datei-Modus
        input_file = input("Input-Datei: ").strip()
        if not input_file:
            input_file = r"C:/eigene Programme/VS_Code_Programme/HKA/DSP/Inputsignale/50ms/interleaved_1k_8k_vio_50ms.mp3"
        
        target = input(f"Target-Signal ({cfg.TARGET_SIGNAL}): ").strip().upper() or cfg.TARGET_SIGNAL
        cfg.TARGET_SIGNAL = target
        
        filter_obj = FileBasedRealtimeFilter(cfg, input_file)
        output = filter_obj.process()
        
        if output is not None:
            print("Out_put erkannt")
            output_file = f"filtered_signal_{cfg.TARGET_SIGNAL}.mp3"
            filter_obj.save_output(output, output_file)
    
    else:
        print("Ungültiger Modus!")

if __name__ == "__main__":
    main()