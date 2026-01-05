"""
Zeitliches Audio-Deinterleaving System
Rekonstruiert ursprüngliche Audiokanäle aus verschachtelten Signalen
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AudioSegment:
    """Repräsentiert ein Audio-Segment mit Metadaten"""
    data: np.ndarray
    start_idx: int
    end_idx: int
    energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    source_id: Optional[int] = None


class AudioSource:
    """Erzeugt verschiedene Audio-Quellen für Tests"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
    
    def generate_speech_like(self, duration: float) -> np.ndarray:
        """Simuliert sprach-ähnliches Signal mit Formanten"""
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Grundfrequenz variiert (Pitch)
        f0 = 120 + 30 * np.sin(2 * np.pi * 3 * t)
        
        # Formanten (vereinfacht)
        formant1 = np.sin(2 * np.pi * f0 * t)
        formant2 = 0.5 * np.sin(2 * np.pi * 850 * t)
        formant3 = 0.3 * np.sin(2 * np.pi * 1200 * t)
        
        speech = formant1 + formant2 + formant3
        
        # Amplituden-Modulation (Silben)
        envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 4 * t))
        speech *= envelope
        
        return speech * 0.3
    
    def generate_music(self, duration: float) -> np.ndarray:
        """Erzeugt musik-ähnliches Signal"""
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Akkord mit mehreren Harmonien
        notes = [262, 330, 392]  # C-Dur
        music = sum(np.sin(2 * np.pi * f * t) for f in notes) / len(notes)
        
        # Rhythmische Struktur
        rhythm = np.abs(np.sin(2 * np.pi * 2 * t))
        music *= rhythm
        
        return music * 0.4
    
    def generate_noise(self, duration: float) -> np.ndarray:
        """Erzeugt farbiges Rauschen"""
        samples = int(self.sr * duration)
        white = np.random.randn(samples)
        
        # Rosa Rauschen durch Filterung
        b, a = signal.butter(4, 0.3, btype='low')
        pink = signal.filtfilt(b, a, white)
        
        return pink * 0.2


class InterleavedSignalGenerator:
    """Erzeugt verschachtelte Audio-Signale"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.source = AudioSource(sample_rate)
    
    def create_interleaved_signal(
        self,
        duration: float = 10.0,
        segment_length_ms: float = 30.0,
        n_sources: int = 2,
        add_silence: bool = False,
        vary_length: bool = False,
        add_level_jumps: bool = False
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Erzeugt verschachteltes Signal aus mehreren Quellen
        
        Returns:
            interleaved: Das verschachtelte Signal
            sources: Liste der Original-Quellen
        """
        segment_samples = int(self.sr * segment_length_ms / 1000)
        
        # Erzeuge Quell-Signale
        sources = []
        if n_sources >= 1:
            sources.append(self.source.generate_speech_like(duration))
        if n_sources >= 2:
            sources.append(self.source.generate_music(duration))
        if n_sources >= 3:
            sources.append(self.source.generate_noise(duration))
        
        # Berechne Anzahl der Segmente
        min_length = min(len(s) for s in sources)
        n_segments_per_source = min_length // segment_samples
        
        interleaved = []
        
        for seg_idx in range(n_segments_per_source):
            for src_idx in range(n_sources):
                # Segment-Länge variieren
                if vary_length:
                    var_samples = int(segment_samples * (0.8 + 0.4 * np.random.rand()))
                else:
                    var_samples = segment_samples
                
                start = seg_idx * segment_samples
                end = start + var_samples
                
                if end > len(sources[src_idx]):
                    break
                
                segment = sources[src_idx][start:end].copy()
                
                # Pegelsprünge hinzufügen
                if add_level_jumps and np.random.rand() > 0.7:
                    segment *= (0.5 + np.random.rand())
                
                interleaved.append(segment)
                
                # Stille hinzufügen
                if add_silence and np.random.rand() > 0.8:
                    silence_len = int(self.sr * 0.01 * np.random.rand())
                    interleaved.append(np.zeros(silence_len))
        
        # Zusammenfügen
        interleaved_signal = np.concatenate(interleaved)
        
        return interleaved_signal, sources


class AudioDeinterleaver:
    """Hauptklasse für Audio-Deinterleaving"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 256
    ):
        self.sr = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.segments: List[AudioSegment] = []
        self.n_sources = 2
        
    def extract_features(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """Extrahiert Merkmale aus einem Audio-Segment"""
        
        # Energie
        energy = np.sum(audio ** 2) / len(audio)
        
        # Spectral Centroid
        spectrum = np.abs(rfft(audio * np.hamming(len(audio))))
        freqs = rfftfreq(len(audio), 1/self.sr)
        spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
        
        # Zero Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        
        return energy, spectral_centroid, zero_crossings
    
    def detect_segment_boundaries(
        self,
        audio: np.ndarray,
        energy_threshold: float = 0.001,
        min_segment_length: int = 160
    ) -> List[Tuple[int, int]]:
        """
        Erkennt Segmentgrenzen basierend auf Energie und spektralen Änderungen
        """
        # Energie über kurze Fenster
        window_size = 256
        hop = 128
        
        energies = []
        for i in range(0, len(audio) - window_size, hop):
            frame = audio[i:i+window_size]
            energy = np.sum(frame ** 2) / window_size
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Finde abrupte Energie-Änderungen
        energy_diff = np.abs(np.diff(energies))
        energy_diff = np.concatenate([[0], energy_diff])
        
        # Normalisierung
        if np.max(energy_diff) > 0:
            energy_diff /= np.max(energy_diff)
        
        # Schwellwert für Grenzen
        threshold = np.mean(energy_diff) + 2 * np.std(energy_diff)
        threshold = max(threshold, 0.3)
        
        boundaries = []
        last_boundary = 0
        
        for i, diff in enumerate(energy_diff):
            if diff > threshold:
                sample_idx = i * hop
                if sample_idx - last_boundary >= min_segment_length:
                    boundaries.append(sample_idx)
                    last_boundary = sample_idx
        
        # Konvertiere zu Segment-Paaren
        segment_bounds = []
        for i in range(len(boundaries) - 1):
            segment_bounds.append((boundaries[i], boundaries[i+1]))
        
        # Letztes Segment
        if boundaries:
            segment_bounds.append((boundaries[-1], len(audio)))
        
        return segment_bounds
    
    def segment_audio(self, audio: np.ndarray) -> List[AudioSegment]:
        """Segmentiert Audio und extrahiert Merkmale"""
        boundaries = self.detect_segment_boundaries(audio)
        segments = []
        
        for start, end in boundaries:
            if end - start < 160:  # Zu kurz
                continue
            
            segment_data = audio[start:end]
            energy, sc, zcr = self.extract_features(segment_data)
            
            seg = AudioSegment(
                data=segment_data,
                start_idx=start,
                end_idx=end,
                energy=energy,
                spectral_centroid=sc,
                zero_crossing_rate=zcr
            )
            segments.append(seg)
        
        return segments
    
    def classify_segments(
        self,
        segments: List[AudioSegment],
        n_sources: int = 2
    ) -> List[AudioSegment]:
        """
        Klassifiziert Segmente mittels K-Means Clustering
        """
        if not segments:
            return segments
        
        # Feature-Matrix erstellen
        features = np.array([
            [seg.energy, seg.spectral_centroid, seg.zero_crossing_rate]
            for seg in segments
        ])
        
        # Normalisierung
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
        
        # Einfaches K-Means (manuell für Embedded-Kompatibilität)
        n_clusters = min(n_sources, len(segments))
        
        # Initialisiere Zentroide zufällig
        indices = np.random.choice(len(features), n_clusters, replace=False)
        centroids = features[indices]
        
        # K-Means Iterationen
        for _ in range(20):
            # Zuordnung zu nächstem Zentroid
            distances = np.zeros((len(features), n_clusters))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.sum((features - centroid) ** 2, axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            # Update Zentroide
            new_centroids = np.zeros_like(centroids)
            for i in range(n_clusters):
                cluster_points = features[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Zuordnung zu Segmenten
        for seg, label in zip(segments, labels):
            seg.source_id = int(label)
        
        return segments
    
    def reconstruct_source(
        self,
        segments: List[AudioSegment],
        source_id: int
    ) -> np.ndarray:
        """Rekonstruiert einen Audiokanal aus Segmenten"""
        source_segments = [seg for seg in segments if seg.source_id == source_id]
        
        if not source_segments:
            return np.array([])
        
        # Sortiere nach Start-Index
        source_segments.sort(key=lambda s: s.start_idx)
        
        # Zusammensetzen mit Cross-Fade
        reconstructed = []
        fade_len = min(160, len(source_segments[0].data) // 4)
        
        for i, seg in enumerate(source_segments):
            if i == 0:
                reconstructed.append(seg.data)
            else:
                # Cross-fade zwischen Segmenten
                if len(reconstructed) > 0 and fade_len > 0:
                    fade_out = np.linspace(1, 0, fade_len)
                    fade_in = np.linspace(0, 1, fade_len)
                    
                    prev_data = np.concatenate(reconstructed)
                    if len(prev_data) >= fade_len and len(seg.data) >= fade_len:
                        prev_data[-fade_len:] *= fade_out
                        seg.data[:fade_len] *= fade_in
                        reconstructed[-1] = np.concatenate([
                            reconstructed[-1][:-fade_len],
                            prev_data[-fade_len:] + seg.data[:fade_len],
                            seg.data[fade_len:]
                        ])
                        continue
                
                reconstructed.append(seg.data)
        
        return np.concatenate(reconstructed) if reconstructed else np.array([])
    
    def process(self, audio: np.ndarray, n_sources: int = 2) -> List[np.ndarray]:
        """Haupt-Verarbeitungsfunktion"""
        self.n_sources = n_sources
        
        # Segmentierung
        segments = self.segment_audio(audio)
        print(f"Erkannte Segmente: {len(segments)}")
        
        # Klassifikation
        segments = self.classify_segments(segments, n_sources)
        
        # Rekonstruktion aller Quellen
        reconstructed_sources = []
        for source_id in range(n_sources):
            reconstructed = self.reconstruct_source(segments, source_id)
            reconstructed_sources.append(reconstructed)
            print(f"Quelle {source_id}: {len(reconstructed)} Samples rekonstruiert")
        
        self.segments = segments
        return reconstructed_sources


class AudioEvaluator:
    """Evaluiert die Deinterleaving-Performance"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
    
    def plot_results(
        self,
        interleaved: np.ndarray,
        original_sources: List[np.ndarray],
        reconstructed_sources: List[np.ndarray],
        segments: List[AudioSegment]
    ):
        """Visualisiert Ergebnisse"""
        n_sources = len(original_sources)
        
        fig, axes = plt.subplots(n_sources + 2, 1, figsize=(14, 3 * (n_sources + 2)))
        
        # Interleaved Signal
        time = np.arange(len(interleaved)) / self.sr
        axes[0].plot(time, interleaved, linewidth=0.5)
        axes[0].set_title('Interleaved Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Segment-Grenzen einzeichnen
        for seg in segments[:50]:  # Erste 50 zur Übersicht
            axes[0].axvline(seg.start_idx / self.sr, color='red', alpha=0.3, linewidth=0.5)
        
        # Original vs Rekonstruiert
        for i in range(n_sources):
            if i < len(reconstructed_sources):
                # Original
                time_orig = np.arange(len(original_sources[i])) / self.sr
                axes[i+1].plot(time_orig, original_sources[i], 
                             linewidth=0.5, alpha=0.7, label='Original')
                
                # Rekonstruiert
                if len(reconstructed_sources[i]) > 0:
                    time_rec = np.arange(len(reconstructed_sources[i])) / self.sr
                    axes[i+1].plot(time_rec, reconstructed_sources[i], 
                                 linewidth=0.5, alpha=0.7, label='Rekonstruiert')
                
                axes[i+1].set_title(f'Quelle {i}')
                axes[i+1].set_ylabel('Amplitude')
                axes[i+1].legend()
                axes[i+1].grid(True, alpha=0.3)
        
        # Feature Space
        if segments:
            energies = [seg.energy for seg in segments]
            centroids = [seg.spectral_centroid for seg in segments]
            colors = [seg.source_id if seg.source_id is not None else 0 
                     for seg in segments]
            
            scatter = axes[-1].scatter(energies, centroids, c=colors, 
                                      cmap='viridis', alpha=0.6, s=20)
            axes[-1].set_xlabel('Energy')
            axes[-1].set_ylabel('Spectral Centroid (Hz)')
            axes[-1].set_title('Feature Space (Clustering)')
            axes[-1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[-1], label='Source ID')
        
        plt.tight_layout()
        plt.savefig('deinterleaving_results.png', dpi=150, bbox_inches='tight')
        print("Ergebnisse gespeichert in 'deinterleaving_results.png'")
        plt.show()
    
    def calculate_metrics(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> dict:
        """Berechnet Qualitätsmetriken"""
        if len(reconstructed) == 0:
            return {'snr': -np.inf, 'correlation': 0.0}
        
        # Längen anpassen
        min_len = min(len(original), len(reconstructed))
        orig = original[:min_len]
        recon = reconstructed[:min_len]
        
        # SNR
        signal_power = np.sum(orig ** 2)
        noise_power = np.sum((orig - recon) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Korrelation
        correlation = np.corrcoef(orig, recon)[0, 1]
        
        return {
            'snr': snr,
            'correlation': correlation
        }


def main():
    """Hauptfunktion - Demonstriert das System"""
    print("=" * 60)
    print("Audio Deinterleaving System")
    print("=" * 60)
    
    # Parameter
    SAMPLE_RATE = 16000
    DURATION = 5.0
    SEGMENT_LENGTH_MS = 30.0
    N_SOURCES = 2
    
    # 1. Signal erzeugen
    print("\n1. Erzeuge interleaved Signal...")
    generator = InterleavedSignalGenerator(SAMPLE_RATE)
    
    interleaved, original_sources = generator.create_interleaved_signal(
        duration=DURATION,
        segment_length_ms=SEGMENT_LENGTH_MS,
        n_sources=N_SOURCES,
        add_silence=False,
        vary_length=False,
        add_level_jumps=False
    )
    
    print(f"   Interleaved Signal: {len(interleaved)} samples ({len(interleaved)/SAMPLE_RATE:.2f}s)")
    
    # 2. Deinterleaving
    print("\n2. Führe Deinterleaving durch...")
    deinterleaver = AudioDeinterleaver(SAMPLE_RATE)
    reconstructed_sources = deinterleaver.process(interleaved, N_SOURCES)
    
    # 3. Evaluation
    print("\n3. Evaluierung...")
    evaluator = AudioEvaluator(SAMPLE_RATE)
    
    for i in range(N_SOURCES):
        if i < len(reconstructed_sources) and len(reconstructed_sources[i]) > 0:
            metrics = evaluator.calculate_metrics(
                original_sources[i],
                reconstructed_sources[i]
            )
            print(f"   Quelle {i}:")
            print(f"      SNR: {metrics['snr']:.2f} dB")
            print(f"      Korrelation: {metrics['correlation']:.3f}")
    
    # 4. Visualisierung
    print("\n4. Erstelle Visualisierung...")
    evaluator.plot_results(
        interleaved,
        original_sources,
        reconstructed_sources,
        deinterleaver.segments
    )
    
    # 5. Audio-Dateien speichern
    print("\n5. Speichere Audio-Dateien...")
    wav.write('interleaved.wav', SAMPLE_RATE, 
              (interleaved * 32767).astype(np.int16))
    
    for i, recon in enumerate(reconstructed_sources):
        if len(recon) > 0:
            wav.write(f'reconstructed_source_{i}.wav', SAMPLE_RATE,
                     (recon * 32767).astype(np.int16))
    
    print("\n" + "=" * 60)
    print("Fertig! Audio-Dateien und Plots wurden gespeichert.")
    print("=" * 60)


if __name__ == "__main__":
    main()