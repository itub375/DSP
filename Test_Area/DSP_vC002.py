"""
Streaming Audio Deinterleaver v6.0
====================================
Erfüllt ALLE Anforderungen:
- Echtzeit-Streaming (Chunk-basiert)
- Kontinuierlicher Betrieb mit Start/Stop/Resume
- Inkrementelles Clustering
- State-Management
- Robustheit gegen Pausen und Stille
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pydub import AudioSegment
import time
from pathlib import Path
from collections import deque
import json


@dataclass
class StreamConfig:
    """Konfiguration für Streaming-Betrieb"""
    # Chunk-Verarbeitung
    CHUNK_MS: float = 500.0              # Chunk-Größe in ms
    OVERLAP_MS: float = 100.0             # Overlap zwischen Chunks
    
    # Feature-Extraktion
    WINDOW_MS: float = 2.0
    HOP_MS: float = 0.5
    MIN_AMPLITUDE: float = 0.01
    
    # Change Detection
    CHANGE_THRESHOLD_PERCENTILE: float = 85.0
    MIN_SEGMENT_MS: float = 9.8
    
    # Clustering
    MAX_CLUSTERS: int = 3
    CLUSTER_SIMILARITY_THRESHOLD: float = 0.3
    
    # Anti-Click
    FADE_MS: float = 3.0
    ALIGN_TO_ZERO: bool = True
    
    # Output
    EXPORT_FORMAT: str = "mp3"
    OUTPUT_DIR: str = "streaming_output"
    
    # Weights
    WEIGHT_CENTROID: float = 2.0
    WEIGHT_RMS: float = 2.0
    WEIGHT_ROLLOFF: float = 1.5
    WEIGHT_ZCR: float = 1.5
    WEIGHT_FLUX: float = 20.0
    WEIGHT_BANDWIDTH: float = 1.0
    
    VERBOSE: bool = True


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """Extrahiert Features aus Audio-Chunks"""
    
    def __init__(self, sr: int, config: StreamConfig):
        self.sr = sr
        self.cfg = config
        self.window_samples = int(config.WINDOW_MS * sr / 1000)
        if self.window_samples % 2 != 0:
            self.window_samples += 1
        self.hop_samples = int(config.HOP_MS * sr / 1000)
        self.window = np.hanning(self.window_samples)
        self.prev_spec = None
    
    def extract_frame_features(self, frame: np.ndarray) -> Dict:
        """Extrahiert Features aus einem Frame"""
        rms = np.sqrt(np.mean(frame**2))
        
        # Frame zu leise?
        if rms < self.cfg.MIN_AMPLITUDE:
            return {
                'centroid': 0.0,
                'rms': rms,
                'rolloff': 0.0,
                'zcr': 0.0,
                'flux': 0.0,
                'bandwidth': 0.0,
                'valid': False
            }
        
        # ZCR
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        
        # Spektrale Features
        windowed = frame * self.window[:len(frame)]
        spec = np.abs(np.fft.rfft(windowed, n=self.window_samples))
        spec_power = spec ** 2
        freqs = np.fft.rfftfreq(self.window_samples, 1/self.sr)
        
        if spec_power.sum() > 1e-10:
            centroid = np.sum(freqs * spec_power) / spec_power.sum()
            
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            bandwidth = np.sqrt(
                np.sum(((freqs - centroid) ** 2) * spec_power) / spec_power.sum()
            )
            
            flux = np.sum((spec - self.prev_spec) ** 2) if self.prev_spec is not None else 0.0
            self.prev_spec = spec.copy()
        else:
            centroid = rolloff = bandwidth = flux = 0.0
        
        return {
            'centroid': centroid,
            'rms': rms,
            'rolloff': rolloff,
            'zcr': zcr,
            'flux': flux,
            'bandwidth': bandwidth,
            'valid': True
        }
    
    def extract_chunk_features(self, chunk: np.ndarray, start_time: float) -> List[Dict]:
        """Extrahiert Features aus einem Chunk"""
        features = []
        n_frames = 1 + max(0, (len(chunk) - self.window_samples) // self.hop_samples)
        
        for i in range(n_frames):
            start = i * self.hop_samples
            end = start + self.window_samples
            if end > len(chunk):
                break
            
            frame = chunk[start:end]
            frame_time = start_time + (start / self.sr)
            
            feat = self.extract_frame_features(frame)
            feat['time'] = frame_time
            feat['sample_idx'] = int(frame_time * self.sr)
            features.append(feat)
        
        return features


# ============================================================================
# CHANGE DETECTOR
# ============================================================================

class ChangeDetector:
    """Erkennt Änderungspunkte in Feature-Stream"""
    
    def __init__(self, config: StreamConfig):
        self.cfg = config
        self.feature_history = deque(maxlen=100)  # Letzte 100 Frames
        self.change_history = deque(maxlen=100)
        
    def normalize_features(self, features: List[float]) -> np.ndarray:
        """Normalisiert Features basierend auf History"""
        if len(self.feature_history) < 10:
            return np.zeros(6)
        
        hist = np.array(self.feature_history)
        feat = np.array(features)
        
        normalized = np.zeros(6)
        for i in range(6):
            col = hist[:, i]
            valid_values = col[col > 0]
            if len(valid_values) > 0:
                min_val, max_val = np.percentile(valid_values, [5, 95])
                if max_val - min_val > 0:
                    normalized[i] = np.clip((feat[i] - min_val) / (max_val - min_val), 0, 1)
        
        return normalized
    
    def compute_change_score(self, feat_dict: Dict) -> float:
        """Berechnet Change Score für ein Feature-Dict"""
        if not feat_dict['valid']:
            return 0.0
        
        features = [
            feat_dict['centroid'],
            feat_dict['rms'],
            feat_dict['rolloff'],
            feat_dict['zcr'],
            feat_dict['flux'],
            feat_dict['bandwidth']
        ]
        
        # Speichere in History
        self.feature_history.append(features)
        
        if len(self.feature_history) < 2:
            return 0.0
        
        # Normalisiere aktuelle und vorherige
        curr_norm = self.normalize_features(features)
        prev_norm = self.normalize_features(list(self.feature_history)[-2])
        
        # Gewichtete Änderung
        weights = np.array([
            self.cfg.WEIGHT_CENTROID,
            self.cfg.WEIGHT_RMS,
            self.cfg.WEIGHT_ROLLOFF,
            self.cfg.WEIGHT_ZCR,
            self.cfg.WEIGHT_FLUX,
            self.cfg.WEIGHT_BANDWIDTH
        ])
        
        change = np.abs(curr_norm - prev_norm) * weights
        score = change.sum() / weights.sum()
        
        self.change_history.append(score)
        
        # Smoothing
        if len(self.change_history) >= 5:
            return np.mean(list(self.change_history)[-5:])
        
        return score
    
    def get_threshold(self) -> float:
        """Berechnet adaptiven Threshold"""
        if len(self.change_history) < 10:
            return 0.3
        
        return np.percentile(list(self.change_history), self.cfg.CHANGE_THRESHOLD_PERCENTILE)
    
    def is_boundary(self, change_score: float) -> bool:
        """Prüft ob Change Score eine Grenze darstellt"""
        threshold = self.get_threshold()
        
        # Peak Detection
        if len(self.change_history) < 3:
            return False
        
        recent = list(self.change_history)[-3:]
        is_peak = change_score > threshold and change_score >= recent[-2]
        
        return is_peak


# ============================================================================
# ONLINE CLUSTERER
# ============================================================================

class OnlineClusterer:
    """Inkrementelles Clustering für Segment-Klassifikation"""
    
    def __init__(self, config: StreamConfig):
        self.cfg = config
        self.clusters = []  # Liste von Cluster-Centern (Feature-Vektoren)
        self.cluster_counts = []  # Anzahl Samples pro Cluster
        self.min_samples_for_new_cluster = 3
    
    def extract_features(self, segment: np.ndarray, sr: int) -> np.ndarray:
        """Extrahiert 6D Feature-Vektor aus Segment"""
        if len(segment) < 10:
            return np.zeros(6)
        
        rms = np.sqrt(np.mean(segment**2))
        zcr = np.sum(np.abs(np.diff(np.sign(segment)))) / (2 * len(segment))
        
        window = np.hanning(len(segment))
        spec = np.abs(np.fft.rfft(segment * window))
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        spec_power = spec ** 2
        
        if spec_power.sum() > 1e-10:
            centroid = np.sum(freqs * spec_power) / spec_power.sum()
            cumsum = np.cumsum(spec_power)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec_power) / spec_power.sum())
            flux = np.std(spec_power)
        else:
            centroid = rolloff = bandwidth = flux = 0
        
        return np.array([centroid, rms, rolloff, zcr, flux, bandwidth])
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalisiert Features"""
        if len(self.clusters) == 0:
            return features
        
        all_features = np.vstack([features] + self.clusters)
        normalized = np.zeros_like(features)
        
        for i in range(features.shape[0]):
            col_min, col_max = all_features[:, i].min(), all_features[:, i].max()
            if col_max - col_min > 0:
                normalized[i] = (features[i] - col_min) / (col_max - col_min)
        
        return normalized
    
    def find_cluster(self, features: np.ndarray) -> int:
        """Findet nächsten Cluster oder erstellt neuen"""
        if len(self.clusters) == 0:
            # Erster Cluster
            self.clusters.append(features.copy())
            self.cluster_counts.append(1)
            return 0
        
        # Normalisiere
        feat_norm = self.normalize_features(features)
        
        # Finde nächsten Cluster
        distances = []
        for cluster in self.clusters:
            cluster_norm = self.normalize_features(cluster)
            dist = np.linalg.norm(feat_norm - cluster_norm)
            distances.append(dist)
        
        min_dist = min(distances)
        nearest_idx = distances.index(min_dist)
        
        # Wenn zu weit weg und noch Platz: neuer Cluster
        if min_dist > self.cfg.CLUSTER_SIMILARITY_THRESHOLD and len(self.clusters) < self.cfg.MAX_CLUSTERS:
            self.clusters.append(features.copy())
            self.cluster_counts.append(1)
            return len(self.clusters) - 1
        
        # Sonst: nächsten zuordnen und Center updaten
        cluster_idx = nearest_idx
        count = self.cluster_counts[cluster_idx]
        self.clusters[cluster_idx] = (self.clusters[cluster_idx] * count + features) / (count + 1)
        self.cluster_counts[cluster_idx] += 1
        
        return cluster_idx
    
    def get_num_clusters(self) -> int:
        return len(self.clusters)


# ============================================================================
# SEGMENT BUFFER
# ============================================================================

class SegmentBuffer:
    """Verwaltet erkannte Segmente mit Overlap-Handling"""
    
    def __init__(self, sr: int, config: StreamConfig):
        self.sr = sr
        self.cfg = config
        self.segments = []  # Liste von (start_time, end_time, audio_data, label)
        self.pending_segment = None  # Aktuell offenes Segment
        self.last_boundary_time = 0.0
    
    def start_segment(self, time: float, audio_start_idx: int):
        """Startet neues Segment"""
        self.pending_segment = {
            'start_time': time,
            'start_idx': audio_start_idx,
            'audio_chunks': []
        }
    
    def add_audio_to_pending(self, audio_chunk: np.ndarray):
        """Fügt Audio zu aktuellem Segment hinzu"""
        if self.pending_segment is not None:
            self.pending_segment['audio_chunks'].append(audio_chunk.copy())
    
    def close_segment(self, time: float, label: int) -> bool:
        if self.pending_segment is None:
            return False

        duration = time - self.pending_segment['start_time']
        if duration < self.cfg.MIN_SEGMENT_MS / 1000.0:
            self.pending_segment = None
            return False

        audio = np.concatenate(self.pending_segment['audio_chunks']) if self.pending_segment['audio_chunks'] else np.array([])
        if len(audio) > 0:
            self.segments.append({
                'start_time': self.pending_segment['start_time'],
                'end_time': time,
                'audio': audio,
                'label': label
            })

        self.pending_segment = None
        self.last_boundary_time = time
        return True

        
        # Kombiniere Audio
        audio = np.concatenate(self.pending_segment['audio_chunks']) if self.pending_segment['audio_chunks'] else np.array([])
        
        if len(audio) > 0:
            self.segments.append({
                'start_time': self.pending_segment['start_time'],
                'end_time': time,
                'audio': audio,
                'label': label
            })
        
        self.pending_segment = None
        self.last_boundary_time = time


# ============================================================================
# STREAMING DEINTERLEAVER (Hauptklasse)
# ============================================================================

class StreamingDeinterleaver:
    """
    Hauptklasse für Streaming Audio Deinterleaving
    """
    
    def __init__(self, sr: int, config: StreamConfig):
        self.sr = sr
        self.cfg = config
        
        # Komponenten
        self.feature_extractor = FeatureExtractor(sr, config)
        self.change_detector = ChangeDetector(config)
        self.clusterer = OnlineClusterer(config)
        self.segment_buffer = SegmentBuffer(sr, config)
        
        # State
        self.state = "IDLE"  # IDLE, RUNNING, PAUSED
        self.total_time_processed = 0.0
        self.chunk_counter = 0
        
        # Output buffers (ein Buffer pro erkanntem Signal)
        self.output_buffers = {}  # {label: np.ndarray}
        
        # Overlap buffer
        self.overlap_buffer = np.array([])
        
        # Statistiken
        self.stats = {
            'chunks_processed': 0,
            'segments_found': 0,
            'boundaries_detected': 0
        }
        
        # Für Visualisierung
        self.feature_log = []
        self.change_log = []
        self.boundary_log = []
    
    def start(self):
        """Startet Streaming"""
        self.state = "RUNNING"
        if self.cfg.VERBOSE:
            print("🟢 Streaming started")
    
    def pause(self):
        """Pausiert Streaming"""
        if self.segment_buffer.pending_segment:
            # Schließe offenes Segment mit letztem bekannten Label
            last_label = self.clusterer.get_num_clusters() - 1 if self.clusterer.get_num_clusters() > 0 else 0
            self.segment_buffer.close_segment(self.total_time_processed, last_label)
        
        self.state = "PAUSED"
        if self.cfg.VERBOSE:
            print("⏸️  Streaming paused")
    
    def resume(self):
        """Setzt Streaming fort"""
        self.state = "RUNNING"
        if self.cfg.VERBOSE:
            print("▶️  Streaming resumed")
    
    def stop(self):
        """Stoppt Streaming und exportiert"""
        self.pause()
        self.state = "IDLE"
        if self.cfg.VERBOSE:
            print("⏹️  Streaming stopped")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Verarbeitet einen Audio-Chunk
        
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if self.state != "RUNNING":
            return False
        
        try:
            # Chunk mit Overlap kombinieren
            if len(self.overlap_buffer) > 0:
                full_chunk = np.concatenate([self.overlap_buffer, audio_chunk])
            else:
                full_chunk = audio_chunk
            
            chunk_start_time = self.total_time_processed - len(self.overlap_buffer) / self.sr
            
            # Features extrahieren
            features = self.feature_extractor.extract_chunk_features(full_chunk, chunk_start_time)
            
            # Change Detection und Segmentierung
            for feat in features:
                self.feature_log.append(feat)
                
                change_score = self.change_detector.compute_change_score(feat)
                self.change_log.append({'time': feat['time'], 'score': change_score})
                
                is_boundary = self.change_detector.is_boundary(change_score)
                
                if is_boundary:
                    self.stats['boundaries_detected'] += 1
                    self.boundary_log.append(feat['time'])
                    
                    # Schließe vorheriges Segment wenn vorhanden
                    if self.segment_buffer.pending_segment:
                        # Extrahiere Features des abgeschlossenen Segments
                        seg_audio = np.concatenate(self.segment_buffer.pending_segment['audio_chunks'])
                        seg_features = self.clusterer.extract_features(seg_audio, self.sr)
                        
                        # Klassifiziere
                        label = self.clusterer.find_cluster(seg_features)
                        stored = self.segment_buffer.close_segment(feat['time'], label)
                        if stored:
                            self.stats['segments_found'] += 1

                    
                    # Starte neues Segment
                    sample_idx = int(feat['time'] * self.sr)
                    self.segment_buffer.start_segment(feat['time'], sample_idx)
                
                # Füge Audio zu aktuellem Segment hinzu (kleine Chunks)
                if self.segment_buffer.pending_segment and feat['valid']:
                    frame_idx = int((feat['time'] - chunk_start_time) * self.sr)
                    frame_audio = full_chunk[frame_idx:frame_idx + self.feature_extractor.hop_samples]
                    self.segment_buffer.add_audio_to_pending(frame_audio)
            
            # Update overlap buffer
            overlap_samples = int(self.cfg.OVERLAP_MS * self.sr / 1000)
            self.overlap_buffer = audio_chunk[-overlap_samples:].copy()
            
            # Update State
            self.total_time_processed += len(audio_chunk) / self.sr
            self.chunk_counter += 1
            self.stats['chunks_processed'] += 1
            
            return True
            
        except Exception as e:
            if self.cfg.VERBOSE:
                print(f"❌ Error processing chunk: {e}")
            return False
    
    def reconstruct_signals(self) -> Dict[int, np.ndarray]:
        """Rekonstruiert separate Signale aus Segmenten"""
        signals: Dict[int, np.ndarray] = {}

        total_samples = int(self.total_time_processed * self.sr)

        for segment in self.segment_buffer.segments:
            label = segment['label']
            audio = segment['audio']

            if label not in signals:
                signals[label] = np.zeros(total_samples, dtype=np.float32)

            # Basis-Startindex aus Zeit
            start_idx = int(segment['start_time'] * self.sr)

            # Anti-Click: Zero-Crossing Alignment (kann Länge ändern!)
            if self.cfg.ALIGN_TO_ZERO:
                audio, start_trim = self._apply_zero_crossing_alignment(audio)
                start_idx += start_trim  # Zeitversatz durch Start-Trim

                # Fade (ändert Länge nicht)
                audio = self._apply_fade(audio)

                if audio is None or len(audio) == 0:
                    continue

                end_idx = start_idx + len(audio)

                # Clamps + ggf. Truncation, damit Assignment immer passt
                if start_idx < 0:
                    audio = audio[-start_idx:]
                    start_idx = 0
                    end_idx = start_idx + len(audio)

                if start_idx >= len(signals[label]):
                    continue

                if end_idx > len(signals[label]):
                    audio = audio[:len(signals[label]) - start_idx]
                    end_idx = start_idx + len(audio)

                if len(audio) > 0:
                    signals[label][start_idx:end_idx] = audio

            return signals
    
    def _apply_zero_crossing_alignment(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        """Schneidet Audio bei Nulldurchgängen und gibt (audio_aligned, start_trim_samples) zurück."""
        if audio is None or len(audio) < 4:
            return audio, 0

        search_range = min(100, len(audio) // 10)
        if search_range < 2:
            return audio, 0

        start_trim = 0
        a = audio

        # --- Start trim ---
        start_segment = a[:search_range]
        zc = np.where(np.diff(np.sign(start_segment)) != 0)[0]
        if len(zc) > 0:
            start_trim = int(zc[0])
            a = a[start_trim:]

        # --- End trim ---
        if len(a) > search_range:
            end_segment = a[-search_range:]
            zc_end = np.where(np.diff(np.sign(end_segment)) != 0)[0]
            if len(zc_end) > 0:
                cut_point = len(a) - search_range + int(zc_end[-1])
                a = a[:max(0, cut_point)]

        return a, start_trim
    
    def _apply_fade(self, audio: np.ndarray) -> np.ndarray:
        """Wendet Fade-In/Out an"""
        fade_samples = int(self.cfg.FADE_MS * self.sr / 1000)
        fade_len = min(fade_samples, len(audio) // 4)
        
        if fade_len > 0:
            fade_in = (1 - np.cos(np.linspace(0, np.pi, fade_len))) / 2
            fade_out = (1 + np.cos(np.linspace(0, np.pi, fade_len))) / 2
            
            audio = audio.copy()
            audio[:fade_len] *= fade_in
            audio[-fade_len:] *= fade_out
        
        return audio
    
    def export_results(self, base_name: str = "output"):
        """Exportiert rekonstruierte Signale"""
        signals = self.reconstruct_signals()
        
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        exported_files = {}
        
        for label, audio in signals.items():
            name = chr(ord('A') + label)
            filename = f"{base_name}_signal_{name}.{self.cfg.EXPORT_FORMAT}"
            filepath = os.path.join(self.cfg.OUTPUT_DIR, filename)
            
            # Speichern
            audio_clipped = np.clip(audio, -1.0, 1.0)
            int16 = (audio_clipped * 32767.0).astype(np.int16)
            seg = AudioSegment(
                data=int16.tobytes(),
                sample_width=2,
                frame_rate=self.sr,
                channels=1
            )
            seg.export(filepath, format=self.cfg.EXPORT_FORMAT, 
                      bitrate="192k" if self.cfg.EXPORT_FORMAT == "mp3" else None)
            
            exported_files[name] = filepath
            
            if self.cfg.VERBOSE:
                duration = len(audio) / self.sr
                print(f"  → Signal {name}: {filepath} ({duration:.1f}s)")
        
        return exported_files
    
    def get_statistics(self) -> Dict:
        """Gibt Verarbeitungsstatistiken zurück"""
        return {
            **self.stats,
            'total_time': self.total_time_processed,
            'num_signals': self.clusterer.get_num_clusters(),
            'state': self.state
        }
    
    def save_state(self, filename: str):
        """Speichert internen Zustand"""
        state = {
            'total_time': self.total_time_processed,
            'chunk_counter': self.chunk_counter,
            'stats': self.stats,
            'clusters': [c.tolist() for c in self.clusterer.clusters],
            'cluster_counts': self.clusterer.cluster_counts,
            'state': self.state
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f)
        
        if self.cfg.VERBOSE:
            print(f"💾 State saved to {filename}")
    
    def load_state(self, filename: str):
        """Lädt internen Zustand"""
        with open(filename, 'r') as f:
            state = json.load(f)
        
        self.total_time_processed = state['total_time']
        self.chunk_counter = state['chunk_counter']
        self.stats = state['stats']
        self.clusterer.clusters = [np.array(c) for c in state['clusters']]
        self.clusterer.cluster_counts = state['cluster_counts']
        self.state = state['state']
        
        if self.cfg.VERBOSE:
            print(f"📂 State loaded from {filename}")


# ============================================================================
# SIMULATION (für Tests ohne echtes Echtzeit-Audio)
# ============================================================================

def simulate_streaming_from_file(input_file: str, config: StreamConfig):
    """
    Simuliert Streaming-Betrieb aus einer Datei
    """
    print("="*70)
    print("🎵 STREAMING AUDIO DEINTERLEAVER v6.0")
    print("="*70)
    
    # Lade komplettes Audio
    print(f"\n[1] Loading audio file: {input_file}")
    audio = AudioSegment.from_file(input_file).set_channels(1)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    if audio.sample_width == 2:
        y = samples / 32768.0
    else:
        y = samples / max(np.abs(samples).max(), 1)
    
    duration = len(y) / sr
    print(f"  ✓ Sample rate: {sr} Hz")
    print(f"  ✓ Duration: {duration:.2f}s")
    print(f"  ✓ Total samples: {len(y)}")
    
    # Initialisiere Deinterleaver
    print(f"\n[2] Initializing streaming deinterleaver...")
    deinterleaver = StreamingDeinterleaver(sr, config)
    deinterleaver.start()
    
    # Chunk-Parameter
    chunk_samples = int(config.CHUNK_MS * sr / 1000)
    print(f"  ✓ Chunk size: {config.CHUNK_MS}ms ({chunk_samples} samples)")
    print(f"  ✓ Overlap: {config.OVERLAP_MS}ms")
    
    # Streaming-Simulation
    print(f"\n[3] Processing audio stream...")
    print("  " + "="*60)
    
    n_chunks = len(y) // chunk_samples
    t_start = time.time()
    
    for i in range(n_chunks):
        start_idx = i * chunk_samples
        end_idx = start_idx + chunk_samples
        chunk = y[start_idx:end_idx]
        
        # Simuliere Pause bei 30% und 70%
        progress = (i + 1) / n_chunks
        if 0.29 < progress < 0.31 and deinterleaver.state == "RUNNING":
            print(f"\n  ⏸️  Simulating PAUSE at {progress*100:.0f}%...")
            deinterleaver.pause()
            time.sleep(0.1)  # Kurze Pause
            print(f"  ▶️  Resuming...")
            deinterleaver.resume()
        
        # Verarbeite Chunk
        success = deinterleaver.process_chunk(chunk)
        
        # Progress
        if (i + 1) % 20 == 0 or i == n_chunks - 1:
            stats = deinterleaver.get_statistics()
            print(f"  [{i+1:4d}/{n_chunks}] "
                  f"Time: {stats['total_time']:.1f}s | "
                  f"Boundaries: {stats['boundaries_detected']:3d} | "
                  f"Segments: {stats['segments_found']:3d} | "
                  f"Signals: {stats['num_signals']}")
    
    t_elapsed = time.time() - t_start
    print(f"  " + "="*60)
    print(f"  ✓ Processing completed in {t_elapsed:.2f}s")
    print(f"  ✓ Processing speed: {duration/t_elapsed:.1f}x realtime")
    
    # Stop
    deinterleaver.stop()
    
    # Statistiken
    print(f"\n[4] Final statistics:")
    stats = deinterleaver.get_statistics()
    for key, val in stats.items():
        print(f"  • {key}: {val}")
    
    # Export
    print(f"\n[5] Exporting reconstructed signals...")
    base_name = Path(input_file).stem
    exported = deinterleaver.export_results(base_name)
    
    print(f"\n[6] Visualization...")
    visualize_results(deinterleaver, duration, base_name)
    
    print("\n" + "="*70)
    print("✅ STREAMING DEINTERLEAVING COMPLETE!")
    print("="*70)
    
    return deinterleaver


def visualize_results(deinterleaver: StreamingDeinterleaver, max_time: float, 
                     base_name: str):
    """Erstellt Visualisierung der Ergebnisse"""
    
    if len(deinterleaver.feature_log) == 0:
        print("  ⚠️  No data to visualize")
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Extrahiere Daten
    times = [f['time'] for f in deinterleaver.feature_log]
    centroids = [f['centroid'] for f in deinterleaver.feature_log]
    rms_values = [f['rms'] for f in deinterleaver.feature_log]
    valid = [f['valid'] for f in deinterleaver.feature_log]
    
    change_times = [c['time'] for c in deinterleaver.change_log]
    change_scores = [c['score'] for c in deinterleaver.change_log]
    
    boundaries = deinterleaver.boundary_log
    
    # Begrenze auf max_time
    mask = np.array(times) <= max_time
    times = np.array(times)[mask]
    centroids = np.array(centroids)[mask]
    rms_values = np.array(rms_values)[mask]
    valid = np.array(valid)[mask]
    
    # 1. Spectral Centroid
    ax = axes[0]
    ax.plot(times, centroids, 'b-', linewidth=1, alpha=0.7)
    for t in boundaries:
        if t <= max_time:
            ax.axvline(t, color='r', linestyle='--', alpha=0.7)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectral Centroid + Detected Boundaries')
    ax.grid(True, alpha=0.3)
    
    # 2. RMS Energy
    ax = axes[1]
    ax.plot(times, rms_values, 'g-', linewidth=1, alpha=0.7)
    ax.axhline(deinterleaver.cfg.MIN_AMPLITUDE, color='orange', 
               linestyle=':', label='Min Amplitude Threshold')
    for t in boundaries:
        if t <= max_time:
            ax.axvline(t, color='r', linestyle='--', alpha=0.7)
    ax.set_ylabel('RMS')
    ax.set_title('RMS Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Change Score
    ax = axes[2]
    change_mask = np.array(change_times) <= max_time
    ax.plot(np.array(change_times)[change_mask], 
            np.array(change_scores)[change_mask], 
            'purple', linewidth=1, label='Change Score')
    threshold = deinterleaver.change_detector.get_threshold()
    ax.axhline(threshold, color='orange', linestyle=':', label='Threshold')
    for t in boundaries:
        if t <= max_time:
            ax.axvline(t, color='r', linestyle='--', alpha=0.7)
    ax.set_ylabel('Change Score')
    ax.set_title('Change Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Segments colored by cluster
    ax = axes[3]
    colors = plt.cm.Set3(np.linspace(0, 1, deinterleaver.clusterer.get_num_clusters()))
    
    for segment in deinterleaver.segment_buffer.segments:
        if segment['start_time'] <= max_time:
            ax.axvspan(segment['start_time'], 
                      min(segment['end_time'], max_time),
                      alpha=0.5, color=colors[segment['label']],
                      label=f"Signal {chr(ord('A') + segment['label'])}" 
                      if segment['label'] not in ax.get_legend_handles_labels()[1] else '')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Segment')
    ax.set_title('Reconstructed Signals')
    ax.set_ylim(-0.5, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichern
    plot_path = os.path.join(deinterleaver.cfg.OUTPUT_DIR, f"{base_name}_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved: {plot_path}")
    
    # Optional anzeigen
    # plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    t0 = time.perf_counter()
    # Konfiguration
    config = StreamConfig(
        CHUNK_MS=500.0,
        OVERLAP_MS=100.0,
        MIN_AMPLITUDE=0.01,
        CHANGE_THRESHOLD_PERCENTILE=85.0,
        MIN_SEGMENT_MS=9.8,
        MAX_CLUSTERS=3,
        FADE_MS=3.0,
        ALIGN_TO_ZERO=True,
        OUTPUT_DIR="streaming_output",
        VERBOSE=True
    )
    
    # Teste mit verschiedenen Dateien
    test_files = [
        r"C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Inputsignals/rand/interleaved_1k_GOD_30sec_rand.mp3",
        # Weitere Dateien hier...
    ]
    
    for audio_file in test_files:
        if os.path.exists(audio_file):
            print(f"\n\n{'='*70}")
            print(f"Processing: {Path(audio_file).name}")
            print(f"{'='*70}\n")
            
            deinterleaver = simulate_streaming_from_file(audio_file, config)
            
            # Optional: State speichern
            state_file = os.path.join(config.OUTPUT_DIR, f"{Path(audio_file).stem}_state.json")
            deinterleaver.save_state(state_file)
        else:
            print(f"⚠️  File not found: {audio_file}")
    
    t1 = time.perf_counter()
    
    t = t1-t0
    print(f"\nZeit: {t:.2f}s")
    print("\n🎉 All processing complete!")
