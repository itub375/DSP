import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_csv_files(directory="."):
    """
    Analysiert alle CSV-Dateien und bestimmt optimale Grenzwerte
    für die Unterscheidung zwischen Musik/Sprache und Sinussignalen.
    """
    
    # Kategorien definieren
    music_speech = [
        'PIMP_CSV.csv', 'Podcast_shorted_CSV.csv', 'RAP_God_CSV.csv',
        'TagesSchau 1_CSV.csv', 'TagesSchau 2_CSV.csv', 'violin_CSV.csv',
        'Without_me_CSV.csv', 'BigDawgs_CSV.csv', 'drum_CSV.csv',
        'jingle_CSV.csv', 'NotLikeUs_CSV.csv'
    ]
    
    sine_signals = [
        'sine_100Hz_CSV.csv', 'sine_700Hz_CSV.csv', 'sine_1kHz_CSV.csv',
        'sine_8kHz_CSV.csv', 'sine_20kHz_CSV.csv'
    ]
    
    noise_signals = [
        'whitenoise_CSV.csv', 'brownnoise_CSV.csv'
    ]
    
    # Datenstrukturen für Statistiken
    stats_music = {
        'centroid': [], 'power': [], 'energy': [],
        'peak_freq': [], 'flux': [], 'bandwidth': [],
        'flatness': [], 'rolloff': []
    }
    
    stats_sine = {
        'centroid': [], 'power': [], 'energy': [],
        'peak_freq': [], 'flux': [], 'bandwidth': [],
        'flatness': [], 'rolloff': []
    }
    
    stats_noise = {
        'centroid': [], 'power': [], 'energy': [],
        'peak_freq': [], 'flux': [], 'bandwidth': [],
        'flatness': [], 'rolloff': []
    }
    
    print("=== ANALYSE DER CSV-DATEIEN ===\n")
    
    # Musik/Sprache analysieren
    print("--- MUSIK/SPRACHE ---")
    for filename in music_speech:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                # Nur numerische Zeilen (vor Statistiken)
                df_numeric = df[df['Zeit (s)'].apply(lambda x: str(x).replace('.', '').isdigit())]
                
                if len(df_numeric) > 0:
                    stats_music['centroid'].append(df_numeric['Spektral Centroid (Hz)'].astype(float).mean())
                    stats_music['power'].append(df_numeric['Power (dB)'].astype(float).mean())
                    stats_music['energy'].append(df_numeric['Energy'].astype(float).mean())
                    stats_music['peak_freq'].append(df_numeric['Peak Frequenz (Hz)'].astype(float).mean())
                    stats_music['flux'].append(df_numeric['Spectral Flux'].astype(float).mean())
                    stats_music['bandwidth'].append(df_numeric['Bandbreite (Hz)'].astype(float).mean())
                    stats_music['flatness'].append(df_numeric['Spectral Flatness'].astype(float).mean())
                    stats_music['rolloff'].append(df_numeric['Spectral Rolloff (Hz)'].astype(float).mean())
                    
                    print(f"✓ {filename}")
            except Exception as e:
                print(f"✗ {filename}: {e}")
    
    # Sinussignale analysieren
    print("\n--- SINUSSIGNALE ---")
    for filename in sine_signals:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df_numeric = df[df['Zeit (s)'].apply(lambda x: str(x).replace('.', '').isdigit())]
                
                if len(df_numeric) > 0:
                    stats_sine['centroid'].append(df_numeric['Spektral Centroid (Hz)'].astype(float).mean())
                    stats_sine['power'].append(df_numeric['Power (dB)'].astype(float).mean())
                    stats_sine['energy'].append(df_numeric['Energy'].astype(float).mean())
                    stats_sine['peak_freq'].append(df_numeric['Peak Frequenz (Hz)'].astype(float).mean())
                    stats_sine['flux'].append(df_numeric['Spectral Flux'].astype(float).mean())
                    stats_sine['bandwidth'].append(df_numeric['Bandbreite (Hz)'].astype(float).mean())
                    stats_sine['flatness'].append(df_numeric['Spectral Flatness'].astype(float).mean())
                    stats_sine['rolloff'].append(df_numeric['Spectral Rolloff (Hz)'].astype(float).mean())
                    
                    print(f"✓ {filename}")
            except Exception as e:
                print(f"✗ {filename}: {e}")
    
    # Rauschsignale analysieren
    print("\n--- RAUSCHSIGNALE ---")
    for filename in noise_signals:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df_numeric = df[df['Zeit (s)'].apply(lambda x: str(x).replace('.', '').isdigit())]
                
                if len(df_numeric) > 0:
                    stats_noise['centroid'].append(df_numeric['Spektral Centroid (Hz)'].astype(float).mean())
                    stats_noise['power'].append(df_numeric['Power (dB)'].astype(float).mean())
                    stats_noise['energy'].append(df_numeric['Energy'].astype(float).mean())
                    stats_noise['peak_freq'].append(df_numeric['Peak Frequenz (Hz)'].astype(float).mean())
                    stats_noise['flux'].append(df_numeric['Spectral Flux'].astype(float).mean())
                    stats_noise['bandwidth'].append(df_numeric['Bandbreite (Hz)'].astype(float).mean())
                    stats_noise['flatness'].append(df_numeric['Spectral Flatness'].astype(float).mean())
                    stats_noise['rolloff'].append(df_numeric['Spectral Rolloff (Hz)'].astype(float).mean())
                    
                    print(f"✓ {filename}")
            except Exception as e:
                print(f"✗ {filename}: {e}")
    
    # Statistiken berechnen
    print("\n\n=== FEATURE-STATISTIKEN ===\n")
    
    features = ['centroid', 'power', 'energy', 'peak_freq', 'flux', 'bandwidth', 'flatness', 'rolloff']
    feature_names = ['Spektral Centroid (Hz)', 'Power (dB)', 'Energy', 
                     'Peak Frequenz (Hz)', 'Spectral Flux', 'Bandbreite (Hz)',
                     'Spectral Flatness', 'Spectral Rolloff (Hz)']
    
    results = {}
    
    for feat, name in zip(features, feature_names):
        print(f"--- {name} ---")
        
        music_vals = np.array(stats_music[feat])
        sine_vals = np.array(stats_sine[feat])
        noise_vals = np.array(stats_noise[feat])
        
        if len(music_vals) > 0:
            print(f"Musik/Sprache: Mean={np.mean(music_vals):.2f}, Std={np.std(music_vals):.2f}")
            print(f"               Range=[{np.min(music_vals):.2f}, {np.max(music_vals):.2f}]")
        
        if len(sine_vals) > 0:
            print(f"Sinus:         Mean={np.mean(sine_vals):.2f}, Std={np.std(sine_vals):.2f}")
            print(f"               Range=[{np.min(sine_vals):.2f}, {np.max(sine_vals):.2f}]")
        
        if len(noise_vals) > 0:
            print(f"Rauschen:      Mean={np.mean(noise_vals):.2f}, Std={np.std(noise_vals):.2f}")
            print(f"               Range=[{np.min(noise_vals):.2f}, {np.max(noise_vals):.2f}]")
        
        # Trennbarkeit berechnen
        if len(music_vals) > 0 and len(sine_vals) > 0:
            separation = abs(np.mean(music_vals) - np.mean(sine_vals))
            combined_std = (np.std(music_vals) + np.std(sine_vals)) / 2
            separability = separation / (combined_std + 1e-6)
            
            print(f"Trennbarkeit: {separability:.2f} (höher = besser)")
            results[feat] = {
                'music_mean': np.mean(music_vals),
                'sine_mean': np.mean(sine_vals),
                'separability': separability,
                'suggested_threshold': separation / 2
            }
        
        print()
    
    # Grenzwert-Empfehlungen
    print("\n=== EMPFOHLENE GRENZWERTE FÜR C-CODE ===\n")
    print("// Für Unterscheidung Musik/Sprache vs. Sinus:")
    print()
    
    # Spektral Centroid
    if 'centroid' in results:
        thresh = results['centroid']['suggested_threshold']
        print(f"uint16_t threshCentroid = {int(thresh)};  // {thresh:.1f} Hz")
    
    # Power
    if 'power' in results:
        thresh = results['power']['suggested_threshold']
        print(f"uint16_t threshPower = {int(thresh)};     // {thresh:.1f} dB")
    
    # Energy
    if 'energy' in results:
        thresh = results['energy']['suggested_threshold']
        print(f"uint16_t threshEnergy = {int(thresh)};    // {thresh:.2e}")
    
    # Peak Frequency
    if 'peak_freq' in results:
        thresh = results['peak_freq']['suggested_threshold']
        print(f"uint16_t threshpeakF = {int(thresh)};     // {thresh:.1f} Hz")
    
    # Flux
    if 'flux' in results:
        thresh = results['flux']['suggested_threshold']
        print(f"uint16_t threshFlux = {int(thresh)};      // {thresh:.2e}")
    
    # Bandwidth
    if 'bandwidth' in results:
        thresh = results['bandwidth']['suggested_threshold']
        print(f"uint16_t threshBandwidth = {int(thresh)}; // {thresh:.1f} Hz")
    
    # Flatness
    if 'flatness' in results:
        thresh = results['flatness']['suggested_threshold']
        print(f"float32_t threshFlatness = {thresh:.4f}f;  // {thresh:.4f}")
    
    # Rolloff
    if 'rolloff' in results:
        thresh = results['rolloff']['suggested_threshold']
        print(f"uint16_t threshRolloff = {int(thresh)};   // {thresh:.1f} Hz")
    
    print("\n// Feature-Gewichtungen (basierend auf Trennbarkeit):")
    print()
    
    # Gewichtungen basierend auf Trennbarkeit
    if results:
        total_sep = sum([r['separability'] for r in results.values()])
        
        for feat in ['centroid', 'power', 'energy', 'peak_freq', 'flux', 'flatness', 'rolloff']:
            if feat in results:
                weight = int(10 * results[feat]['separability'] / total_sep)
                weight = max(1, weight)  # Mindestens 1
                
                feat_names = {
                    'centroid': 'weightCentroid',
                    'power': 'weightPower',
                    'energy': 'weightEnergy',
                    'peak_freq': 'weightpeakF',
                    'flux': 'weightFlux',
                    'flatness': 'weightFlatness',
                    'rolloff': 'weightRolloff'
                }
                
                print(f"uint16_t {feat_names[feat]} = {weight};")
    
    # Visualisierung erstellen
    create_visualization(stats_music, stats_sine, stats_noise, features, feature_names)
    
    return results


def create_visualization(stats_music, stats_sine, stats_noise, features, feature_names):
    """Erstellt Visualisierungen der Feature-Verteilungen"""
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.flatten()
    
    for idx, (feat, name) in enumerate(zip(features, feature_names)):
        ax = axes[idx]
        
        data_to_plot = []
        labels = []
        
        if len(stats_music[feat]) > 0:
            data_to_plot.append(stats_music[feat])
            labels.append('Musik/Sprache')
        
        if len(stats_sine[feat]) > 0:
            data_to_plot.append(stats_sine[feat])
            labels.append('Sinus')
        
        if len(stats_noise[feat]) > 0:
            data_to_plot.append(stats_noise[feat])
            labels.append('Rauschen')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Farben
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
            
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualisierung gespeichert als 'feature_analysis.png'")
    plt.close()


# Hauptprogramm
if __name__ == "__main__":
    # Verzeichnis mit CSV-Dateien (anpassen falls nötig)
    csv_directory = "C:/eigene Programme/VS_Code_Programme/HKA/DSP/CSV"
    
    print("Feature Threshold Analyzer")
    print("=" * 50)
    print()
    
    results = analyze_csv_files(csv_directory)
    
    print("\n" + "=" * 50)
    print("Analyse abgeschlossen!")