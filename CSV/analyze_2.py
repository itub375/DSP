import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_csv_files(directory="."):
    """
    Analysiert alle CSV-Dateien und bestimmt optimale Grenzwerte
    für die Unterscheidung zwischen Musik, Sprache und Sinussignalen.
    """
    
    # Kategorien definieren
    music_files = [
        'PIMP_CSV.csv', 'RAP_God_CSV.csv', 'violin_CSV.csv',
        'Without_me_CSV.csv', 'BigDawgs_CSV.csv', 'drum_CSV.csv',
        'jingle_CSV.csv', 'NotLikeUs_CSV.csv'
    ]
    
    speech_files = [
        'Podcast_shorted_CSV.csv', 'TagesSchau 1_CSV.csv', 'TagesSchau 2_CSV.csv'
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
    
    stats_speech = {
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
    
    # Musik analysieren
    print("--- MUSIK ---")
    for filename in music_files:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df_numeric = df[df['Zeit (s)'].apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit())]
                
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
    
    # Sprache analysieren
    print("\n--- SPRACHE ---")
    for filename in speech_files:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df_numeric = df[df['Zeit (s)'].apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit())]
                
                if len(df_numeric) > 0:
                    stats_speech['centroid'].append(df_numeric['Spektral Centroid (Hz)'].astype(float).mean())
                    stats_speech['power'].append(df_numeric['Power (dB)'].astype(float).mean())
                    stats_speech['energy'].append(df_numeric['Energy'].astype(float).mean())
                    stats_speech['peak_freq'].append(df_numeric['Peak Frequenz (Hz)'].astype(float).mean())
                    stats_speech['flux'].append(df_numeric['Spectral Flux'].astype(float).mean())
                    stats_speech['bandwidth'].append(df_numeric['Bandbreite (Hz)'].astype(float).mean())
                    stats_speech['flatness'].append(df_numeric['Spectral Flatness'].astype(float).mean())
                    stats_speech['rolloff'].append(df_numeric['Spectral Rolloff (Hz)'].astype(float).mean())
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
                df_numeric = df[df['Zeit (s)'].apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit())]
                
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
                df_numeric = df[df['Zeit (s)'].apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit())]
                
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
    
    results_music_vs_sine = {}
    results_speech_vs_sine = {}
    results_music_vs_speech = {}
    
    for feat, name in zip(features, feature_names):
        print(f"--- {name} ---")
        
        music_vals = np.array(stats_music[feat])
        speech_vals = np.array(stats_speech[feat])
        sine_vals = np.array(stats_sine[feat])
        noise_vals = np.array(stats_noise[feat])
        
        if len(music_vals) > 0:
            print(f"Musik:   Mean={np.mean(music_vals):.2f}, Std={np.std(music_vals):.2f}")
            print(f"         Range=[{np.min(music_vals):.2f}, {np.max(music_vals):.2f}]")
        
        if len(speech_vals) > 0:
            print(f"Sprache: Mean={np.mean(speech_vals):.2f}, Std={np.std(speech_vals):.2f}")
            print(f"         Range=[{np.min(speech_vals):.2f}, {np.max(speech_vals):.2f}]")
        
        if len(sine_vals) > 0:
            print(f"Sinus:   Mean={np.mean(sine_vals):.2f}, Std={np.std(sine_vals):.2f}")
            print(f"         Range=[{np.min(sine_vals):.2f}, {np.max(sine_vals):.2f}]")
        
        if len(noise_vals) > 0:
            print(f"Rauschen:Mean={np.mean(noise_vals):.2f}, Std={np.std(noise_vals):.2f}")
            print(f"         Range=[{np.min(noise_vals):.2f}, {np.max(noise_vals):.2f}]")
        
        # Trennbarkeit Musik vs Sinus
        if len(music_vals) > 0 and len(sine_vals) > 0:
            separation = abs(np.mean(music_vals) - np.mean(sine_vals))
            combined_std = (np.std(music_vals) + np.std(sine_vals)) / 2
            separability = separation / (combined_std + 1e-6)
            
            print(f"Trennbarkeit Musik/Sinus: {separability:.2f}")
            results_music_vs_sine[feat] = {
                'music_mean': np.mean(music_vals),
                'sine_mean': np.mean(sine_vals),
                'separability': separability,
                'suggested_threshold': (np.mean(music_vals) + np.mean(sine_vals)) / 2
            }
        
        # Trennbarkeit Sprache vs Sinus
        if len(speech_vals) > 0 and len(sine_vals) > 0:
            separation = abs(np.mean(speech_vals) - np.mean(sine_vals))
            combined_std = (np.std(speech_vals) + np.std(sine_vals)) / 2
            separability = separation / (combined_std + 1e-6)
            
            print(f"Trennbarkeit Sprache/Sinus: {separability:.2f}")
            results_speech_vs_sine[feat] = {
                'speech_mean': np.mean(speech_vals),
                'sine_mean': np.mean(sine_vals),
                'separability': separability,
                'suggested_threshold': (np.mean(speech_vals) + np.mean(sine_vals)) / 2
            }
        
        # Trennbarkeit Musik vs Sprache
        if len(music_vals) > 0 and len(speech_vals) > 0:
            separation = abs(np.mean(music_vals) - np.mean(speech_vals))
            combined_std = (np.std(music_vals) + np.std(speech_vals)) / 2
            separability = separation / (combined_std + 1e-6)
            
            print(f"Trennbarkeit Musik/Sprache: {separability:.2f}")
            results_music_vs_speech[feat] = {
                'music_mean': np.mean(music_vals),
                'speech_mean': np.mean(speech_vals),
                'separability': separability,
                'suggested_threshold': (np.mean(music_vals) + np.mean(speech_vals)) / 2
            }
        
        print()
    
    # Grenzwert-Empfehlungen
    print("\n" + "="*70)
    print("=== EMPFOHLENE GRENZWERTE FÜR C-CODE ===")
    print("="*70)
    
    # MUSIK vs SINUS
    print("\n--- KONFIGURATION 1: MUSIK vs SINUS ---")
    print("// Für Unterscheidung zwischen Musik und Sinussignalen\n")
    
    print_thresholds(results_music_vs_sine)
    print_weights(results_music_vs_sine, "Musik/Sinus")
    
    # SPRACHE vs SINUS
    print("\n--- KONFIGURATION 2: SPRACHE vs SINUS ---")
    print("// Für Unterscheidung zwischen Sprache und Sinussignalen\n")
    
    print_thresholds(results_speech_vs_sine)
    print_weights(results_speech_vs_sine, "Sprache/Sinus")
    
    # MUSIK vs SPRACHE
    print("\n--- KONFIGURATION 3: MUSIK vs SPRACHE ---")
    print("// Für Unterscheidung zwischen Musik und Sprache\n")
    
    print_thresholds(results_music_vs_speech)
    print_weights(results_music_vs_speech, "Musik/Sprache")
    
    # EMPFEHLUNG
    print("\n" + "="*70)
    print("=== EMPFEHLUNG ===")
    print("="*70)
    give_recommendation(results_music_vs_sine, results_speech_vs_sine, results_music_vs_speech)
    
    # Visualisierung erstellen
    create_visualization(stats_music, stats_speech, stats_sine, stats_noise, features, feature_names)
    
    return results_music_vs_sine, results_speech_vs_sine, results_music_vs_speech


def print_thresholds(results):
    """Gibt Grenzwerte aus"""
    
    if 'centroid' in results:
        thresh = results['centroid']['suggested_threshold']
        print(f"uint16_t threshCentroid = {int(thresh)};  // {thresh:.1f} Hz")
    
    if 'power' in results:
        thresh = results['power']['suggested_threshold']
        print(f"uint16_t threshPower = {max(1, int(abs(thresh)))};     // {thresh:.1f} dB")
    
    if 'energy' in results:
        thresh = results['energy']['suggested_threshold']
        print(f"uint16_t threshEnergy = {max(1, int(thresh))};    // {thresh:.2e}")
    
    if 'peak_freq' in results:
        thresh = results['peak_freq']['suggested_threshold']
        print(f"uint16_t threshpeakF = {int(thresh)};     // {thresh:.1f} Hz")
    
    if 'flux' in results:
        thresh = results['flux']['suggested_threshold']
        print(f"uint16_t threshFlux = {max(1, int(thresh*100))};      // {thresh:.2e}")
    
    if 'bandwidth' in results:
        thresh = results['bandwidth']['suggested_threshold']
        print(f"uint16_t threshBandwidth = {int(thresh)}; // {thresh:.1f} Hz")
    
    if 'flatness' in results:
        thresh = results['flatness']['suggested_threshold']
        print(f"float32_t threshFlatness = {thresh:.4f}f;  // {thresh:.4f}")
    
    if 'rolloff' in results:
        thresh = results['rolloff']['suggested_threshold']
        print(f"uint16_t threshRolloff = {int(thresh)};   // {thresh:.1f} Hz")


def print_weights(results, label):
    """Gibt Gewichtungen aus"""
    print(f"\n// Feature-Gewichtungen (basierend auf Trennbarkeit für {label}):\n")
    
    if results:
        total_sep = sum([r['separability'] for r in results.values()])
        
        for feat in ['centroid', 'power', 'energy', 'peak_freq', 'flux', 'bandwidth', 'flatness', 'rolloff']:
            if feat in results:
                weight = int(10 * results[feat]['separability'] / total_sep)
                weight = max(1, weight)
                
                feat_names = {
                    'centroid': 'weightCentroid',
                    'power': 'weightPower',
                    'energy': 'weightEnergy',
                    'peak_freq': 'weightpeakF',
                    'flux': 'weightFlux',
                    'bandwidth': 'weightBandwidth',
                    'flatness': 'weightFlatness',
                    'rolloff': 'weightRolloff'
                }
                
                print(f"uint16_t {feat_names[feat]} = {weight};  // Trennbarkeit: {results[feat]['separability']:.2f}")


def give_recommendation(music_sine, speech_sine, music_speech):
    """Gibt eine Empfehlung basierend auf den Trennbarkeiten"""
    
    # Durchschnittliche Trennbarkeit berechnen
    avg_music_sine = np.mean([r['separability'] for r in music_sine.values()])
    avg_speech_sine = np.mean([r['separability'] for r in speech_sine.values()])
    avg_music_speech = np.mean([r['separability'] for r in music_speech.values()])
    
    print(f"\nDurchschnittliche Trennbarkeit:")
    print(f"  Musik vs Sinus:   {avg_music_sine:.2f}")
    print(f"  Sprache vs Sinus: {avg_speech_sine:.2f}")
    print(f"  Musik vs Sprache: {avg_music_speech:.2f}\n")
    
    # Beste Features identifizieren
    print("Beste Features für jede Unterscheidung:\n")
    
    print("Musik vs Sinus:")
    sorted_music_sine = sorted(music_sine.items(), key=lambda x: x[1]['separability'], reverse=True)
    for i, (feat, data) in enumerate(sorted_music_sine[:3]):
        print(f"  {i+1}. {feat}: Trennbarkeit {data['separability']:.2f}")
    
    print("\nSprache vs Sinus:")
    sorted_speech_sine = sorted(speech_sine.items(), key=lambda x: x[1]['separability'], reverse=True)
    for i, (feat, data) in enumerate(sorted_speech_sine[:3]):
        print(f"  {i+1}. {feat}: Trennbarkeit {data['separability']:.2f}")
    
    print("\nMusik vs Sprache:")
    sorted_music_speech = sorted(music_speech.items(), key=lambda x: x[1]['separability'], reverse=True)
    for i, (feat, data) in enumerate(sorted_music_speech[:3]):
        print(f"  {i+1}. {feat}: Trennbarkeit {data['separability']:.2f}")
    
    # Hauptempfehlung
    print("\n" + "-"*70)
    print("HAUPTEMPFEHLUNG:")
    print("-"*70)
    
    if avg_music_sine > avg_speech_sine and avg_music_sine > avg_music_speech:
        print("\n✓ Verwende KONFIGURATION 1 (Musik vs Sinus)")
        print("  → Beste Gesamttrennbarkeit")
        print("  → Optimal wenn hauptsächlich Musik von Sinussignalen unterschieden werden soll")
    elif avg_speech_sine > avg_music_sine and avg_speech_sine > avg_music_speech:
        print("\n✓ Verwende KONFIGURATION 2 (Sprache vs Sinus)")
        print("  → Beste Gesamttrennbarkeit")
        print("  → Optimal wenn hauptsächlich Sprache von Sinussignalen unterschieden werden soll")
    else:
        print("\n✓ Verwende eine HYBRIDE KONFIGURATION")
        print("  → Nutze die besten Features aus beiden Konfigurationen")
        print(f"  → Fokus auf: {sorted_music_sine[0][0]}, {sorted_speech_sine[0][0]}")
    
    # Spezielle Hinweise
    print("\nWICHTIGE HINWEISE:")
    if 'flatness' in music_sine and music_sine['flatness']['separability'] > 5:
        print("  • Flatness ist SEHR gut zur Sinus-Erkennung (hohe Gewichtung empfohlen!)")
    
    if 'bandwidth' in music_sine and music_sine['bandwidth']['separability'] > 3:
        print("  • Bandwidth ist sehr effektiv (Musik/Sprache haben breiteres Spektrum)")
    
    if avg_music_speech < 1.5:
        print("  ⚠ Musik und Sprache sind schwer zu unterscheiden - Kombination mehrerer Features nötig")


def create_visualization(stats_music, stats_speech, stats_sine, stats_noise, features, feature_names):
    """Erstellt Visualisierungen der Feature-Verteilungen"""
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    for idx, (feat, name) in enumerate(zip(features, feature_names)):
        ax = axes[idx]
        
        data_to_plot = []
        labels = []
        
        if len(stats_music[feat]) > 0:
            data_to_plot.append(stats_music[feat])
            labels.append('Musik')
        
        if len(stats_speech[feat]) > 0:
            data_to_plot.append(stats_speech[feat])
            labels.append('Sprache')
        
        if len(stats_sine[feat]) > 0:
            data_to_plot.append(stats_sine[feat])
            labels.append('Sinus')
        
        if len(stats_noise[feat]) > 0:
            data_to_plot.append(stats_noise[feat])
            labels.append('Rauschen')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(name, fontsize=13, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', rotation=15)
            
            # Y-Achse wissenschaftliche Notation für sehr kleine/große Werte
            if feat in ['energy', 'flux']:
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    plt.suptitle('Feature-Verteilungen: Musik, Sprache, Sinus & Rauschen', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualisierung gespeichert als 'feature_analysis.png'")
    plt.close()


# Hauptprogramm
if __name__ == "__main__":
    csv_directory = "C:/eigene Programme/VS_Code_Programme/HKA/DSP/CSV"
    
    print("Feature Threshold Analyzer - Musik vs Sprache vs Sinus")
    print("=" * 70)
    print()
    
    results = analyze_csv_files(csv_directory)
    
    print("\n" + "=" * 70)
    print("Analyse abgeschlossen!")
    print("=" * 70)