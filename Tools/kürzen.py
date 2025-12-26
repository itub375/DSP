from pydub import AudioSegment

# ===== Einstellungen =====
input_file  = "C:/eigene_Programme/VS_Code_Programme/HKA/DSP/Raw_signals/Podcast.mp3"         # Eingabedatei
output_file = r"Podcast_shorted.mp3" # Ausgabedatei

# Variante 1: Zeit in Millisekunden angeben
trim_ms = 13500  # z.B. 2000 ms = 2 Sekunden

# ODER Variante 2: Zeit in Sekunden angeben
# trim_sec = 2.0
# trim_ms = int(trim_sec * 1000)

# ===== MP3 laden =====
audio = AudioSegment.from_file(input_file, format="mp3")

# Sicherheitscheck
if len(audio) <= trim_ms:
    raise ValueError(f"Datei ist nur {len(audio)} ms lang – weniger als trim_ms={trim_ms} ms.")

# ===== vorne abschneiden =====
# audio[start_ms:end_ms] – wenn end_ms weggelassen wird, geht es bis zum Ende
gekürzt = audio[trim_ms:]

# ===== neue MP3 speichern =====
gekürzt.export(output_file, format="mp3")
print(f"Fertig! Gespeichert als: {output_file}")
