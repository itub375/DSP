from pathlib import Path
from pydub import AudioSegment

def mp4_to_mp3(mp4_path: str, mp3_path: str | None = None, bitrate: str = "192k") -> str:
    mp4 = Path(mp4_path)
    if not mp4.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {mp4}")

    mp3 = Path(mp3_path) if mp3_path else mp4.with_suffix(".mp3")

    audio = AudioSegment.from_file(mp4, format="mp4")
    audio.export(mp3, format="mp3", bitrate=bitrate)

    return str(mp3)

if __name__ == "__main__":
    inp = input("Welche MP4-Datei soll umgewandelt werden? ").strip().strip('"')
    out = mp4_to_mp3(inp)
    print(f"Fertig: {out}")
