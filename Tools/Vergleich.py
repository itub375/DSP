# compare_csv_times.py
# Vergleicht change_ms (CSV_A) mit time_ms (CSV_B)
# Output: matches.csv, only_in_A.csv, only_in_B.csv

import argparse
import os
import sys
import pandas as pd


def read_csv_auto(path: str) -> pd.DataFrame:
    """
    Liest CSV mit automatischer Trennzeichen-Erkennung (',' ';' '\t' ...).
    Funktioniert gut für "deutsche" Excel-CSVs mit ';'.
    """
    return pd.read_csv(path, sep=None, engine="python")


def to_numeric_series(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Spalte '{col}' nicht gefunden in {name}. Verfügbare Spalten: {list(df.columns)}")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.astype(float)


def match_with_tolerance(a_vals, b_vals, tol_ms: float):
    """
    One-to-one Matching via Two-Pointer (greedy).
    Beide Listen müssen sortiert sein.
    Rückgabe: matches(list of dict), only_a(list), only_b(list)
    """
    i, j = 0, 0
    matches = []
    only_a = []
    only_b = []

    while i < len(a_vals) and j < len(b_vals):
        a = a_vals[i]
        b = b_vals[j]
        diff = b - a

        if abs(diff) <= tol_ms:
            matches.append({"change_ms": a, "time_ms": b, "diff_ms": diff})
            i += 1
            j += 1
        elif b < a - tol_ms:
            only_b.append(b)
            j += 1
        else:  # a < b - tol_ms
            only_a.append(a)
            i += 1

    # Rest
    while i < len(a_vals):
        only_a.append(a_vals[i])
        i += 1
    while j < len(b_vals):
        only_b.append(b_vals[j])
        j += 1

    return matches, only_a, only_b


def main():
    parser = argparse.ArgumentParser(description="Vergleicht change_ms (CSV_A) mit time_ms (CSV_B).")
    parser.add_argument("--csv_a", type=str, help="Pfad zu CSV_A (mit Spalte change_ms)")
    parser.add_argument("--csv_b", type=str, help="Pfad zu CSV_B (mit Spalte time_ms)")
    parser.add_argument("--tol", type=float, default=0.0, help="Toleranz in ms (z.B. 3 = ±3 ms). Default: 0")
    parser.add_argument("--out_dir", type=str, default="compare_out", help="Output-Verzeichnis")
    args = parser.parse_args()

    # Wenn nicht per Argument übergeben: per Terminal abfragen
    csv_a = args.csv_a or input("Welche Datei soll bearbeitet werden? (CSV_A Pfad) ").strip().strip('"')
    csv_b = args.csv_b or input("Welche Vergleichsdatei soll genutzt werden? (CSV_B Pfad) ").strip().strip('"')

    if not os.path.isfile(csv_a):
        print(f"Fehler: CSV_A nicht gefunden: {csv_a}")
        sys.exit(1)
    if not os.path.isfile(csv_b):
        print(f"Fehler: CSV_B nicht gefunden: {csv_b}")
        sys.exit(1)

    df_a = read_csv_auto(csv_a)
    df_b = read_csv_auto(csv_b)

    a = to_numeric_series(df_a, "change_ms", "CSV_A").tolist()
    b = to_numeric_series(df_b, "time_ms", "CSV_B").tolist()

    a.sort()
    b.sort()

    matches, only_a, only_b = match_with_tolerance(a, b, 3)

    os.makedirs(args.out_dir, exist_ok=True)

    pd.DataFrame(matches).to_csv(os.path.join(args.out_dir, "matches.csv"), index=False)
    pd.DataFrame({"change_ms_only": only_a}).to_csv(os.path.join(args.out_dir, "only_in_A.csv"), index=False)
    pd.DataFrame({"time_ms_only": only_b}).to_csv(os.path.join(args.out_dir, "only_in_B.csv"), index=False)

    nA = len(a)
    nB = len(b)
    m = len(matches)

    # Prozentwerte (sicher gegen Division durch 0)
    pct_A = (m / nA * 100.0) if nA else 0.0
    pct_B = (m / nB * 100.0) if nB else 0.0
    pct_overall = (2 * m / (nA + nB) * 100.0) if (nA + nB) else 0.0  # "generell" (Dice/F1-ähnlich)

    print("\n--- Vergleich abgeschlossen ---")
    print(f"CSV_A Werte (change_ms): {nA}")
    print(f"CSV_B Werte (time_ms):   {nB}")
    print(f"Matches (±{args.tol} ms): {m}")
    print(f"Nur in A: {len(only_a)}")
    print(f"Nur in B: {len(only_b)}")
    print(f"\nÜbereinstimmung:")
    print(f" - Abdeckung A: {pct_A:.2f}%")
    print(f" - Abdeckung B: {pct_B:.2f}%")
    print(f" - Generell (Dice): {pct_overall:.2f}%")
    print(f"\nOutputs gespeichert in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
