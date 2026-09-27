import pandas as pd
from pathlib import Path

root = Path(r"gp\Maharashtra\Nashik\Sinnar")

print("=" * 100)
print("SINNAR - UPSTREAM SATELLITE DATA CHECK")
print("=" * 100)

for p in root.rglob("*.csv"):
    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    cols = list(df.columns)

    s2 = [c for c in cols if "NDVI" in c.upper() or "EVI" in c.upper() or "SAVI" in c.upper()]
    s1 = [c for c in cols if "ascending" in c.lower()]

    if s2 or s1:
        print("\nFILE:", p)
        print("Rows:", len(df))

        if s2:
            print("\nSentinel-2 related:")
            for c in s2:
                print(" ", c, "| non-NaN:", df[c].notna().sum())

        if s1:
            print("\nSentinel-1 ascending related:")
            for c in s1:
                print(" ", c, "| non-NaN:", df[c].notna().sum())

print("\n" + "=" * 100)
print("END")
print("=" * 100)
