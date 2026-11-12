
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

folder = ROOT / "data" / "raw" / "clinical"

files = list(folder.rglob("DATADIC*.csv"))

if not files:
    raise FileNotFoundError("Clinical DATADIC not found")

df = pd.read_csv(files[0], low_memory=False)

keywords = [
    "DXSUM",
    "DIAGNOSIS",
    "MMSE",
    "CDR",
    "ADAS",
    "DEMOGRAPHIC",
    "PTDEMOG",
    "FAQ",
    "MOCA"
]

mask = (
    df["TBLNAME"].astype(str).str.contains(
        "|".join(keywords), case=False, na=False
    )
    |
    df["CRFNAME"].astype(str).str.contains(
        "|".join(keywords), case=False, na=False
    )
)

result = df.loc[
    mask,
    ["CRFNAME", "TBLNAME", "FLDNAME", "TEXT", "TYPE", "UNITS"]
].drop_duplicates()

print("\n===== RELEVANT ADNI CLINICAL TABLES =====")

for table in result["TBLNAME"].dropna().unique():
    print("\nTABLE:", table)

    print(
        result[result["TBLNAME"] == table][
            ["FLDNAME", "TEXT", "TYPE", "UNITS"]
        ].head(15).to_string(index=False)
    )

output = ROOT / "reports"
output.mkdir(exist_ok=True)

result.to_csv(
    output / "relevant_clinical_tables.csv",
    index=False
)

print("\nSaved table information in reports/")
