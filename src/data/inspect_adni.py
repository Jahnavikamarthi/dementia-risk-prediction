
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOLDERS = [
    PROJECT_ROOT / "data" / "raw" / "clinical",
    PROJECT_ROOT / "data" / "raw" / "mri_numerical"
]

for folder in FOLDERS:

    print("\n================================")
    print("FOLDER:", folder.name)
    print("================================")

    if not folder.exists():
        print("Folder not found!")
        continue

    files = [
        f for f in folder.rglob("*")
        if f.is_file()
    ]

    if not files:
        print("No files found!")
        continue

    for file in files:

        print("\nFILE:", file.name)
        print("FORMAT:", file.suffix)

        try:

            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file, low_memory=False)

            elif file.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file)

            elif file.suffix.lower() == ".txt":
                df = pd.read_csv(file, sep=None, engine="python")

            else:
                print("Unsupported format for inspection.")
                continue

            print("Rows:", len(df))
            print("Columns:", len(df.columns))

            print("\nColumn names:")
            print(df.columns.tolist())

            print("\nFirst 5 rows:")
            print(df.head())

        except Exception as error:
            print("Reading error:", error)
