
from pathlib import Path
import pandas as pd

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output folders
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"


def explore_datasets():
    """Read and inspect CSV datasets."""

    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

    csv_files = list(RAW_FOLDER.glob("*.csv"))

    if not csv_files:
        print("No CSV datasets found in data/raw.")
        print("Add your ADNI CSV files when they are available.")
        return

    for file in csv_files:
        print("\nDataset:", file.name)

        try:
            df = pd.read_csv(file)

            print("Shape:", df.shape)
            print("\nColumn names:")
            print(df.columns.tolist())

            print("\nMissing values:")
            print(df.isnull().sum())

        except Exception as error:
            print("Error reading file:", error)


if __name__ == "__main__":
    print("ADNI Data Preprocessing")
    print("-----------------------")
    explore_datasets()
