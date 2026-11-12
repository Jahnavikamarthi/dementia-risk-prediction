
from pathlib import Path
import pandas as pd

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset folder
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"

print("Dementia Risk Prediction Project")
print("--------------------------------")

# Check whether the dataset folder exists
if DATA_FOLDER.exists():
    print("Dataset folder found!")

    files = list(DATA_FOLDER.glob("*"))

    if not files:
        print("No datasets found yet. Add your dataset later.")
    else:
        for file in files:
            print("Found:", file.name)

            if file.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(file)
                    print("Rows and columns:", df.shape)
                    print(df.head())
                except Exception as error:
                    print("Could not read file:", error)
else:
    print("Dataset folder not found!")
