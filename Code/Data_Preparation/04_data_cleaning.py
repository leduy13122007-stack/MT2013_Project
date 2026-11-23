from pathlib import Path
import pandas as pd

# Define paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Data" / "kc_house_data.csv"
OUTPUT_PATH = PROJECT_DIR / "Data" / "kc_house_data_cleaned.csv"

# Load original dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")

# Remove the suspicious record where bedrooms = 33
df_cleaned = df[df["bedrooms"] != 33].copy()

# Check cleaned dataset
print("\nCleaned dataset shape:", df_cleaned.shape)
print("\nRemaining bedrooms = 33:", (df_cleaned["bedrooms"] == 33).sum())
print("\nMissing values:")
print(df_cleaned.isnull().sum())

# Export cleaned dataset
df_cleaned.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset exported successfully!")
print("Saved to:", OUTPUT_PATH)