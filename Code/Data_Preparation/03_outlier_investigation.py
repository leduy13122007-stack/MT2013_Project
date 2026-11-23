
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Data" / "kc_house_data.csv"

df = pd.read_csv(DATA_PATH)

# Variables to investigate
variables = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement"
]

for col in variables:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR

    outliers = df[
        (df[col] < lower) | (df[col] > upper)
    ].sort_values(col, ascending=False)

    print(f"\n{'='*60}")
    print(f"EXTREME VALUES: {col}")
    print(f"{'='*60}")

    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())
    print("Upper bound:", upper)

    print("\nTop 10 largest values:")
    print(
        outliers[
            ["id", "price", "bedrooms", "bathrooms",
             "sqft_living", "sqft_lot", col]
        ].head(10).to_string(index=False)
    )

# Check bedrooms = 33
print("\n--- CHECK BEDROOMS = 33 ---")
print(
    df[df["bedrooms"] == 33].to_string(index=False)
)
