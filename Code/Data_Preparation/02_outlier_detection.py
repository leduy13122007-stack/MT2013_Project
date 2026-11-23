
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Data" / "kc_house_data.csv"
OUTPUT_DIR = PROJECT_DIR / "Data" / "Outlier_Analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 2. Select numerical variables
variables = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement"
]

# 3. Calculate IQR
results = []

for col in variables:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    results.append({
        "Variable": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Outlier Count": len(outliers),
        "Outlier Percentage (%)": len(outliers) / len(df) * 100
    })

# 4. Display results
outlier_df = pd.DataFrame(results)

print("\n--- IQR OUTLIER ANALYSIS ---")
print(outlier_df.to_string(index=False))

# 5. Export results
outlier_df.to_csv(
    OUTPUT_DIR / "outlier_summary.csv",
    index=False
)

# 6. Draw boxplots
for col in variables:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"boxplot_{col}.png")
    plt.close()

print("\nOutlier analysis completed.")
print("Results saved in:", OUTPUT_DIR)
