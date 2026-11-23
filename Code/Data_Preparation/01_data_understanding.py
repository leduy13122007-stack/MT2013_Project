import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("Data/kc_house_data.csv")

# 2. Dataset overview
print("Dataset shape:", df.shape)

# 3. Display first 5 rows
print(df.head())

# 4. Dataset information
df.info()

# 5. Descriptive statistics
print(df.describe().T)
# 6. Column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 7. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 8. Duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 9. Repeated IDs
print("\n--- REPEATED IDs ---")
print(df["id"].duplicated().sum())

# 10. Date range
print("\n--- DATE RANGE ---")
df["date"] = pd.to_datetime(df["date"])
print("Start:", df["date"].min())
print("End:", df["date"].max())
# 11. Data Dictionary

data_dictionary = {
    "id": ["Identifier", "Unique property identifier"],
    "date": ["Datetime", "Date of sale"],
    "price": ["Quantitative", "Selling price (USD)"],
    "bedrooms": ["Quantitative", "Number of bedrooms"],
    "bathrooms": ["Quantitative", "Number of bathrooms"],
    "sqft_living": ["Quantitative", "Living area (sqft)"],
    "sqft_lot": ["Quantitative", "Lot area (sqft)"],
    "floors": ["Quantitative", "Number of floors"],
    "waterfront": ["Qualitative", "Whether the property has a waterfront view"],
    "view": ["Ordinal", "View quality rating (0-4)"],
    "condition": ["Ordinal", "Property condition rating (1-5)"],
    "grade": ["Ordinal", "Construction and design quality rating"],
    "sqft_above": ["Quantitative", "Above-ground living area (sqft)"],
    "sqft_basement": ["Quantitative", "Basement area (sqft)"],
    "yr_built": ["Quantitative", "Year built"],
    "yr_renovated": ["Quantitative", "Year renovated"],
    "zipcode": ["Qualitative", "ZIP code"],
    "lat": ["Quantitative", "Latitude"],
    "long": ["Quantitative", "Longitude"],
    "sqft_living15": ["Quantitative", "Average living area of nearby 15 houses (sqft)"],
    "sqft_lot15": ["Quantitative", "Average lot area of nearby 15 houses (sqft)"]
}

dictionary_df = pd.DataFrame.from_dict(
    data_dictionary,
    orient="index",
    columns=["Variable Type", "Description"]
)

dictionary_df.index.name = "Variable"

print("\n--- DATA DICTIONARY ---")
print(dictionary_df)

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"

dictionary_df.to_csv(
    DATA_DIR / "data_dictionary.csv",
    encoding="utf-8-sig"
)

print("\nData Dictionary exported successfully!")