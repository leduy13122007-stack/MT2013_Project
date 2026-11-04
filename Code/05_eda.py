from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Data" / "kc_house_data_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "Data" / "EDA"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Display basic information
print("Dataset shape:", df.shape)
print("\nDescriptive statistics:")
print(df.describe())
# Select numerical variables
variables = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot"
]

# Plot histograms
for variable in variables:
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=variable,
        bins=30,
        kde=True
    )

    plt.title(f"Distribution of {variable}")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{variable}_distribution.png")
    plt.close()

print("\nUnivariate analysis completed!")
# ==========================================
# CATEGORICAL ANALYSIS
# ==========================================

categorical_variables = [
    "waterfront",
    "condition",
    "grade"
]

for variable in categorical_variables:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=variable,
        order=sorted(df[variable].unique())
    )

    plt.title(f"Distribution of {variable}")
    plt.xlabel(variable)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{variable}_distribution.png"
    )

    plt.close()

    print(f"\nFrequency table: {variable}")
    print(df[variable].value_counts().sort_index())

print("\nCategorical analysis completed!")
# ==========================================
# BIVARIATE ANALYSIS
# ==========================================

# 1. Price vs Living Area
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="sqft_living",
    y="price",
    alpha=0.3
)

plt.title("Price vs Living Area")
plt.xlabel("Living Area (sqft)")
plt.ylabel("Price (USD)")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "price_vs_sqft_living.png")
plt.close()


# 2. Price vs Bedrooms
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="bedrooms",
    y="price"
)

plt.title("Price Distribution by Bedrooms")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price (USD)")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "price_vs_bedrooms.png")
plt.close()


# 3. Price vs Grade
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="grade",
    y="price"
)

plt.title("Price Distribution by Grade")
plt.xlabel("Grade")
plt.ylabel("Price (USD)")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "price_vs_grade.png")
plt.close()

print("\nBivariate analysis completed!")
# ==========================================
# CORRELATION ANALYSIS
# ==========================================

# Select numerical variables
correlation_variables = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "sqft_living15",
    "sqft_lot15"
]

# Calculate Pearson correlation matrix
correlation_matrix = df[correlation_variables].corr(method="pearson")

# Display correlation with price
print("\nCorrelation with price:")
print(
    correlation_matrix["price"]
    .sort_values(ascending=False)
)

# Plot heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
plt.close()

# Export correlation matrix
correlation_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

print("\nCorrelation analysis completed!")