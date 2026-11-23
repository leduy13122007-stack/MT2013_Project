"""
MT2013 - Task 1: Estimation (King County House Sales)
=====================================================
Place at : Code/Task_1_Estimation/point_estimation.py
Input    : Data/kc_house_data_cleaned.csv
Output   : Data/Task_1_Estimation/results/  (tables .csv, output log .txt)
           Data/Task_1_Estimation/plots/    (figures .png)

Sections:
  1. Point estimation (total and mean of price)
  2. 95% confidence interval for the mean (large-sample, Z = 1.96)
  3. Assumptions and data distribution (skewness, histogram, boxplot)

Libs: pandas numpy matplotlib
"""

import math
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "Data" / "kc_house_data_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "Data" / "Task_1_Estimation"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VARIABLE = "price"
Z_CRITICAL = 1.96   # large-sample 95% CI (kept as in the report)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

_logfile = open(RESULTS_DIR / "task1_output_log.txt", "w", encoding="utf-8")


def log(*args):
    """Print to console and write the same text to the log file."""
    text = " ".join(str(a) for a in args)
    print(text)
    _logfile.write(text + "\n")
    _logfile.flush()


def banner(title):
    log("=" * 55)
    log(title)
    log("=" * 55)


def subsection(title):
    log("\n" + title)
    log("-" * 55)


def save_table(table, filename):
    table.to_csv(RESULTS_DIR / filename, index=False)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


# ==========================================
# 3. LOAD DATA
# ==========================================

banner("TASK 1 - ESTIMATION")

df = pd.read_csv(DATA_PATH)

log(f"Dataset: {DATA_PATH.name}")
log(f"Dataset shape: {df.shape}")
log(f"Quantitative variable: {VARIABLE}")
log(f"Missing values: {df[VARIABLE].isnull().sum()}")

# Remove missing values in the selected variable, if any
x = df[VARIABLE].dropna()

# ==========================================
# 4. POINT ESTIMATION
# ==========================================

subsection("1. POINT ESTIMATION")

n = len(x)
total_price = x.sum()
sample_mean = x.mean()

log(f"Variable: {VARIABLE}")
log(f"Sample size (n): {n:,}")
log(f"Total selling price: ${total_price:,.2f}")
log(f"Sample mean: ${sample_mean:,.2f}")

save_table(pd.DataFrame({
    "Metric": ["Variable", "Sample Size (n)", "Total Selling Price", "Sample Mean"],
    "Value": [VARIABLE, n, total_price, sample_mean]
}), "point_estimation_results.csv")

# ==========================================
# 5. 95% CONFIDENCE INTERVAL
# ==========================================

subsection("2. 95% CONFIDENCE INTERVAL")

x_bar = sample_mean
s = x.std(ddof=1)
se = s / math.sqrt(n)
margin = Z_CRITICAL * se

lower = x_bar - margin
upper = x_bar + margin

log("Large-sample CI: x-bar +/- 1.96 * s / sqrt(n)")
log(f"Sample size: {n}")
log(f"Sample mean: ${x_bar:,.2f}")
log(f"Sample standard deviation: ${s:,.2f}")
log(f"Standard error: ${se:,.2f}")
log(f"Margin of error: ${margin:,.2f}")
log(f"95% Confidence Interval: (${lower:,.2f}, ${upper:,.2f})")

save_table(pd.DataFrame({
    "Metric": ["Sample Size (n)", "Sample Mean", "Sample Standard Deviation",
               "Standard Error", "Z critical", "Margin of Error",
               "95% CI Lower", "95% CI Upper"],
    "Value": [n, x_bar, s, se, Z_CRITICAL, margin, lower, upper]
}), "confidence_interval_results.csv")

# ==========================================
# 6. ASSUMPTIONS AND DATA DISTRIBUTION
# ==========================================

subsection("3. ASSUMPTIONS AND DATA DISTRIBUTION")

skewness = x.skew()
log(f"Skewness: {skewness:.4f}")

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(x, bins=50, edgecolor="black")
plt.xlabel("House Price (USD)")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")
save_plot("price_histogram.png")

# Boxplot
plt.figure(figsize=(8, 3))
plt.boxplot(x, vert=False)
plt.xlabel("House Price (USD)")
plt.title("Boxplot of House Prices")
save_plot("price_boxplot.png")

save_table(pd.DataFrame({
    "Metric": ["Skewness"],
    "Value": [skewness]
}), "distribution_check_results.csv")

log("Histogram and boxplot saved successfully.")

# ==========================================
# 7. FINISH
# ==========================================

log("")
banner("TASK 1 COMPLETED")
log(f"Results and log saved in: {RESULTS_DIR}")
log(f"Plots saved in: {PLOTS_DIR}")
_logfile.close()