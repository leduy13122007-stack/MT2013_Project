
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_DIR / "Data" / "kc_house_data_cleaned.csv"

OUTPUT_DIR = PROJECT_DIR / "Data" / "Task_3_Regression"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

X_COL = "sqft_living"
Y_COL = "price"

ALPHA = 0.05

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

data = df[[X_COL, Y_COL]].dropna()

x = data[X_COL]
y = data[Y_COL]

print("=" * 55)
print("TASK 3 - CORRELATION AND REGRESSION")
print("=" * 55)

print(f"Dataset: {DATA_PATH.name}")
print(f"Number of observations: {len(data)}")
print(f"Independent variable: {X_COL}")
print(f"Dependent variable: {Y_COL}")

# ==========================================
# 3. PEARSON CORRELATION ANALYSIS
# ==========================================

print("\n3.4.1 PEARSON CORRELATION ANALYSIS")

r, p_value = stats.pearsonr(x, y)

n = len(data)

# Fisher's z transformation for 95% CI
z = np.arctanh(r)
se_z = 1 / np.sqrt(n - 3)

z_critical = stats.norm.ppf(1 - ALPHA / 2)

ci_lower = np.tanh(z - z_critical * se_z)
ci_upper = np.tanh(z + z_critical * se_z)

print(f"\nSample size (n): {n}")
print(f"Pearson correlation coefficient (r): {r:.6f}")
print(f"P-value: {p_value:.6e}")
print(f"95% CI for Pearson correlation: [{ci_lower:.6f}, {ci_upper:.6f}]")

# Hypothesis testing
# H0: rho = 0
# H1: rho != 0

print("\nHypothesis Testing:")
print("H0: Population correlation coefficient (rho) = 0")
print("H1: Population correlation coefficient (rho) != 0")
print(f"Significance level: {ALPHA}")

if p_value < ALPHA:
    print("Decision: Reject H0")
    print("Conclusion: There is statistically significant evidence")
    print("of a linear relationship between living area and house price.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: Insufficient evidence of a linear relationship.")

# ==========================================
# 4. SAVE PEARSON RESULTS
# ==========================================

pearson_results = pd.DataFrame({
    "Metric": [
        "Sample Size",
        "Pearson Correlation",
        "P-value",
        "95% CI Lower",
        "95% CI Upper"
    ],
    "Value": [
        n,
        r,
        p_value,
        ci_lower,
        ci_upper
    ]
})

pearson_results.to_csv(
    RESULTS_DIR / "pearson_correlation.csv",
    index=False
)

print("\nPearson results saved successfully.")

print("\n" + "=" * 55)
print("PEARSON ANALYSIS COMPLETED")
print("=" * 55)

# ==========================================
# 5. SIMPLE LINEAR REGRESSION MODEL
# ==========================================

print("\n3.4.2 SIMPLE LINEAR REGRESSION MODEL")

# Add intercept
X = sm.add_constant(x)

# Fit OLS model
model = sm.OLS(y, X).fit()

# Extract regression coefficients
intercept = model.params["const"]
slope = model.params[X_COL]

print("\nRegression Equation:")
print(f"Predicted price = {intercept:.4f} + {slope:.4f} * sqft_living")

print("\nModel Summary:")
print(f"R-squared: {model.rsquared:.6f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.6f}")
print(f"F-statistic: {model.fvalue:.4f}")
print(f"F-test p-value: {model.f_pvalue:.6e}")

print("\nRegression Coefficients:")
print(f"Intercept: {intercept:.4f}")
print(f"Slope: {slope:.4f}")

print("\nSlope Interpretation:")
print(
    f"For each additional square foot of living area, "
    f"the estimated mean house price increases by "
    f"${slope:.2f}."
)

# Save regression results
regression_results = pd.DataFrame({
    "Metric": [
        "Intercept",
        "Slope",
        "R-squared",
        "Adjusted R-squared",
        "F-statistic",
        "F-test p-value"
    ],
    "Value": [
        intercept,
        slope,
        model.rsquared,
        model.rsquared_adj,
        model.fvalue,
        model.f_pvalue
    ]
})

regression_results.to_csv(
    RESULTS_DIR / "simple_regression_results.csv",
    index=False
)

# Save full OLS summary
with open(RESULTS_DIR / "regression_summary.txt", "w") as f:
    f.write(model.summary().as_text())

print("\nRegression results saved successfully.")
# 6. CONFIDENCE INTERVALS FOR REGRESSION COEFFICIENTS

print("\n3.4.3 CONFIDENCE INTERVALS FOR REGRESSION COEFFICIENTS")

# Calculate 95% confidence intervals
conf_int = model.conf_int(alpha=ALPHA)

# Create coefficient summary table
ci_results = pd.DataFrame({
    "Coefficient": model.params.index,
    "Estimate": model.params.values,
    "Standard Error": model.bse.values,
    "t-value": model.tvalues.values,
    "P-value": model.pvalues.values,
    "CI Lower": conf_int.iloc[:, 0].values,
    "CI Upper": conf_int.iloc[:, 1].values
})

print("\nRegression Coefficient Confidence Intervals:")
print(ci_results.to_string(index=False))

# Save results
ci_results.to_csv(
    RESULTS_DIR / "regression_confidence_intervals.csv",
    index=False
)

print("\nConfidence interval results saved successfully.")
# 7. PREDICTION OF NEW OBSERVATIONS

print("\n3.4.4 PREDICTION OF NEW OBSERVATIONS")

# Select a representative living area (median)
x0 = x.median()

# Prepare input for prediction
new_data = pd.DataFrame({
    "const": [1],
    X_COL: [x0]
})

# Calculate prediction and confidence/prediction intervals
prediction = model.get_prediction(new_data)
prediction_summary = prediction.summary_frame(alpha=ALPHA)

# Extract results
predicted_price = prediction_summary["mean"].iloc[0]
mean_ci_lower = prediction_summary["mean_ci_lower"].iloc[0]
mean_ci_upper = prediction_summary["mean_ci_upper"].iloc[0]
prediction_lower = prediction_summary["obs_ci_lower"].iloc[0]
prediction_upper = prediction_summary["obs_ci_upper"].iloc[0]

# Display results
print(f"Selected living area (x0): {x0:.2f} sqft")
print(f"Predicted house price: ${predicted_price:,.2f}")

print("\n95% Confidence Interval for Mean Price:")
print(f"[${mean_ci_lower:,.2f}, ${mean_ci_upper:,.2f}]")

print("\n95% Prediction Interval for a New House:")
print(f"[${prediction_lower:,.2f}, ${prediction_upper:,.2f}]")

# Save results
prediction_summary.to_csv(
    RESULTS_DIR / "prediction_intervals.csv",
    index=False
)

print("\nPrediction results saved successfully.")
# 8. RESIDUAL ANALYSIS

print("\n3.4.5 RESIDUAL ANALYSIS")

# Calculate fitted values and residuals
fitted_values = model.fittedvalues
residuals = model.resid

# Residual statistics
print(f"Mean of residuals: {residuals.mean():.6f}")
print(f"Standard deviation of residuals: {residuals.std():.2f}")
print(f"Minimum residual: {residuals.min():.2f}")
print(f"Maximum residual: {residuals.max():.2f}")

# 1. Residuals vs Fitted Values
plt.figure(figsize=(8, 5))
plt.scatter(fitted_values, residuals, alpha=0.3, s=10)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted Values (Predicted Price)")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "residuals_vs_fitted.png", dpi=300)
plt.close()

# 2. Normal Q-Q Plot
plt.figure(figsize=(8, 5))
sm.qqplot(residuals, line="45", fit=True)
plt.title("Normal Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "normal_qq_plot.png", dpi=300)
plt.close()

# Save residual data
residual_results = pd.DataFrame({
    "Actual Price": y.values,
    "Fitted Price": fitted_values.values,
    "Residual": residuals.values
})

residual_results.to_csv(
    RESULTS_DIR / "residual_analysis.csv",
    index=False
)

print("\nResidual plots and results saved successfully.")
print(f"Plots saved in: {PLOTS_DIR}")
# Cook's Distance
print("\nCook's Distance Analysis")

influence = model.get_influence()
cooks_distance = influence.cooks_distance[0]

# Reference threshold
threshold = 4 / len(data)

print(f"Maximum Cook's Distance: {cooks_distance.max():.6f}")
print(f"Reference Threshold (4/n): {threshold:.6f}")
print(f"Observations above threshold: {(cooks_distance > threshold).sum()}")

# Plot Cook's Distance
plt.figure(figsize=(8, 5))
plt.stem(range(len(cooks_distance)), cooks_distance)
plt.axhline(
    y=threshold,
    color="red",
    linestyle="--",
    label="Reference threshold (4/n)"
)

plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance Plot")
plt.legend()
plt.tight_layout()

plt.savefig(PLOTS_DIR / "cooks_distance.png", dpi=300)
plt.close()
# Identify Top 10 Observations by Cook's Distance
top_indices = np.argsort(cooks_distance)[-10:][::-1]

top_cooks = pd.DataFrame({
    "Observation Index": top_indices,
    "Property ID": df.loc[data.index[top_indices], "id"].values,
    "sqft_living": x.iloc[top_indices].values,
    "price": y.iloc[top_indices].values,
    "Cook's Distance": cooks_distance[top_indices]
})

print("\nTop 10 Observations by Cook's Distance:")
print(top_cooks.to_string(index=False))

top_cooks.to_csv(
    RESULTS_DIR / "top_10_cooks_distance.csv",
    index=False
)
# 9. REGRESSION ON TRANSFORMED VARIABLES
print("\n3.4.6 REGRESSION ON TRANSFORMED VARIABLES")

# Apply natural logarithm transformation to house price
log_y = np.log(y)

# Fit the transformed regression model
model_log = sm.OLS(log_y, X).fit()

log_intercept = model_log.params["const"]
log_slope = model_log.params[X_COL]

print("\nTransformed Regression Model:")
print(f"ln(price) = {log_intercept:.6f} + {log_slope:.6f} * sqft_living")

print(f"\nR-squared: {model_log.rsquared:.6f}")
print(f"Adjusted R-squared: {model_log.rsquared_adj:.6f}")

# Residual statistics
log_fitted = model_log.fittedvalues
log_residuals = model_log.resid

print("\nResidual Statistics:")
print(f"Mean of residuals: {log_residuals.mean():.6f}")
print(f"Standard deviation of residuals: {log_residuals.std():.6f}")
print(f"Minimum residual: {log_residuals.min():.6f}")
print(f"Maximum residual: {log_residuals.max():.6f}")

# Residuals vs Fitted plot
plt.figure(figsize=(8, 5))
plt.scatter(log_fitted, log_residuals, alpha=0.3, s=10)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted Values (Predicted ln(Price))")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values (Log-transformed Model)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "log_residuals_vs_fitted.png", dpi=300)
plt.close()

# Normal Q-Q plot
plt.figure(figsize=(8, 5))
sm.qqplot(log_residuals, line="45", fit=True)
plt.title("Normal Q-Q Plot of Residuals (Log-transformed Model)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "log_normal_qq_plot.png", dpi=300)
plt.close()

# Breusch-Pagan test for heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(log_residuals, X)

print("\nBreusch-Pagan Test:")
print(f"LM Statistic: {bp_test[0]:.6f}")
print(f"LM p-value: {bp_test[1]:.6e}")

# Save transformed model summary
with open(RESULTS_DIR / "log_regression_summary.txt", "w") as f:
    f.write(model_log.summary().as_text())

print("\nTransformed regression plots and results saved successfully.")
print(f"Plots saved in: {PLOTS_DIR}")