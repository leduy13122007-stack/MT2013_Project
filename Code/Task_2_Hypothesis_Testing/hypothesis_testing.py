import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ==========================================
# INITIALIZATION AND DATA LOADING
# ==========================================
# Load the cleaned dataset according to the required folder structure
df = pd.read_csv('../../Data/kc_house_data_cleaned.csv')

print("="*50)
print("PART I: ONE-SAMPLE HYPOTHESIS TESTING")
print("="*50)

price_data = df['price'].dropna()
mu_0 = 540000

# Perform One-sample t-test
t_stat, p_val = stats.ttest_1samp(a=price_data, popmean=mu_0)

print(f"Sample Mean: ${price_data.mean():.2f}")
print(f"Test Statistic (t): {t_stat:.4f}")
print(f"P-value: {p_val:.4e}\n")


print("="*50)
print("PART II: TWO-SAMPLE INFERENCE & POWER ANALYSIS")
print("="*50)

waterfront = df[df['waterfront'] == 1]['price']
no_waterfront = df[df['waterfront'] == 0]['price']

# Welch's t-test (unequal variances)
t_stat_w, p_val_w = stats.ttest_ind(waterfront, no_waterfront, equal_var=False)

# 95% Confidence Interval
cm = sm.stats.CompareMeans(sm.stats.DescrStatsW(waterfront), sm.stats.DescrStatsW(no_waterfront))
ci = cm.tconfint_diff(alpha=0.05, usevar='unequal')

print(f"Welch's t-statistic: {t_stat_w:.4f}")
print(f"P-value: {p_val_w:.4e}")
print(f"95% Confidence Interval for (mu1 - mu2): {ci}")

# Type II Error, Power & Sample Size
delta = 100000  # Target difference of 100,000 USD
pooled_std = df['price'].std()
effect_size = delta / pooled_std

power_analysis = TTestIndPower()
required_n = power_analysis.solve_power(effect_size=effect_size, power=0.90, alpha=0.05, ratio=1.0)
print(f"Required sample size per group (Power=0.90, Delta=$100K): {np.ceil(required_n)}\n")


print("="*50)
print("PART III: PAIRED T-TEST")
print("="*50)

sqft_liv = df['sqft_living'].dropna()
sqft_liv15 = df['sqft_living15'].dropna()

# Paired t-test
t_stat_paired, p_val_paired = stats.ttest_rel(sqft_liv, sqft_liv15)

print(f"Paired t-statistic: {t_stat_paired:.4f}")
print(f"P-value: {p_val_paired:.4e}\n")


print("="*50)
print("PART IV: ONE-WAY ANOVA & POST-HOC ANALYSIS")
print("="*50)

# Group house prices by condition (1 to 5)
groups = [df['price'][df['condition'] == i] for i in range(1, 6)]

# Run ANOVA
F_stat, p_val_anova = stats.f_oneway(*groups)
print(f"ANOVA F-statistic: {F_stat:.4f}")
print(f"P-value: {p_val_anova:.4e}")

# Assumption Check 1: Levene's Test (Equal Variances)
levene_stat, levene_p = stats.levene(*groups)
print(f"Levene's test p-value: {levene_p:.4e}\n")

# Post-hoc Analysis: Tukey's HSD
print("--- Tukey HSD Analysis Results ---")
tukey = pairwise_tukeyhsd(endog=df['price'], groups=df['condition'], alpha=0.05)
print(tukey)

# ==========================================
# VISUALIZATION
# ==========================================
# Ensure output directory exists
os.makedirs('../../Data/Task_2_Hypothesis_Testing', exist_ok=True)

# 1. Q-Q Plot to check normality of residuals (ANOVA)
residuals = df['price'] - df.groupby('condition')['price'].transform('mean')
fig_qq = sm.qqplot(residuals, line='s')
plt.title("Q-Q Plot of Residuals")
plt.savefig('../../Data/Task_2_Hypothesis_Testing/qq_plot_residuals.png')
plt.close()

# 2. Multiple Comparisons plot from Tukey HSD
fig_tukey = tukey.plot_simultaneous(figsize=(8, 6))
plt.title("Tukey HSD - Multiple Comparisons")
plt.xlabel("Price")
plt.ylabel("Condition")
plt.savefig('../../Data/Task_2_Hypothesis_Testing/tukey_hsd_plot.png')
plt.close()

print("\nPlots have been saved to Data/Task_2_Hypothesis_Testing/")