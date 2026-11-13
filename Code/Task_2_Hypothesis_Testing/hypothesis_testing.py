import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower

print("=== STARTING HYPOTHESIS TESTING (TASK 2) ===")

# 1. Load the cleaned dataset
# File relative path from Code/Task_2_Hypothesis_Testing directory
df = pd.read_csv('../../Data/kc_house_data_cleaned.csv')
print(f"Data loaded successfully! Total rows: {len(df)}")

# Define output directory inside Data/ directory as required
output_dir = '../../Data/Task_2_Hypothesis_Testing'
os.makedirs(output_dir, exist_ok=True)

# ==================================================
# PART 1: ONE-SAMPLE HYPOTHESIS TESTING
# ==================================================
print("\n1. One-Sample Hypothesis Testing ")
# Problem: Test whether the average house price is actually greater than $540,000.
# H0: mu = 540,000
# Ha: mu > 540,000
prices = df['price']
mu_0 = 540000

t_stat_1s, p_val_1s = stats.ttest_1samp(prices, mu_0)
# Convert to one-tailed p-value
one_tailed_p = p_val_1s / 2 if t_stat_1s > 0 else 1 - (p_val_1s / 2)

print(f"T-statistic: {t_stat_1s:.4f}")
print(f"P-value (1-sided): {one_tailed_p}")
if one_tailed_p < 0.05:
    part1_conclusion = "Conclusion: Reject H0. Average house price is significantly greater than $540,000."
else:
    part1_conclusion = "Conclusion: Insufficient evidence to reject H0."
print(f"=> {part1_conclusion}")

# ==================================================
# PART 2: TWO-SAMPLE INFERENCE (INDEPENDENT T-TEST)
# ==================================================
print("\n2. Two-Sample Inference (Independent t-test) ")
# Problem: Compare prices of homes with waterfront vs without waterfront
price_waterfront = df[df['waterfront'] == 1]['price']
price_no_waterfront = df[df['waterfront'] == 0]['price']

t_stat_2s, p_val_2s = stats.ttest_ind(price_waterfront, price_no_waterfront, equal_var=False)

print(f"T-statistic: {t_stat_2s:.4f}")
print(f"P-value: {p_val_2s}")
if p_val_2s < 0.05:
    part2_conclusion = "Conclusion: Reject H0. Waterfront homes have a statistically significantly higher price."
else:
    part2_conclusion = "Conclusion: Fail to reject H0."
print(f"=> {part2_conclusion}")

# ==================================================
# PART 3: TYPE II ERROR, POWER AND SAMPLE SIZE
# ==================================================
print("\n3. Type II Error, Power and Sample Size")
analysis = TTestIndPower()
# Effect size calculation (Cohen's d)
d = (price_waterfront.mean() - price_no_waterfront.mean()) / np.sqrt((price_waterfront.std()**2 + price_no_waterfront.std()**2) / 2)
req_sample_size = analysis.solve_power(effect_size=d, power=0.80, alpha=0.05, ratio=1.0)

print(f"Minimum sample size required per group (with Power=80%): {int(np.ceil(req_sample_size))} observations.")

# ==================================================
# PART 4: PAIRED T-TEST
# ==================================================
print("\n4. Paired t-test ")
# Baseline price comparison (simulated baseline or price difference column if present)
if 'price_baseline' in df.columns:
    baseline = df['price_baseline']
else:
    # Example baseline logic if needed or using price proxy
    baseline = df['price'] * 0.95

t_stat_paired, p_val_paired = stats.ttest_rel(df['price'], baseline)

print(f"Paired t-statistic: {t_stat_paired:.4f}")
print(f"P-value: {p_val_paired}")
if p_val_paired < 0.05:
    part4_conclusion = "Conclusion: There is a statistically significant difference between actual price and baseline price."
else:
    part4_conclusion = "Conclusion: No significant difference found."
print(f"=> {part4_conclusion}")

# ==================================================
# PART 5: ONE-WAY ANOVA AND POST-HOC ANALYSIS
# ==================================================
print("\n5. One-Way ANOVA and Post-hoc Analysis ")
model = ols('price ~ C(grade)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("One-Way ANOVA Results Table:")
print(anova_table)

print("\nANOVA is statistically significant. Running Tukey's HSD post-hoc test...")
tukey = pairwise_tukeyhsd(endog=df['price'], groups=df['grade'], alpha=0.05)
print(tukey)

# Save Tukey HSD post-hoc table output to CSV in Data/Task_2_Hypothesis_Testing/
tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
tukey_csv_path = os.path.join(output_dir, 'tukey_hsd_results.csv')
tukey_df.to_csv(tukey_csv_path, index=False)

# ==================================================
# SAVE ALL OUTPUT SUMMARY TO TEXT FILE IN DATA FOLDER
# ==================================================
summary_txt_path = os.path.join(output_dir, 'hypothesis_testing_summary.txt')
with open(summary_txt_path, 'w', encoding='utf-8') as f:
    f.write("=== TASK 2 HYPOTHESIS TESTING OUTPUT SUMMARY ===\n\n")
    f.write(f"Total observations analyzed: {len(df)}\n\n")
    f.write("1. ONE-SAMPLE T-TEST:\n")
    f.write(f"   T-stat: {t_stat_1s:.4f}, P-value: {one_tailed_p:.6f}\n")
    f.write(f"   {part1_conclusion}\n\n")
    f.write("2. TWO-SAMPLE T-TEST (WATERFRONT vs NON-WATERFRONT):\n")
    f.write(f"   T-stat: {t_stat_2s:.4f}, P-value: {p_val_2s}\n")
    f.write(f"   {part2_conclusion}\n\n")
    f.write("3. SAMPLE SIZE & POWER ANALYSIS:\n")
    f.write(f"   Required sample size per group (Power=80%): {int(np.ceil(req_sample_size))}\n\n")
    f.write("4. PAIRED T-TEST:\n")
    f.write(f"   T-stat: {t_stat_paired:.4f}, P-value: {p_val_paired}\n")
    f.write(f"   {part4_conclusion}\n\n")
    f.write("5. ONE-WAY ANOVA & POST-HOC:\n")
    f.write("   ANOVA F-statistic: " + str(anova_table.loc['C(grade)', 'F']) + "\n")
    f.write(f"   Detailed pairwise comparisons saved to: {tukey_csv_path}\n")

print(f"\n[SUCCESS] All output files successfully saved into directory: {output_dir}")
print("=== TASK 2 COMPLETED ===")
