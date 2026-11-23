"""
MT2013 - Task 2: Hypothesis Testing (King County House Sales)
=============================================================
Place at : Code/Task_2_Hypothesis_Testing/hypothesis_testing.py
Input    : Data/kc_house_data_cleaned.csv
Output   : Data/Task_2_Hypothesis_Testing/results/  (tables .csv, output log .txt)
           Data/Task_2_Hypothesis_Testing/plots/    (figures .png)

Sections:
  3.2.1  One-sample t-test (mean price)                    -> a) - f)
  3.2.2  Two-sample inference (price: waterfront vs not)   -> g) - i), k)
  3.2.3  Type II error, power, sample size                 -> j)
  3.2.4  Paired t-test (sqft_living vs sqft_living15)      -> l)
  3.2.5  One-way ANOVA (price by grade group) + post-hoc   -> m) - o)

Libs: pandas numpy scipy statsmodels matplotlib
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_DIR / "Data" / "kc_house_data_cleaned.csv"

OUTPUT_DIR = PROJECT_DIR / "Data" / "Task_2_Hypothesis_Testing"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Test parameters (unchanged from the report)
ALPHA = 0.05
POWER_TARGET = 0.90

MU0 = 500_000               # A: hypothesized mean price (USD)
DELTA0 = 0.0                # B: H0: mu1 - mu2 = DELTA0
DELTA_PRACTICAL = 100_000   # C: practically meaningful difference (USD)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

_logfile = open(RESULTS_DIR / "task2_output_log.txt", "w", encoding="utf-8")


def log(*args):
    """Print to console and write the same text to the log file."""
    text = " ".join(str(a) for a in args)
    print(text)
    _logfile.write(text + "\n")
    _logfile.flush()


def banner(title):
    log("\n" + "=" * 55)
    log(title)
    log("=" * 55)


def subsection(title):
    log("\n" + title)


def decision(p):
    return "Reject H0" if p < ALPHA else "Fail to reject H0"


def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def save_table(table, filename, index=False):
    table.to_csv(RESULTS_DIR / filename, index=index)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


# ==========================================
# 3. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

# Grade groups for the ANOVA (same bins and labels as before)
df["grade_group"] = pd.cut(
    df["grade"],
    bins=[0, 6, 7, 9, 20],
    labels=["1 Low (<=6)", "2 Average (7)", "3 Good (8-9)", "4 High (>=10)"]
)

banner("TASK 2 - HYPOTHESIS TESTING")
log(f"Dataset: {DATA_PATH.name}")
log(f"Number of observations: {len(df)}")
log(f"Number of columns: {df.shape[1]}")
log(f"Significance level (alpha): {ALPHA}")

# Information about repeated property IDs (used when discussing independence).
# This is only a diagnostic count; no observation is removed.
if "id" in df.columns:
    n_dup_rows = int(df["id"].duplicated().sum())
    n_dup_ids = int(df.loc[df["id"].duplicated(), "id"].nunique())
    log(f"Repeated property IDs: {n_dup_ids} IDs appear more than once "
        f"({n_dup_rows} extra rows). No observation is removed here.")
else:
    n_dup_rows, n_dup_ids = None, None

summary = []   # rows for summary_tests.csv

# ==========================================
# 4. 3.2.1 ONE-SAMPLE t-TEST (a - f)
# ==========================================

banner("3.2.1 ONE-SAMPLE HYPOTHESIS TEST (mean price)")

x = df["price"].dropna()
n = len(x)
x_bar = x.mean()
s = x.std(ddof=1)
se = s / np.sqrt(n)

log(f"a) Research question: is the mean sale price different from the benchmark "
    f"of {MU0:,.0f} USD?")
log(f"b) Hypothesized value mu0 = {MU0:,.0f} USD (benchmark / target)")
log(f"c) H0: mu = {MU0:,.0f}     H1: mu != {MU0:,.0f}")
log(f"d) Significance level alpha = {ALPHA}")

t0 = (x_bar - MU0) / se
p_value = 2 * stats.t.sf(abs(t0), n - 1)
ci = stats.t.interval(1 - ALPHA, n - 1, loc=x_bar, scale=se)
assert np.isclose(t0, stats.ttest_1samp(x, MU0).statistic)

log(f"e) Sample size (n): {n}")
log(f"   Sample mean: ${x_bar:,.2f}")
log(f"   Sample standard deviation: ${s:,.2f}")
log(f"   Standard error: ${se:,.2f}")
log(f"   t0 = (x-bar - mu0) / (s / sqrt(n)) = {t0:.4f}, df = {n - 1}")
log(f"   P-value: {fmt_p(p_value)}")
log(f"   95% CI for mu: [${ci[0]:,.2f}, ${ci[1]:,.2f}]")
log(f"f) Decision: {decision(p_value)}")
if p_value < ALPHA:
    log(f"   Conclusion: at alpha = {ALPHA} there is sufficient evidence that the mean "
        f"sale price differs from {MU0:,.0f} USD (sample mean {x_bar:,.0f} USD).")
else:
    log(f"   Conclusion: there is not enough evidence that the mean sale price differs "
        f"from {MU0:,.0f} USD.")

summary.append(dict(test="one-sample t (price)", statistic=t0,
                    p_value=p_value, decision=decision(p_value)))
save_table(pd.DataFrame({
    "Metric": ["mu0", "n", "Sample mean", "Sample SD", "Standard error",
               "t0", "df", "P-value", "95% CI Lower", "95% CI Upper"],
    "Value": [MU0, n, x_bar, s, se, t0, n - 1, p_value, ci[0], ci[1]]
}), "A_one_sample_ttest.csv")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(x, bins=80, color="steelblue", alpha=0.8)
ax.axvline(MU0, color="red", ls="--", label=f"H0: {MU0:,.0f}")
ax.axvline(x_bar, color="black", label=f"x-bar = {x_bar:,.0f}")
ax.set_xlabel("price (USD)")
ax.set_ylabel("count")
ax.legend()
ax.set_title("A. Price distribution vs hypothesized mean")
save_plot("A_price_hist.png")

log("\nOne-sample test results saved successfully.")

# ==========================================
# 5. 3.2.2 TWO-SAMPLE INFERENCE (g, h, i, k)
# ==========================================

banner("3.2.2 TWO-SAMPLE INFERENCE (price: waterfront vs non-waterfront)")

grp1 = df.loc[df["waterfront"] == 1, "price"].dropna()
grp2 = df.loc[df["waterfront"] == 0, "price"].dropna()
n1, n2 = len(grp1), len(grp2)
m1, m2 = grp1.mean(), grp2.mean()
v1, v2 = grp1.var(ddof=1), grp2.var(ddof=1)
diff = m1 - m2
z_crit = stats.norm.ppf(1 - ALPHA / 2)

log(f"g) H0: mu1 - mu2 = {DELTA0:,.0f}     H1: mu1 - mu2 != {DELTA0:,.0f}")
log("   (group 1 = waterfront, group 2 = non-waterfront)")
log(f"   Group 1: n1 = {n1}, mean = ${m1:,.2f}, SD = ${np.sqrt(v1):,.2f}")
log(f"   Group 2: n2 = {n2}, mean = ${m2:,.2f}, SD = ${np.sqrt(v2):,.2f}")
log(f"   Observed difference (x1-bar - x2-bar): ${diff:,.2f}")

# --- h) z-test (large samples: s^2 replaces sigma^2) ---
se_diff = np.sqrt(v1 / n1 + v2 / n2)
z0 = (diff - DELTA0) / se_diff
p_z = 2 * stats.norm.sf(abs(z0))
ci_z = (diff - z_crit * se_diff, diff + z_crit * se_diff)

subsection("h) z-test (large samples, s^2 used in place of sigma^2)")
log(f"   Z0 = (x1-bar - x2-bar - Delta0) / sqrt(s1^2/n1 + s2^2/n2) = {z0:.4f}")
log(f"   P-value: {fmt_p(p_z)}  ->  {decision(p_z)}")
log(f"   95% CI for mu1 - mu2: [${ci_z[0]:,.2f}, ${ci_z[1]:,.2f}]")

# --- i) variances unknown: pooled vs Welch ---
levene_b = stats.levene(grp1, grp2, center="median")

subsection("i) Variances unknown: pooled t-test vs Welch t-test")
log(f"   Levene test (median-centred): W = {levene_b.statistic:.3f}, "
    f"p = {fmt_p(levene_b.pvalue)};  s1^2 / s2^2 = {v1 / v2:.2f}")

sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
se_pooled = np.sqrt(sp2 * (1 / n1 + 1 / n2))
df_pooled = n1 + n2 - 2
t_pooled = (diff - DELTA0) / se_pooled
p_pooled = 2 * stats.t.sf(abs(t_pooled), df_pooled)
tc_pooled = stats.t.ppf(1 - ALPHA / 2, df_pooled)
ci_pooled = (diff - tc_pooled * se_pooled, diff + tc_pooled * se_pooled)

df_welch = (v1 / n1 + v2 / n2) ** 2 / (
    (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
t_welch = (diff - DELTA0) / se_diff
p_welch = 2 * stats.t.sf(abs(t_welch), df_welch)
tc_welch = stats.t.ppf(1 - ALPHA / 2, df_welch)
ci_welch = (diff - tc_welch * se_diff, diff + tc_welch * se_diff)
assert np.isclose(t_welch, stats.ttest_ind(grp1, grp2, equal_var=False).statistic)

log(f"   Pooled t-test: t0 = {t_pooled:.4f}, df = {df_pooled}, p = {fmt_p(p_pooled)}; "
    f"95% CI [${ci_pooled[0]:,.2f}, ${ci_pooled[1]:,.2f}]")
log(f"   Welch t-test : t0 = {t_welch:.4f}, df = {df_welch:.1f}, p = {fmt_p(p_welch)}; "
    f"95% CI [${ci_welch[0]:,.2f}, ${ci_welch[1]:,.2f}]")

use_welch = levene_b.pvalue < ALPHA
chosen = "Welch" if use_welch else "Pooled"
t_chosen, p_chosen, ci_chosen = (
    (t_welch, p_welch, ci_welch) if use_welch else (t_pooled, p_pooled, ci_pooled))
log(f"   Choice: {chosen} t-test, because Levene's test "
    f"{'rejects' if use_welch else 'does not reject'} equal variances at alpha = {ALPHA} "
    f"(and n1 = {n1} << n2 = {n2}).")

# --- k) interpretation and consistency ---
inside = ci_chosen[0] <= DELTA0 <= ci_chosen[1]
consistent = (inside == (p_chosen >= ALPHA))

subsection("k) Interpretation and consistency")
log(f"   {chosen} 95% CI = [${ci_chosen[0]:,.2f}, ${ci_chosen[1]:,.2f}]; "
    f"Delta0 = {DELTA0:,.0f} is {'INSIDE' if inside else 'OUTSIDE'} the CI "
    f"-> {'fail to reject' if inside else 'reject'} H0")
log(f"   Test decision: {decision(p_chosen)}  =>  "
    f"{'consistent' if consistent else 'INCONSISTENT'} with the confidence interval.")

# The conclusion handles both directions of the difference.
if p_chosen < ALPHA and diff > 0:
    conclusion_b = (f"at alpha = {ALPHA}, waterfront houses sell for significantly more on "
                    f"average (estimated difference {diff:,.0f} USD, "
                    f"95% CI {ci_chosen[0]:,.0f} to {ci_chosen[1]:,.0f}).")
elif p_chosen < ALPHA and diff < 0:
    conclusion_b = (f"at alpha = {ALPHA}, waterfront houses sell for significantly less on "
                    f"average (estimated difference {diff:,.0f} USD, "
                    f"95% CI {ci_chosen[0]:,.0f} to {ci_chosen[1]:,.0f}).")
else:
    conclusion_b = (f"at alpha = {ALPHA} there is no significant difference in mean price "
                    f"between the two groups (estimated difference {diff:,.0f} USD).")
log("   Conclusion: " + conclusion_b)

summary.append(dict(test="two-sample z (waterfront)", statistic=z0,
                    p_value=p_z, decision=decision(p_z)))
summary.append(dict(test=f"two-sample {chosen} t (waterfront)", statistic=t_chosen,
                    p_value=p_chosen, decision=decision(p_chosen)))

save_table(pd.DataFrame({
    "method": ["z", "pooled t", "Welch t"],
    "statistic": [z0, t_pooled, t_welch],
    "df": [np.nan, df_pooled, df_welch],
    "p_value": [p_z, p_pooled, p_welch],
    "ci_low": [ci_z[0], ci_pooled[0], ci_welch[0]],
    "ci_high": [ci_z[1], ci_pooled[1], ci_welch[1]]
}), "B_two_sample_results.csv")

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([grp1, grp2], showfliers=False)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"waterfront\n(n={n1})", f"non-waterfront\n(n={n2})"])
ax.set_ylabel("price (USD)")
ax.set_title("B. Price by waterfront")
save_plot("B_price_by_waterfront_boxplot.png")

log("\nTwo-sample results saved successfully.")

# ==========================================
# 6. 3.2.3 TYPE II ERROR, POWER, SAMPLE SIZE (j)
# ==========================================

banner("3.2.3 TYPE II ERROR, POWER AND SAMPLE SIZE")

log(f"j) Practically meaningful difference Delta = {DELTA_PRACTICAL:,.0f} USD, "
    f"alpha = {ALPHA}")
log("   Assumptions: two-sided z-approximation, sample variances s1^2, s2^2 used as")
log("   the true variances, and the current group sizes n1, n2 for the power calculation.")

ncp = DELTA_PRACTICAL / se_diff
beta = stats.norm.cdf(z_crit - ncp) - stats.norm.cdf(-z_crit - ncp)
power = 1 - beta

log(f"   SE of the difference = ${se_diff:,.2f};  Delta / SE = {ncp:.3f}")
log("   beta = P(Type II error) = Phi(z_a/2 - Delta/SE) - Phi(-z_a/2 - Delta/SE)")
log(f"   beta (Type II error probability): {beta:.4f}")
log(f"   Power = 1 - beta: {power:.4f}")

z_beta = stats.norm.ppf(POWER_TARGET)
n_required = int(np.ceil((z_crit + z_beta) ** 2 * (v1 + v2) / DELTA_PRACTICAL ** 2))

subsection(f"   Required sample size for power {POWER_TARGET:.2f} (equal n per group)")
log("   n = (z_a/2 + z_b)^2 * (s1^2 + s2^2) / Delta^2")
log(f"     = ({z_crit:.3f} + {z_beta:.3f})^2 * ({v1:,.0f} + {v2:,.0f}) "
    f"/ {DELTA_PRACTICAL:,.0f}^2")
log(f"   Required n per group: {n_required}")
log("   (This formula ignores the far rejection tail, which is negligible here.)")

if power >= POWER_TARGET:
    log(f"   Interpretation: the current samples already reach the target power "
        f"of {POWER_TARGET:.0%}.")
else:
    log(f"   Interpretation: with the current samples the test detects a "
        f"{DELTA_PRACTICAL:,.0f} USD difference with probability {power:.0%} only; "
        f"about {n_required} houses per group are needed to reach {POWER_TARGET:.0%}.")

summary.append(dict(test=f"power (waterfront, Delta={DELTA_PRACTICAL})", statistic=power,
                    p_value=np.nan, decision=f"beta={beta:.4f}; n/group={n_required}"))
save_table(pd.DataFrame({
    "Metric": ["Delta (USD)", "alpha", "SE of difference", "Delta / SE",
               "beta (Type II error)", "Power", "Target power",
               "Required n per group"],
    "Value": [DELTA_PRACTICAL, ALPHA, se_diff, ncp, beta, power, POWER_TARGET,
              n_required]
}), "C_power_results.csv")

n_grid = np.arange(10, max(n_required * 2, 200), 5)
power_curve = [
    stats.norm.cdf(DELTA_PRACTICAL / np.sqrt((v1 + v2) / m) - z_crit)
    + stats.norm.cdf(-DELTA_PRACTICAL / np.sqrt((v1 + v2) / m) - z_crit)
    for m in n_grid
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n_grid, power_curve)
ax.axhline(POWER_TARGET, color="red", ls="--", label=f"target {POWER_TARGET}")
ax.axvline(n_required, color="gray", ls=":", label=f"n = {n_required}")
ax.set_xlabel("n per group")
ax.set_ylabel("power")
ax.legend()
ax.set_title(f"C. Power curve (Delta = {DELTA_PRACTICAL:,.0f})")
save_plot("C_power_curve.png")

log("\nPower analysis results saved successfully.")

# ==========================================
# 7. 3.2.4 PAIRED t-TEST (l)
# ==========================================

banner("3.2.4 PAIRED t-TEST (sqft_living vs sqft_living15)")

d = (df["sqft_living"] - df["sqft_living15"]).dropna()
n_d, d_bar, s_d = len(d), d.mean(), d.std(ddof=1)
se_d = s_d / np.sqrt(n_d)
t_d = d_bar / se_d
p_d = 2 * stats.t.sf(abs(t_d), n_d - 1)
ci_d = stats.t.interval(1 - ALPHA, n_d - 1, loc=d_bar, scale=se_d)
assert np.isclose(t_d, stats.ttest_1samp(d, 0).statistic)

log("l) Pairs: each house vs the average of its 15 nearest neighbours (dependent samples)")
log("   d_i = sqft_living_i - sqft_living15_i;  one-sample t-test on d_i, df = n - 1")
log("   H0: mu_d = 0     H1: mu_d != 0")
log(f"   Sample size (n): {n_d}")
log(f"   Mean difference (d-bar): {d_bar:,.2f} sqft")
log(f"   SD of differences (s_d): {s_d:,.2f} sqft")
log(f"   t0 = d-bar / (s_d / sqrt(n)) = {t_d:.4f}, df = {n_d - 1}")
log(f"   P-value: {fmt_p(p_d)}")
log(f"   95% CI for mu_d: [{ci_d[0]:,.2f}, {ci_d[1]:,.2f}]")
log(f"   Decision: {decision(p_d)}")
if p_d < ALPHA:
    log(f"   Conclusion: a house's living area differs significantly from its neighbours' "
        f"average (on average {abs(d_bar):,.0f} sqft {'larger' if d_bar > 0 else 'smaller'}).")
else:
    log("   Conclusion: no significant difference between a house's living area and "
        "its neighbours' average.")
log("   Note: houses within the same area share neighbours, so the d_i may be spatially")
log("   correlated; the independence of the pairs is assumed, not verified.")

summary.append(dict(test="paired t (sqft_living - sqft_living15)", statistic=t_d,
                    p_value=p_d, decision=decision(p_d)))
save_table(pd.DataFrame({
    "Metric": ["n", "Mean difference", "SD of differences", "t0", "df", "P-value",
               "95% CI Lower", "95% CI Upper"],
    "Value": [n_d, d_bar, s_d, t_d, n_d - 1, p_d, ci_d[0], ci_d[1]]
}), "D_paired_ttest.csv")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(d, bins=80, color="seagreen", alpha=0.8)
ax.axvline(0, color="red", ls="--")
ax.axvline(d_bar, color="black")
ax.set_xlabel("d = sqft_living - sqft_living15")
ax.set_ylabel("count")
ax.set_title("D. Differences d = sqft_living - sqft_living15")
save_plot("D_paired_differences.png")

log("\nPaired t-test results saved successfully.")

# ==========================================
# 8. 3.2.5 ONE-WAY ANOVA (m, n, o)
# ==========================================

banner("3.2.5 ONE-WAY ANOVA (price by grade group)")

dat = df[["price", "grade_group"]].dropna().copy()
dat["grade_group"] = dat["grade_group"].astype(str)
levels = sorted(dat["grade_group"].unique())
groups = [dat.loc[dat["grade_group"] == lv, "price"].values for lv in levels]
k, N = len(levels), len(dat)

log(f"m) Response: price;  factor: grade group (k = {k} groups, N = {N})")
log("   H0: mu_1 = mu_2 = ... = mu_k     H1: at least two means differ")
log(f"   Significance level: {ALPHA}")

desc = dat.groupby("grade_group")["price"].agg(["count", "mean", "std", "median"])
subsection("Group descriptive statistics:")
log(desc.round(2).to_string())
save_table(desc, "E_group_descriptives.csv", index=True)

# --- n) classical ANOVA table ---
grand_mean = dat["price"].mean()
ss_tr = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_e = sum(((g - g.mean()) ** 2).sum() for g in groups)
ss_t = ((dat["price"] - grand_mean) ** 2).sum()
df_tr, df_e = k - 1, N - k
ms_tr, mse = ss_tr / df_tr, ss_e / df_e
F0 = ms_tr / mse
p_F = stats.f.sf(F0, df_tr, df_e)
assert np.isclose(F0, stats.f_oneway(*groups).statistic)

anova_table = pd.DataFrame({
    "Source": ["Treatment", "Error", "Total"],
    "SS": [ss_tr, ss_e, ss_t],
    "df": [df_tr, df_e, N - 1],
    "MS": [ms_tr, mse, np.nan],
    "F0": [F0, np.nan, np.nan],
    "p_value": [p_F, np.nan, np.nan]
})

subsection("n) Classical one-way ANOVA table (main result reported)")
log(anova_table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
log(f"   F0 = MS_Treatment / MSE = {F0:.4f}, p-value = {fmt_p(p_F)}, "
    f"eta^2 = SS_Tr / SS_T = {ss_tr / ss_t:.4f}")
log(f"   Decision: {decision(p_F)}")
save_table(anova_table, "E_anova_table.csv")
summary.append(dict(test="one-way ANOVA (price ~ grade_group)", statistic=F0,
                    p_value=p_F, decision=decision(p_F)))

# --- o) assumption checks ---
fitted = dat.groupby("grade_group")["price"].transform("mean").values
resid = dat["price"].values - fitted
levene_a = stats.levene(*groups, center="median")
bartlett_a = stats.bartlett(*groups)
resid_skew = stats.skew(resid)
resid_kurt = stats.kurtosis(resid)
variances_unequal = levene_a.pvalue < ALPHA

subsection("o) Assumption checks")
log("   Independence: each row is a separate sale, but the dataset contains repeated")
if n_dup_ids is not None:
    log(f"   property IDs ({n_dup_ids} IDs sold more than once, {n_dup_rows} extra rows).")
else:
    log("   property IDs (see Data Cleaning).")
log("   Repeated sales of the same house are not strictly independent, so independence")
log("   is ASSUMED (approximately) and is not guaranteed by the data.")
log(f"   Normality of residuals: see Q-Q plot (skewness = {resid_skew:.2f}, "
    f"excess kurtosis = {resid_kurt:.2f}); with large group sizes the F-test is")
log("   fairly robust to moderate non-normality, but the price is strongly right-skewed.")
log(f"   Equal variances: Levene F = {levene_a.statistic:.3f}, p = {fmt_p(levene_a.pvalue)};  "
    f"Bartlett chi2 = {bartlett_a.statistic:.3f}, p = {fmt_p(bartlett_a.pvalue)}")
if variances_unequal:
    log("   -> Equal variances are REJECTED. The classical ANOVA / Tukey HSD results are")
    log("      therefore cross-checked with Welch ANOVA / Games-Howell below.")
else:
    log("   -> Equal variances are not rejected; classical ANOVA / Tukey HSD are appropriate.")

save_table(pd.DataFrame({
    "Check": ["Levene F", "Levene p", "Bartlett chi2", "Bartlett p",
              "Residual skewness", "Residual excess kurtosis",
              "Repeated property IDs", "Extra rows from repeated IDs"],
    "Value": [levene_a.statistic, levene_a.pvalue, bartlett_a.statistic,
              bartlett_a.pvalue, resid_skew, resid_kurt,
              n_dup_ids if n_dup_ids is not None else np.nan,
              n_dup_rows if n_dup_rows is not None else np.nan]
}), "E_assumption_tests.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sm.qqplot(resid, line="s", ax=axes[0], markersize=2)
axes[0].set_title("Q-Q plot of residuals")
axes[1].scatter(fitted, resid, s=3, alpha=0.3)
axes[1].axhline(0, color="red")
axes[1].set_xlabel("fitted (group mean)")
axes[1].set_ylabel("residual")
axes[1].set_title("Residuals vs fitted")
axes[2].boxplot(groups, showfliers=False)
axes[2].set_xticks(range(1, k + 1))
axes[2].set_xticklabels([lv.split(" ")[0] for lv in levels])
axes[2].set_xlabel("grade group")
axes[2].set_title("Price by grade group")
save_plot("E_anova_assumptions.png")

# --- Post-hoc: Tukey HSD and Fisher's LSD (run only if classical ANOVA rejects H0) ---
tukey_df = None
if p_F < ALPHA:
    subsection("Classical ANOVA rejects H0 -> post-hoc multiple comparisons")

    tukey = pairwise_tukeyhsd(dat["price"], dat["grade_group"], alpha=ALPHA)
    tukey_df = pd.DataFrame(tukey._results_table.data[1:],
                            columns=tukey._results_table.data[0])
    tukey_df = tukey_df.astype({"meandiff": float, "p-adj": float, "lower": float,
                                "upper": float, "reject": bool})
    log("Tukey HSD (assumes equal variances):")
    log(tukey_df.to_string(index=False))
    save_table(tukey_df, "E_tukey.csv")

    lsd_rows = []
    t_crit = stats.t.ppf(1 - ALPHA / 2, df_e)
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = len(groups[i]), len(groups[j])
            mean_diff = groups[i].mean() - groups[j].mean()
            se_ij = np.sqrt(mse * (1 / ni + 1 / nj))
            lsd_rows.append(dict(
                group_i=levels[i], group_j=levels[j], diff=mean_diff,
                LSD=t_crit * se_ij,
                p_value=2 * stats.t.sf(abs(mean_diff / se_ij), df_e),
                significant=abs(mean_diff) > t_crit * se_ij))
    lsd_df = pd.DataFrame(lsd_rows)
    log("\nFisher's LSD (t with N-k df, pooled MSE; NOT adjusted for multiple comparisons):")
    log(lsd_df.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    save_table(lsd_df, "E_fisher_lsd.csv")

    order = dat.groupby("grade_group")["price"].mean().sort_values()
    log("\nInterpretation: mean price differs across grade groups; ordered by mean: "
        + " < ".join(f"{g} ({v:,.0f})" for g, v in order.items())
        + f". Tukey finds {int(tukey_df['reject'].sum())} of {len(tukey_df)} "
        "pairs significantly different.")
else:
    log("\nH0 not rejected -> no post-hoc comparison needed.")

# ==========================================
# 9. 3.2.5b ROBUSTNESS CHECK: WELCH ANOVA + GAMES-HOWELL
# ==========================================

banner("3.2.5b ROBUSTNESS CHECK: WELCH ANOVA + GAMES-HOWELL")

log("Role of these methods: the classical ANOVA / Tukey HSD above are the main results")
log("reported. Welch ANOVA and Games-Howell do NOT assume equal variances and serve as a")
log("robustness check, because the equal-variance assumption is "
    + ("rejected." if variances_unequal else "not rejected."))

n_i = np.array([len(g) for g in groups])
m_i = np.array([g.mean() for g in groups])
v_i = np.array([g.var(ddof=1) for g in groups])
w_i = n_i / v_i
w_sum = w_i.sum()
mean_w = (w_i * m_i).sum() / w_sum
lam = (((1 - w_i / w_sum) ** 2) / (n_i - 1)).sum()
F_welch = (((w_i * (m_i - mean_w) ** 2).sum() / (k - 1))
           / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam))
df1_welch, df2_welch = k - 1, (k ** 2 - 1) / (3 * lam)
p_F_welch = stats.f.sf(F_welch, df1_welch, df2_welch)

subsection("Welch's ANOVA (does not assume equal variances)")
log(f"   F = {F_welch:.4f}, df1 = {df1_welch}, df2 = {df2_welch:.1f}, "
    f"p-value = {fmt_p(p_F_welch)}")
log(f"   Decision: {decision(p_F_welch)}")
log(f"   Classical ANOVA: F = {F0:.4f}, p-value = {fmt_p(p_F)} ({decision(p_F)})")
log("   Same decision as classical ANOVA: "
    + str(decision(p_F_welch) == decision(p_F)))

summary.append(dict(test="Welch ANOVA (price ~ grade_group)", statistic=F_welch,
                    p_value=p_F_welch, decision=decision(p_F_welch)))
save_table(pd.DataFrame({
    "Method": ["Classical ANOVA", "Welch ANOVA"],
    "F": [F0, F_welch],
    "df1": [df_tr, df1_welch],
    "df2": [df_e, df2_welch],
    "p_value": [p_F, p_F_welch],
    "Decision": [decision(p_F), decision(p_F_welch)]
}), "E_welch_anova.csv")

gh_rows = []
for i in range(k):
    for j in range(i + 1, k):
        mean_diff = m_i[i] - m_i[j]
        se_ij = np.sqrt(v_i[i] / n_i[i] + v_i[j] / n_i[j])
        df_ij = (v_i[i] / n_i[i] + v_i[j] / n_i[j]) ** 2 / (
            (v_i[i] / n_i[i]) ** 2 / (n_i[i] - 1) + (v_i[j] / n_i[j]) ** 2 / (n_i[j] - 1))
        q = abs(mean_diff) * np.sqrt(2) / se_ij
        p_gh = stats.studentized_range.sf(q, k, df_ij)
        half_width = stats.studentized_range.ppf(1 - ALPHA, k, df_ij) * se_ij / np.sqrt(2)
        gh_rows.append(dict(group_i=levels[i], group_j=levels[j], diff=mean_diff,
                            df=df_ij, p_adj=p_gh, ci_low=mean_diff - half_width,
                            ci_high=mean_diff + half_width, reject=p_gh < ALPHA))
gh_df = pd.DataFrame(gh_rows)

subsection("Games-Howell post-hoc (does not assume equal variances; diff = mean_i - mean_j)")
log(gh_df.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
save_table(gh_df, "E_games_howell.csv")

# Side-by-side comparison with Tukey HSD (only if Tukey was run)
if tukey_df is not None:
    cmp_df = tukey_df[["group1", "group2", "meandiff", "reject"]].merge(
        gh_df[["group_i", "group_j", "diff", "reject"]],
        left_on=["group1", "group2"], right_on=["group_i", "group_j"],
        suffixes=("_tukey", "_gh"))
    # Tukey meandiff = mean(group2) - mean(group1); Games-Howell diff = mean_i - mean_j.
    # The sign is flipped so both columns are expressed as mean(group2) - mean(group1).
    cmp_df = pd.DataFrame({
        "group1": cmp_df["group1"], "group2": cmp_df["group2"],
        "Tukey_diff": cmp_df["meandiff"], "Tukey_reject": cmp_df["reject_tukey"],
        "GH_diff": -cmp_df["diff"], "GH_reject": cmp_df["reject_gh"]})
    subsection("Tukey HSD vs Games-Howell (diff = mean(group2) - mean(group1)):")
    log(cmp_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    log("   Consistent conclusion for all pairs: "
        + str(bool((cmp_df["Tukey_reject"] == cmp_df["GH_reject"]).all())))
    save_table(cmp_df, "E_tukey_vs_games_howell.csv")
else:
    log("\nTukey HSD was not run (classical ANOVA did not reject H0), so no comparison.")

log("\nANOVA results saved successfully.")

# ==========================================
# 10. SAVE SUMMARY AND FINISH
# ==========================================

save_table(pd.DataFrame(summary), "summary_tests.csv")

banner("TASK 2 COMPLETED")
log(f"Tables and log saved in: {RESULTS_DIR}")
log(f"Plots saved in: {PLOTS_DIR}")
_logfile.close()