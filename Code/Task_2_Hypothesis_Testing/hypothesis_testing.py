"""
MT2013 - Task 2: Hypothesis Testing (King County House Sales)
=============================================================
Place at : Code/Task_2_Hypothesis_Testing/hypothesis_testing.py
Input    : Data/kc_house_data_cleaned.csv
Output   : Data/Task_2_Hypothesis_Testing/   (log .txt, tables .csv, figures/ .png)

One question per type of test:
  A. One-sample t-test (mean price)                      -> a) - f)
  B. Two-sample inference (price: waterfront vs not)     -> g) - i), k)
  C. Type II error, power, sample size                   -> j)
  D. Paired t-test (sqft_living vs sqft_living15)        -> l)
  E. One-way ANOVA (price by grade group) + post-hoc     -> m) - o)

Libs: pandas numpy scipy statsmodels matplotlib
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ----------------------------------------------------------------------------
# CONFIG (edit here)
# ----------------------------------------------------------------------------
ALPHA = 0.05
POWER_TARGET = 0.90

MU0 = 500_000            # A: hypothesized mean price (USD)
DELTA0 = 0.0             # B: H0: mu1 - mu2 = DELTA0
DELTA_PRACTICAL = 100_000  # C: practically meaningful difference (USD)

# ----------------------------------------------------------------------------
# PATHS + LOGGING
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parents[1] / "Data", Path.cwd() / "Data", Path.cwd().parent / "Data"]
DATA_DIR = next((p for p in CANDIDATES if (p / "kc_house_data_cleaned.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError("Cannot find Data/kc_house_data_cleaned.csv")
OUT = DATA_DIR / "Task_2_Hypothesis_Testing"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

_logfile = open(OUT / "task2_output_log.txt", "w", encoding="utf-8")


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    _logfile.write(s + "\n")
    _logfile.flush()


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def decision(p):
    return "Reject H0" if p < ALPHA else "Fail to reject H0"


def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "kc_house_data_cleaned.csv")
df["grade_group"] = pd.cut(df["grade"], bins=[0, 6, 7, 9, 20],
                           labels=["1 Low (<=6)", "2 Average (7)", "3 Good (8-9)", "4 High (>=10)"])
log(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns;  alpha = {ALPHA}")

summary = []   # rows for summary_tests.csv

# ============================================================================
# A. ONE-SAMPLE t-TEST  (a - f)
# ============================================================================
section("A. ONE-SAMPLE HYPOTHESIS TEST (mean price)")
x = df["price"].dropna()
n, xbar, s = len(x), x.mean(), x.std(ddof=1)
se = s / np.sqrt(n)

log("a) Research question: is the mean sale price different from the benchmark of "
    f"{MU0:,.0f} USD?")
log(f"b) Hypothesized value mu0 = {MU0:,.0f} USD (benchmark / target)")
log(f"c) H0: mu = {MU0:,.0f}     H1: mu != {MU0:,.0f}")
log(f"d) Significance level alpha = {ALPHA}")
t0 = (xbar - MU0) / se
p_val = 2 * stats.t.sf(abs(t0), n - 1)
ci = stats.t.interval(1 - ALPHA, n - 1, loc=xbar, scale=se)
assert np.isclose(t0, stats.ttest_1samp(x, MU0).statistic)
log(f"e) n = {n}, x-bar = {xbar:,.2f}, s = {s:,.2f}, SE = {se:,.2f}")
log(f"   t0 = (x-bar - mu0) / (s/sqrt(n)) = {t0:.4f}, df = {n - 1}, p-value = {fmt_p(p_val)}")
log(f"   95% CI for mu: ({ci[0]:,.2f}, {ci[1]:,.2f})")
log(f"f) Decision: {decision(p_val)}")
log("   Conclusion: " + (
    f"at alpha = {ALPHA} there is sufficient evidence that the mean sale price differs from "
    f"{MU0:,.0f} USD (sample mean {xbar:,.0f} USD)." if p_val < ALPHA else
    f"there is not enough evidence that the mean sale price differs from {MU0:,.0f} USD."))
summary.append(dict(test="one-sample t (price)", statistic=t0, p_value=p_val, decision=decision(p_val)))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(x, bins=80, color="steelblue", alpha=.8)
ax.axvline(MU0, color="red", ls="--", label=f"H0: {MU0:,.0f}")
ax.axvline(xbar, color="black", label=f"x-bar = {xbar:,.0f}")
ax.set_xlabel("price (USD)"); ax.set_ylabel("count"); ax.legend()
ax.set_title("A. Price distribution vs hypothesized mean")
savefig("A_price_hist.png")

# ============================================================================
# B. TWO-SAMPLE INFERENCE (g, h, i, k)
# ============================================================================
section("B. TWO-SAMPLE INFERENCE (price: waterfront vs non-waterfront)")
a = df.loc[df["waterfront"] == 1, "price"].dropna()
b = df.loc[df["waterfront"] == 0, "price"].dropna()
n1, n2 = len(a), len(b)
m1, m2 = a.mean(), b.mean()
v1, v2 = a.var(ddof=1), b.var(ddof=1)
diff = m1 - m2
zc = stats.norm.ppf(1 - ALPHA / 2)

log(f"g) H0: mu1 - mu2 = {DELTA0:,.0f}     H1: mu1 - mu2 != {DELTA0:,.0f}   "
    "(group 1 = waterfront, group 2 = non-waterfront)")
log(f"   group 1: n1 = {n1}, mean = {m1:,.2f}, sd = {np.sqrt(v1):,.2f}")
log(f"   group 2: n2 = {n2}, mean = {m2:,.2f}, sd = {np.sqrt(v2):,.2f}")
log(f"   observed difference = {diff:,.2f}")

# h) z-test (large samples: s^2 replaces sigma^2)
se_z = np.sqrt(v1 / n1 + v2 / n2)
z0 = (diff - DELTA0) / se_z
p_z = 2 * stats.norm.sf(abs(z0))
ci_z = (diff - zc * se_z, diff + zc * se_z)
log("\nh) z-test (large samples)")
log(f"   Z0 = (x1-bar - x2-bar - Delta0) / sqrt(s1^2/n1 + s2^2/n2) = {z0:.4f}, p-value = {fmt_p(p_z)} "
    f"-> {decision(p_z)}")
log(f"   95% CI = x1-bar - x2-bar +/- z(a/2)*SE = ({ci_z[0]:,.2f}, {ci_z[1]:,.2f})")

# i) variances unknown: pooled vs Welch
lev = stats.levene(a, b, center="median")
log("\ni) Variances unknown: pooled t-test vs Welch t-test")
log(f"   Levene test: W = {lev.statistic:.3f}, p = {fmt_p(lev.pvalue)};  s1^2/s2^2 = {v1 / v2:.2f}")

sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
se_p = np.sqrt(sp2 * (1 / n1 + 1 / n2))
df_p = n1 + n2 - 2
t_p = (diff - DELTA0) / se_p
p_p = 2 * stats.t.sf(abs(t_p), df_p)
tc_p = stats.t.ppf(1 - ALPHA / 2, df_p)
ci_p = (diff - tc_p * se_p, diff + tc_p * se_p)

df_w = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
t_w = (diff - DELTA0) / se_z
p_w = 2 * stats.t.sf(abs(t_w), df_w)
tc_w = stats.t.ppf(1 - ALPHA / 2, df_w)
ci_w = (diff - tc_w * se_z, diff + tc_w * se_z)
assert np.isclose(t_w, stats.ttest_ind(a, b, equal_var=False).statistic)

log(f"   Pooled t-test: t0 = {t_p:.4f}, df = {df_p}, p = {fmt_p(p_p)}; "
    f"95% CI ({ci_p[0]:,.2f}, {ci_p[1]:,.2f})")
log(f"   Welch t-test : t0 = {t_w:.4f}, df = {df_w:.1f}, p = {fmt_p(p_w)}; "
    f"95% CI ({ci_w[0]:,.2f}, {ci_w[1]:,.2f})")
use_welch = lev.pvalue < ALPHA
chosen = "Welch" if use_welch else "Pooled"
t_c, p_c, ci_c = (t_w, p_w, ci_w) if use_welch else (t_p, p_p, ci_p)
log(f"   Choice: {chosen} t-test, because Levene's test "
    f"{'rejects' if use_welch else 'does not reject'} equal variances at alpha = {ALPHA} "
    f"(and n1 = {n1} << n2 = {n2}).")

# k) consistency
inside = ci_c[0] <= DELTA0 <= ci_c[1]
log("\nk) Interpretation & consistency")
log(f"   {chosen} 95% CI = ({ci_c[0]:,.2f}, {ci_c[1]:,.2f}); Delta0 = {DELTA0:,.0f} is "
    f"{'INSIDE' if inside else 'OUTSIDE'} the CI -> {'fail to reject' if inside else 'reject'} H0; "
    f"test says: {decision(p_c)} => {'consistent' if inside == (p_c >= ALPHA) else 'INCONSISTENT'}.")
log("   Conclusion: " + (
    f"at alpha = {ALPHA}, waterfront houses sell for significantly more on average "
    f"(estimated difference {diff:,.0f} USD, 95% CI {ci_c[0]:,.0f} to {ci_c[1]:,.0f})."
    if p_c < ALPHA and diff > 0 else
    "there is no significant difference in mean price between the two groups."))
summary.append(dict(test="two-sample z (waterfront)", statistic=z0, p_value=p_z, decision=decision(p_z)))
summary.append(dict(test=f"two-sample {chosen} t (waterfront)", statistic=t_c, p_value=p_c,
                    decision=decision(p_c)))
pd.DataFrame({"method": ["z", "pooled t", "Welch t"], "statistic": [z0, t_p, t_w],
              "df": [np.nan, df_p, df_w], "p_value": [p_z, p_p, p_w],
              "ci_low": [ci_z[0], ci_p[0], ci_w[0]],
              "ci_high": [ci_z[1], ci_p[1], ci_w[1]]}).to_csv(OUT / "B_two_sample_results.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([a, b], showfliers=False)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"waterfront\n(n={n1})", f"non-waterfront\n(n={n2})"])
ax.set_ylabel("price (USD)"); ax.set_title("B. Price by waterfront")
savefig("B_price_by_waterfront_boxplot.png")

# ============================================================================
# C. TYPE II ERROR, POWER, SAMPLE SIZE (j)
# ============================================================================
section("C. TYPE II ERROR, POWER AND SAMPLE SIZE")
log(f"j) Practically meaningful difference Delta = {DELTA_PRACTICAL:,.0f} USD, alpha = {ALPHA}")
ncp = DELTA_PRACTICAL / se_z
beta = stats.norm.cdf(zc - ncp) - stats.norm.cdf(-zc - ncp)
log(f"   SE of the difference = {se_z:,.2f};  Delta/SE = {ncp:.3f}")
log(f"   beta = P(Type II error) = Phi(z_a/2 - Delta/SE) - Phi(-z_a/2 - Delta/SE) = {beta:.4f}")
log(f"   power = 1 - beta = {1 - beta:.4f}")
z_b = stats.norm.ppf(POWER_TARGET)
n_req = int(np.ceil((zc + z_b) ** 2 * (v1 + v2) / DELTA_PRACTICAL ** 2))
log(f"   Sample size for power {POWER_TARGET:.2f} (equal n per group):")
log(f"   n = (z_a/2 + z_b)^2 (s1^2 + s2^2) / Delta^2 = ({zc:.3f} + {z_b:.3f})^2 * "
    f"({v1:,.0f} + {v2:,.0f}) / {DELTA_PRACTICAL:,.0f}^2 = {n_req} per group")
log("   Interpretation: " + (
    f"the current samples already reach the target power {POWER_TARGET:.0%}." if 1 - beta >= POWER_TARGET else
    f"with the current samples the test detects a {DELTA_PRACTICAL:,.0f} USD difference with probability "
    f"{1 - beta:.0%} only; about {n_req} houses per group are needed to reach {POWER_TARGET:.0%}."))
summary.append(dict(test="power (waterfront, Delta=%d)" % DELTA_PRACTICAL, statistic=1 - beta,
                    p_value=np.nan, decision=f"beta={beta:.4f}; n/group={n_req}"))

ns = np.arange(10, max(n_req * 2, 200), 5)
pw = [stats.norm.cdf(DELTA_PRACTICAL / np.sqrt((v1 + v2) / k) - zc) +
      stats.norm.cdf(-DELTA_PRACTICAL / np.sqrt((v1 + v2) / k) - zc) for k in ns]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ns, pw)
ax.axhline(POWER_TARGET, color="red", ls="--", label=f"target {POWER_TARGET}")
ax.axvline(n_req, color="gray", ls=":", label=f"n = {n_req}")
ax.set_xlabel("n per group"); ax.set_ylabel("power"); ax.legend()
ax.set_title(f"C. Power curve (Delta = {DELTA_PRACTICAL:,.0f})")
savefig("C_power_curve.png")

# ============================================================================
# D. PAIRED t-TEST (l)
# ============================================================================
section("D. PAIRED t-TEST (sqft_living vs sqft_living15)")
d = (df["sqft_living"] - df["sqft_living15"]).dropna()
nd, dbar, sd = len(d), d.mean(), d.std(ddof=1)
t_d = dbar / (sd / np.sqrt(nd))
p_d = 2 * stats.t.sf(abs(t_d), nd - 1)
ci_d = stats.t.interval(1 - ALPHA, nd - 1, loc=dbar, scale=sd / np.sqrt(nd))
assert np.isclose(t_d, stats.ttest_1samp(d, 0).statistic)
log("l) Pairs: each house vs the average of its 15 nearest neighbours (dependent samples),")
log("   d_i = sqft_living_i - sqft_living15_i;  one-sample t-test on d_i with df = n - 1")
log("   H0: mu_d = 0     H1: mu_d != 0")
log(f"   n = {nd}, d-bar = {dbar:,.2f}, s_d = {sd:,.2f}")
log(f"   t0 = {t_d:.4f}, df = {nd - 1}, p-value = {fmt_p(p_d)}")
log(f"   95% CI for mu_d: ({ci_d[0]:,.2f}, {ci_d[1]:,.2f})")
log(f"   Decision: {decision(p_d)}")
log("   Conclusion: " + (
    f"a house's living area differs significantly from its neighbours' average "
    f"(on average {abs(dbar):,.0f} sqft {'larger' if dbar > 0 else 'smaller'})." if p_d < ALPHA else
    "no significant difference between a house's living area and its neighbours' average."))
summary.append(dict(test="paired t (sqft_living - sqft_living15)", statistic=t_d, p_value=p_d,
                    decision=decision(p_d)))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(d, bins=80, color="seagreen", alpha=.8)
ax.axvline(0, color="red", ls="--"); ax.axvline(dbar, color="black")
ax.set_title("D. Differences d = sqft_living - sqft_living15")
savefig("D_paired_differences.png")

# ============================================================================
# E. ONE-WAY ANOVA (m, n, o)
# ============================================================================
section("E. ONE-WAY ANOVA (price by grade group)")
dat = df[["price", "grade_group"]].dropna().copy()
dat["grade_group"] = dat["grade_group"].astype(str)
levels = sorted(dat["grade_group"].unique())
groups = [dat.loc[dat["grade_group"] == lv, "price"].values for lv in levels]
k, N = len(levels), len(dat)

log(f"m) Response: price;  factor: grade group (k = {k} groups, N = {N})")
log("   H0: mu_1 = mu_2 = ... = mu_k     H1: at least two means differ")
desc = dat.groupby("grade_group")["price"].agg(["count", "mean", "std", "median"])
log(desc.round(2).to_string())
desc.to_csv(OUT / "E_group_descriptives.csv")

grand = dat["price"].mean()
ss_tr = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_e = sum(((g - g.mean()) ** 2).sum() for g in groups)
ss_t = ((dat["price"] - grand) ** 2).sum()
df_tr, df_e = k - 1, N - k
ms_tr, mse = ss_tr / df_tr, ss_e / df_e
F0 = ms_tr / mse
pF = stats.f.sf(F0, df_tr, df_e)
assert np.isclose(F0, stats.f_oneway(*groups).statistic)

table = pd.DataFrame({"Source": ["Treatment", "Error", "Total"],
                      "SS": [ss_tr, ss_e, ss_t], "df": [df_tr, df_e, N - 1],
                      "MS": [ms_tr, mse, np.nan], "F0": [F0, np.nan, np.nan],
                      "p_value": [pF, np.nan, np.nan]})
log("\nn) ANOVA table")
log(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
log(f"   F0 = MS_Treatment / MSE = {F0:.4f}, p-value = {fmt_p(pF)}, eta^2 = SS_Tr/SS_T = {ss_tr / ss_t:.4f}")
log(f"   Decision: {decision(pF)}")
table.to_csv(OUT / "E_anova_table.csv", index=False)
summary.append(dict(test="one-way ANOVA (price ~ grade_group)", statistic=F0, p_value=pF,
                    decision=decision(pF)))

# o) assumptions
log("\no) Assumption checks")
log("   Independence: each row is a different sale (no repeated measures).")
fitted = dat.groupby("grade_group")["price"].transform("mean").values
resid = dat["price"].values - fitted
lev_a = stats.levene(*groups, center="median")
bar_a = stats.bartlett(*groups)
log(f"   Normality of residuals: see Q-Q plot (skewness = {stats.skew(resid):.2f}, "
    f"excess kurtosis = {stats.kurtosis(resid):.2f})")
log(f"   Equal variances: Levene F = {lev_a.statistic:.3f}, p = {fmt_p(lev_a.pvalue)};  "
    f"Bartlett chi2 = {bar_a.statistic:.3f}, p = {fmt_p(bar_a.pvalue)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sm.qqplot(resid, line="s", ax=axes[0], markersize=2)
axes[0].set_title("Q-Q plot of residuals")
axes[1].scatter(fitted, resid, s=3, alpha=.3)
axes[1].axhline(0, color="red")
axes[1].set_xlabel("fitted (group mean)"); axes[1].set_ylabel("residual")
axes[1].set_title("Residuals vs fitted")
axes[2].boxplot(groups, showfliers=False)
axes[2].set_xticks(range(1, k + 1))
axes[2].set_xticklabels([lv.split(" ")[0] for lv in levels])
axes[2].set_xlabel("grade group"); axes[2].set_title("Price by grade group")
savefig("E_anova_assumptions.png")

# post-hoc
if pF < ALPHA:
    log("\n   H0 rejected -> multiple comparisons")
    tk = pairwise_tukeyhsd(dat["price"], dat["grade_group"], alpha=ALPHA)
    tk_df = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
    log("   Tukey HSD:")
    log(tk_df.to_string(index=False))
    tk_df.to_csv(OUT / "E_tukey.csv", index=False)

    rows = []
    tcrit = stats.t.ppf(1 - ALPHA / 2, df_e)
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = len(groups[i]), len(groups[j])
            dif = groups[i].mean() - groups[j].mean()
            se_ij = np.sqrt(mse * (1 / ni + 1 / nj))
            rows.append(dict(group_i=levels[i], group_j=levels[j], diff=dif, LSD=tcrit * se_ij,
                             p_value=2 * stats.t.sf(abs(dif / se_ij), df_e),
                             significant=abs(dif) > tcrit * se_ij))
    lsd = pd.DataFrame(rows)
    log("\n   Fisher's LSD (t with N-k df, MSE pooled):")
    log(lsd.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    lsd.to_csv(OUT / "E_fisher_lsd.csv", index=False)

    order = dat.groupby("grade_group")["price"].mean().sort_values()
    log("\n   Interpretation: mean price differs across grade groups; ordered by mean: "
        + " < ".join(f"{g} ({v:,.0f})" for g, v in order.items())
        + f". Tukey finds {int(tk_df['reject'].sum())} of {len(tk_df)} pairs significantly different.")
else:
    log("   H0 not rejected -> no post-hoc comparison needed.")
# ============================================================================
# E2. ROBUSTNESS CHECK: WELCH ANOVA + GAMES-HOWELL (variances are unequal)
# ============================================================================
section("E2. ROBUSTNESS CHECK: WELCH ANOVA + GAMES-HOWELL")
n_i = np.array([len(g) for g in groups]); m_i = np.array([g.mean() for g in groups])
v_i = np.array([g.var(ddof=1) for g in groups])
w = n_i / v_i; W = w.sum(); xw = (w * m_i).sum() / W
lam = (((1 - w / W) ** 2) / (n_i - 1)).sum()
F_w = ((w * (m_i - xw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
df1_w, df2_w = k - 1, (k ** 2 - 1) / (3 * lam)
p_w = stats.f.sf(F_w, df1_w, df2_w)
log("Welch's ANOVA (does not assume equal variances)")
log(f"   F = {F_w:.4f}, df1 = {df1_w}, df2 = {df2_w:.1f}, p-value = {fmt_p(p_w)}")
log(f"   Classical ANOVA: F = {F0:.4f}, p-value = {fmt_p(pF)}")
summary.append(dict(test="Welch ANOVA (price ~ grade_group)", statistic=F_w, p_value=p_w,
                    decision=decision(p_w)))

rows = []
for i in range(k):
    for j in range(i + 1, k):
        dif = m_i[i] - m_i[j]
        se = np.sqrt(v_i[i] / n_i[i] + v_i[j] / n_i[j])
        dfw = (v_i[i] / n_i[i] + v_i[j] / n_i[j]) ** 2 / (
            (v_i[i] / n_i[i]) ** 2 / (n_i[i] - 1) + (v_i[j] / n_i[j]) ** 2 / (n_i[j] - 1))
        q = abs(dif) * np.sqrt(2) / se
        p_gh = stats.studentized_range.sf(q, k, dfw)
        half = stats.studentized_range.ppf(1 - ALPHA, k, dfw) * se / np.sqrt(2)
        rows.append(dict(group_i=levels[i], group_j=levels[j], diff=dif, df=dfw,
                         p_adj=p_gh, ci_low=dif - half, ci_high=dif + half,
                         reject=p_gh < ALPHA))
gh = pd.DataFrame(rows)
log("\nGames-Howell post-hoc (does not assume equal variances):")
log(gh.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
gh.to_csv(OUT / "E_games_howell.csv", index=False)

# side-by-side comparison with Tukey
cmp_ = tk_df[["group1", "group2", "meandiff", "reject"]].copy()
cmp_["GH_diff"] = -gh["diff"].values
cmp_["GH_reject"] = gh["reject"].values
cmp_.columns = ["group1", "group2", "Tukey_diff", "Tukey_reject", "GH_diff", "GH_reject"]
log("\nTukey HSD vs Games-Howell:")
log(cmp_.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
log("   Consistent conclusion: " + str((cmp_["Tukey_reject"] == cmp_["GH_reject"]).all()))
# ----------------------------------------------------------------------------
pd.DataFrame(summary).to_csv(OUT / "summary_tests.csv", index=False)
log(f"\nDone. Results saved in: {OUT}")
_logfile.close()