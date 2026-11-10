import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo thư mục đầu ra
output_dir = 'Data/Task_3_Regression'
os.makedirs(output_dir, exist_ok=True)

# 1. Đọc dữ liệu
data_path = 'Data/kc_house_data.csv'
if not os.path.exists(data_path):
    data_path = 'Data/kc_house_data_cleaned.csv'

if not os.path.exists(data_path):
    print("[LỖI] Chưa thấy file .csv tại thư mục Data!")
    exit()

df = pd.read_csv(data_path)

summary_output = []
summary_output.append("==================================================")
summary_output.append("      BÁO CÁO KẾT QUẢ TASK 3: REGRESSION ANALYSIS  ")
summary_output.append("==================================================\n")

# --------------------------------------------------
# 1. PEARSON CORRELATION
# --------------------------------------------------
corr, p_val = stats.pearsonr(df['sqft_living'], df['price'])
summary_output.append("--- 1. PEARSON CORRELATION ---")
summary_output.append(f"Pearson Correlation (sqft_living vs price): {corr:.4f}")
summary_output.append(f"p-value: {p_val:.4e}\n")

# --------------------------------------------------
# 2. SIMPLE LINEAR REGRESSION MODEL
# --------------------------------------------------
X = sm.add_constant(df['sqft_living'])
y = df['price']
model_simple = sm.OLS(y, X).fit()

summary_output.append("--- 2. SIMPLE LINEAR REGRESSION SUMMARY ---")
summary_output.append(str(model_simple.summary()))
summary_output.append("\n")

# --------------------------------------------------
# 3. CONFIDENCE INTERVALS FOR COEFFICIENTS
# --------------------------------------------------
conf_int = model_simple.conf_int(alpha=0.05)
conf_int.columns = ['95% CI Lower', '95% CI Upper']
summary_output.append("--- 3. CONFIDENCE INTERVALS FOR COEFFICIENTS (95%) ---")
summary_output.append(str(conf_int))
summary_output.append("\n")

# --------------------------------------------------
# 4. PREDICTION AND PREDICTION INTERVALS
# --------------------------------------------------
# Dự báo cho một ngôi nhà có diện tích trung bình
mean_sqft = df['sqft_living'].mean()
new_data = pd.DataFrame({'const': [1.0], 'sqft_living': [mean_sqft]})

pred_res = model_simple.get_prediction(new_data)
pred_summary = pred_res.summary_frame(alpha=0.05)

summary_output.append(f"--- 4. PREDICTION AT SQFT_LIVING = {mean_sqft:.2f} ---")
summary_output.append(f"Predicted Value (mean): ${pred_summary['mean'].values[0]:,.2f}")
summary_output.append(f"95% CI for Mean Response: [${pred_summary['mean_ci_lower'].values[0]:,.2f}, ${pred_summary['mean_ci_upper'].values[0]:,.2f}]")
summary_output.append(f"95% Prediction Interval: [${pred_summary['obs_ci_lower'].values[0]:,.2f}, ${pred_summary['obs_ci_upper'].values[0]:,.2f}]\n")

# --------------------------------------------------
# 5. RESIDUAL ANALYSIS & MODEL ADEQUACY (CHARTS)
# --------------------------------------------------
df['residuals'] = model_simple.resid
df['fitted'] = model_simple.fittedvalues

# Residuals Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='fitted', y='residuals', data=df, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values (Simple Linear Regression)')
plt.xlabel('Fitted Price')
plt.ylabel('Residuals')
plt.savefig(f'{output_dir}/residuals_plot.png')
plt.close()

# Q-Q Plot
plt.figure(figsize=(8, 6))
sm.qqplot(model_simple.resid, line='45', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.savefig(f'{output_dir}/qq_plot.png')
plt.close()

# --------------------------------------------------
# 6. REGRESSION ON TRANSFORMED VARIABLES (Log Transform)
# --------------------------------------------------
df['log_price'] = np.log(df['price'])
df['log_sqft_living'] = np.log(df['sqft_living'])

X_log = sm.add_constant(df['log_sqft_living'])
y_log = df['log_price']
model_log = sm.OLS(y_log, X_log).fit()

summary_output.append("--- 6. LOG-LOG TRANSFORMED REGRESSION SUMMARY ---")
summary_output.append(str(model_log.summary()))

# Lưu tất cả vào file txt
full_summary_text = "\n".join(summary_output)
print(full_summary_text)

with open(f'{output_dir}/regression_summary.txt', 'w', encoding='utf-8') as f:
    f.write(full_summary_text)

print(f"\n[THÀNH CÔNG] Đã bổ sung 100% yêu cầu Task 3! Kết quả lưu tại {output_dir}/")