import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đường dẫn lưu output
output_dir = 'Data/Task_3_Regression'
os.makedirs(output_dir, exist_ok=True)

# 2. Đọc dataset
data_path = 'Data/kc_house_data_cleaned.csv'
if not os.path.exists(data_path):
    data_path = 'Data/kc_house_data.csv'  # Dùng file gốc nếu chưa có file cleaned

if not os.path.exists(data_path):
    print(f"[LỖI] Chưa thấy file .csv tại thư mục Data! Bạn nhớ chép file kc_house_data.csv hoặc kc_house_data_cleaned.csv vào thư mục Data/ nhé.")
else:
    df = pd.read_csv(data_path)

    # 3. Chọn các biến cho bài toán Hồi quy
    candidate_features = ['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'yr_built', 'waterfront']
    features = [col for col in candidate_features if col in df.columns]

    X = df[features]
    y = df['price']

    # Thêm hằng số Beta_0
    X_with_const = sm.add_constant(X)

    # 4. Huấn luyện mô hình OLS
    model = sm.OLS(y, X_with_const).fit()

    # In kết quả & lưu file txt
    summary_text = model.summary().as_text()
    print(summary_text)

    with open(f'{output_dir}/regression_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)

    # 5. Vẽ đồ thị phần dư (Residual Plot)
    df['residuals'] = model.resid
    df['fitted'] = model.fittedvalues

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='fitted', y='residuals', data=df, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Price')
    plt.ylabel('Residuals')
    plt.savefig(f'{output_dir}/residuals_plot.png')
    plt.close()

    # 6. Vẽ đồ thị Q-Q Plot
    plt.figure(figsize=(8, 6))
    sm.qqplot(model.resid, line='45', fit=True)
    plt.title('Q-Q Plot of Residuals')
    plt.savefig(f'{output_dir}/qq_plot.png')
    plt.close()

    print("\n[THÀNH CÔNG] Đã chạy xong Task 3! Kết quả nằm tại thư mục Data/Task_3_Regression/")