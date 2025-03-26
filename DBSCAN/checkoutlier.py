import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Đọc dữ liệu đã tiền xử lý
df = pd.read_csv("D:/DBSCAN/dataset/processed_user_features.csv")

# 1️⃣ Vẽ boxplot để xem outlier
plt.figure(figsize=(15, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot kiểm tra outlier")
plt.show()

# 2️⃣ Dùng IQR để phát hiện outlier
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Xác định outlier (dưới Q1 - 1.5*IQR hoặc trên Q3 + 1.5*IQR)
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
outlier_counts = outliers.sum()

# Hiển thị số lượng outlier trên mỗi cột
print("Số lượng outlier theo IQR:")
print(outlier_counts)

# 3️⃣ Dùng Z-score để kiểm tra outlier
z_scores = np.abs(zscore(df))
outliers_z = (z_scores > 3).sum(axis=0)  # Số lượng outlier với Z-score > 3

print("\nSố lượng outlier theo Z-score:")
print(outliers_z)
