import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Đọc dữ liệu
df = pd.read_csv("D:/DBSCAN/dataset/processed_user_features.csv")

# Kiểm tra dữ liệu trùng lặp
duplicates = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {duplicates}")

# Kiểm tra phân phối dữ liệu với biểu đồ scatter plot
plt.figure(figsize=(8,6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.5)  # Vẽ 2 cột đầu tiên
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("Phân bố dữ liệu (2 biến đầu tiên)")
plt.show()

# Tìm epsilon (eps) bằng k-distance graph
k = 4  # Số hàng xóm tối thiểu
nbrs = NearestNeighbors(n_neighbors=k).fit(df)
distances, indices = nbrs.kneighbors(df)

# Sắp xếp khoảng cách theo thứ tự tăng dần
distances = np.sort(distances[:, k-1], axis=0)

# Vẽ biểu đồ để xác định điểm gấp khúc (elbow)
plt.figure(figsize=(8,6))
plt.plot(distances)
plt.xlabel("Sample Index")
plt.ylabel(f"Distance to {k}th Nearest Neighbor")
plt.title("K-Distance Graph để chọn eps")
plt.show()
