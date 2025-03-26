import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy import stats
from kneed import KneeLocator

# Đọc dữ liệu đã tiền xử lý
file_path = "D:/DBSCAN/dataset/processed_user_features.csv"
df = pd.read_csv(file_path)

# Chuyển dữ liệu thành numpy array
data = df.values  

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Loại bỏ outlier bằng Z-score
z_scores = np.abs(stats.zscore(data_scaled))
data_no_outliers = data_scaled[(z_scores < 3).all(axis=1)]

print(f"Dữ liệu sau khi loại bỏ outlier: {data_no_outliers.shape}")

# Tìm eps tối ưu bằng K-Distance Graph
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(data_no_outliers)
distances, _ = neighbors_fit.kneighbors(data_no_outliers)
distances = np.sort(distances[:, -1])

# Xác định điểm gấp khúc (Elbow Point) để chọn eps
knee = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
eps_optimal = distances[knee.knee]
print(f"Giá trị eps tối ưu: {eps_optimal}")

# Vẽ K-Distance Graph
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.axvline(x=knee.knee, color='r', linestyle='--', label=f'Elbow Point (eps={eps_optimal:.2f})')
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.title("K-Distance Graph để chọn eps")
plt.legend()
plt.show()

# Áp dụng DBSCAN với eps tối ưu
min_samples = 5  
clustering = DBSCAN(eps=eps_optimal, min_samples=min_samples)
labels = clustering.fit_predict(data_no_outliers)

# Thống kê cụm
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

# Hiển thị số liệu
print(f"Số cụm tìm thấy: {n_clusters}")
print(f"Số điểm nhiễu: {n_noise}")
unique, counts = np.unique(labels, return_counts=True)
cluster_stats = dict(zip(unique, counts))
print("Phân bố điểm trong các cụm:", cluster_stats)

# Giảm số chiều bằng PCA (giữ lại 95% thông tin)
pca = PCA(n_components=0.95)
data_pca = pca.fit_transform(data_no_outliers)

# Vẽ kết quả phân cụm với PCA (loại bỏ điểm nhiễu)
mask = labels != -1  
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[mask, 0], data_pca[mask, 1], c=labels[mask], cmap='viridis', s=10)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Kết quả phân cụm DBSCAN (PCA giảm chiều, loại bỏ nhiễu)")
plt.colorbar(label='Cluster ID')
plt.show()
