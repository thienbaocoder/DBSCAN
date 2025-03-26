import pandas as pd
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
file_path = "D:/DBSCAN/dataset/user_personalized_features.csv"
df = pd.read_csv(file_path)

# Loại bỏ cột không cần thiết
df.drop(columns=["User_ID", "Unnamed: 0"], inplace=True, errors='ignore')

# One-hot encoding các cột object
df = pd.get_dummies(df, columns=["Location", "Interests", "Product_Category_Preference"], dtype=int)

# Chuyển boolean thành int
df["Newsletter_Subscription"] = df["Newsletter_Subscription"].astype(int)

# One-hot encoding cột 'Gender'
df = pd.get_dummies(df, columns=["Gender"], dtype=int)

# Chuẩn hóa các cột số
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Lưu dữ liệu đã tiền xử lý
processed_path = "D:/DBSCAN/dataset/processed_user_features.csv"
df.to_csv(processed_path, index=False)

print("✅ Tiền xử lý hoàn tất! Dữ liệu đã được lưu tại:", processed_path)
