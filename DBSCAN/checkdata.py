import pandas as pd

df = pd.read_csv("D:/DBSCAN/dataset/processed_user_features.csv")

# Kiểm tra kiểu dữ liệu
print(df.dtypes)

# Kiểm tra giá trị null
print(df.isnull().sum())

# Kiểm tra thống kê dữ liệu
print(df.describe())

# Kiểm tra số lượng dòng trùng lặp
num_duplicates = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {num_duplicates}")

# Nếu có dòng trùng lặp, loại bỏ chúng
if num_duplicates > 0:
    df = df.drop_duplicates()
    print("Đã loại bỏ các dòng trùng lặp.")

print(f"Số cột dữ liệu: {df.shape[1]}")
