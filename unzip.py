import zipfile
import os

zip_path = "images.zip"
extract_dir = "Dataset"

print(f"[INFO] Đang chạy script tại thư mục: {os.getcwd()}")
print(f"[INFO] File zip cần giải nén: {os.path.abspath(zip_path)}")
print(f"[INFO] Thư mục giải nén: {os.path.abspath(extract_dir)}")

# Tạo thư mục đích nếu chưa có
os.makedirs(extract_dir, exist_ok=True)

# Giải nén với log từng file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    print(f"[INFO] Số lượng file trong zip: {len(file_list)}")
    for f in file_list:
        print(f"    -> Giải nén: {f}")
    zip_ref.extractall(extract_dir)

print("[INFO] Hoàn tất giải nén.")
