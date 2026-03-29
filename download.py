import os
import subprocess
from huggingface_hub import hf_hub_download

# Danh sách zip file cần tải
files = {
    "ImageNet_1K.zip": "images_zip/ImageNet_1K.zip",
    "N24News.zip": "images_zip/N24News.zip",
    "HatefulMemes.zip": "images_zip/HatefulMemes.zip",
    "VOC2007.zip": "images_zip/VOC2007.zip",
    "SUN397.zip": "images_zip/SUN397.zip",
}

dataset = "TIGER-Lab/MMEB-train"
save_dir = "./vlm2vec_train/MMEB-train/images"

os.makedirs(save_dir, exist_ok=True)

def fast_unzip(zip_path, output_dir):
    """
    Giải nén bằng subprocess (unzip -q) để chạy nhanh
    """
    print(f"📦 Unzipping {zip_path} ...")
    subprocess.run(["unzip", "-q", zip_path, "-d", output_dir], check=True)
    os.remove(zip_path)
    print(f"✔️ Done {zip_path}")

# Tải từng file bằng hf_hub_download
local_paths = []
for name, repo_path in files.items():
    print(f"⬇️ Downloading {name} ...")
    downloaded = hf_hub_download(
        repo_id=dataset,
        filename=repo_path,
        local_dir="./downloads",   # tải tạm
        repo_type="dataset"
    )
    local_paths.append(downloaded)

# Giải nén song song
processes = []
for zip_file in local_paths:
    p = subprocess.Popen(["unzip", "-q", zip_file, "-d", save_dir])
    processes.append((p, zip_file))

# Đợi tất cả hoàn thành
for p, zip_file in processes:
    p.wait()
    os.remove(zip_file)
    print(f"✔️ Unzipped & removed {zip_file}")

print("🎉 All done!")
