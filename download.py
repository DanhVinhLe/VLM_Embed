import os
import subprocess
from huggingface_hub import hf_hub_download
from multiprocessing import Pool

# List of zip files to download
FILE_LIST = [
    "A-OKVQA.zip", "CIRR.zip", "ChartQA.zip", "DocVQA.zip", "HatefulMemes.zip",
    "ImageNet_1K.zip", "InfographicsVQA.zip", "MSCOCO.zip", "MSCOCO_i2t.zip",
    "MSCOCO_t2i.zip", "N24News.zip", "NIGHTS.zip", "OK-VQA.zip", "SUN397.zip",
    "VOC2007.zip", "VisDial.zip", "Visual7W.zip", "VisualNews_i2t.zip",
    "VisualNews_t2i.zip", "WebQA.zip"
]

DATASET_REPO = "TIGER-Lab/MMEB-train"
SAVE_DIR = "./vlm2vec_train/MMEB-train/images"
TEMP_DOWNLOAD_DIR = "./downloads"

def download_file(file_name):
    """Downloads a single file from the Hugging Face Hub."""
    repo_path = f"images_zip/{file_name}"
    print(f"⬇️ Downloading {file_name} ...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=repo_path,
            local_dir=TEMP_DOWNLOAD_DIR,
            repo_type="dataset"
        )
        return downloaded_path
    except Exception as e:
        print(f"❌ Failed to download {file_name}. Error: {e}")
        return None

def unzip_file(zip_path):
    """Unzips a file and removes the zip archive."""
    if zip_path is None:
        return
    print(f"📦 Unzipping {zip_path} ...")
    try:
        subprocess.run(["unzip", "-q", zip_path, "-d", SAVE_DIR], check=True)
        os.remove(zip_path)
        print(f"✔️ Unzipped and removed {zip_path}")
    except Exception as e:
        print(f"❌ Failed to unzip {zip_path}. Error: {e}")

def main():
    """Main function to download and unzip all files."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

    # Download files sequentially
    downloaded_paths = [download_file(f) for f in FILE_LIST]

    # Unzip files in parallel
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(unzip_file, [p for p in downloaded_paths if p])
    
    # Clean up the temporary download directory
    if os.path.exists(TEMP_DOWNLOAD_DIR) and not os.listdir(TEMP_DOWNLOAD_DIR):
        os.rmdir(TEMP_DOWNLOAD_DIR)

    print("🎉 All done!")

if __name__ == "__main__":
    main()