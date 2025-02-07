import os
import requests

# 目標資料夾
model_dir = "my_models/sam2"
os.makedirs(model_dir, exist_ok=True)

# 下載連結
urls = [
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
]

def download_file(url, save_path):
    """下載檔案"""
    print(f"下載中: {url} -> {save_path}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"下載完成: {save_path}")
    else:
        print(f"下載失敗: {url}")

# 執行下載
for url in urls:
    filename = os.path.join(model_dir, os.path.basename(url))
    download_file(url, filename)

print("所有模型下載完成！")
