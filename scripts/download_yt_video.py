import os
import subprocess
from glob import glob

def download_youtube_video(video_url, output_path="video.mp4"):
    """
    下載 YouTube 影片 (使用 yt-dlp)。
    
    Args:
        video_url (str): YouTube 影片的 URL。
        output_path (str): 下載後的影片儲存路徑。
    """
    cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio",
        "--merge-output-format", "mp4",
        "-o", output_path,
        video_url
    ]
    subprocess.run(cmd, check=True)
    print(f"下載完成: {output_path}")

def extract_frames(video_path, output_folder="video_frames", fps=1):
    """
    使用 FFmpeg 從影片中擷取影格。
    
    Args:
        video_path (str): 影片路徑。
        output_folder (str): 存放影格的資料夾。
        fps (int): 每秒擷取的影格數。
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        f"{output_folder}/frame_%04d.jpg"
    ]
    subprocess.run(cmd, check=True)
    print(f"影格已儲存於 {output_folder}")

def main():
    # 影片 URL
    youtube_url = "https://www.youtube.com/watch?v=dmAmE34qVL8"
    
    # 下載 YouTube 影片
    video_path = "video.mp4"
    download_youtube_video(youtube_url, video_path)
    
    # 擷取影格（每秒 1 張）
    extract_frames(video_path)

    # 列出擷取到的影格
    image_files = sorted(glob("video_frames/*.jpg"))
    print(f"擷取到 {len(image_files)} 張影格")

if __name__ == "__main__":
    main()
