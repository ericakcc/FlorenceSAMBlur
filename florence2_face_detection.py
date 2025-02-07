import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

# 設定模型快取資料夾
MODEL_NAME = "microsoft/Florence-2-large-ft"
CACHE_DIR = "content/my_models/Florence_2"

# 讀取模型與處理器
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    device_map="cuda",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

def find_all_faces(image):
    """尋找圖片中的所有人臉並標示出來"""
    PROMPT = "<OD>"
    task_type = "<OD>"

    inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,
        do_sample=False
    )
    
    text_generations = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    results = processor.post_process_generation(text_generations, task=task_type, image_size=(image.width, image.height))

    print(results)
    raw_list = [bbox for bbox, label in zip(results[task_type]['bboxes'], results[task_type]['labels']) if label == 'human face']

    # 繪製偵測結果
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(results[task_type]['bboxes'], results[task_type]['labels']):
        if label == 'human face':
            x1, y1, x2, y2 = bbox
            rect_box = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect_box)
            plt.text(x1, y1, label, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.axis('off')
    return raw_list, fig

def calculate_iou(box1, box2):
    """計算兩個框框的 IoU (Intersection over Union)"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 計算交集區域
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 計算各自面積
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 計算聯集面積
    union = area1 + area2 - intersection
    
    return intersection / union

def filter_face_boxes(boxes, image_size, max_size_ratio=0.3, max_iou=0.3, aspect_ratio_range=(0.5, 2.0)):
    """過濾人臉框框
    
    參數:
        boxes: 框框列表 [x1, y1, x2, y2]
        image_size: 圖片大小 (width, height)
        max_size_ratio: 框框最大可接受的比例（相對於圖片大小）
        max_iou: 最大可接受的重疊度
        aspect_ratio_range: 可接受的長寬比範圍
    """
    if not boxes:
        return []
    
    image_width, image_height = image_size
    image_area = image_width * image_height
    filtered_boxes = []
    
    # 依據框框面積從小到大排序（因為臉部通常比較小）
    boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height
        
        # 檢查大小比例
        if area / image_area > max_size_ratio:
            continue
            
        # 檢查長寬比
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue
            
        # 檢查與已保留框框的重疊度
        overlap = False
        for kept_box in filtered_boxes:
            if calculate_iou(box, kept_box) > max_iou:
                overlap = True
                break
                
        if not overlap:
            filtered_boxes.append(box)
    
    return filtered_boxes

def find_main_speakers(image):
    """尋找圖片中的主要發言者人臉"""
    PROMPT = "<CAPTION_TO_PHRASE_GROUNDING> human face (main speaker)"
    task_type = "<CAPTION_TO_PHRASE_GROUNDING>"

    inputs = processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,
        do_sample=False
    )

    text_generations = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    results = processor.post_process_generation(text_generations, task=task_type, image_size=(image.width, image.height))
    print(results)
    
    # 先篩選出主要發言者的清單
    speaker_face_list = [bbox for bbox, label in zip(results[task_type]['bboxes'], results[task_type]['labels']) if label == 'human face']
    
    # 過濾框框，調整參數使其更嚴格
    filtered_faces = filter_face_boxes(
        speaker_face_list, 
        (image.width, image.height),
        max_size_ratio=0.15,  # 降低最大框框比例
        max_iou=0.2,         # 降低允許的重疊度
        aspect_ratio_range=(0.7, 1.4)  # 更嚴格的長寬比
    )
    
    # 繪製偵測結果
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox in filtered_faces:
        x1, y1, x2, y2 = bbox
        rect_box = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect_box)
        plt.text(x1, y1-5, 'main speaker', color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.axis('off')
    return filtered_faces, fig


def main():
    """測試主程式"""
    # 創建輸出資料夾
    output_dir = "test_florence2_output"
    os.makedirs(output_dir, exist_ok=True)
    
    image_path = "video_frames/frame_0100.jpg"

    if not os.path.exists(image_path):
        print(f"錯誤: 找不到圖片 {image_path}")
        return

    image = Image.open(image_path).convert("RGB")

    # print("\n偵測所有人臉...")
    # faces, faces_fig = find_all_faces(image)
    # print("找到的臉部座標:", faces)
    
    # # 保存所有人臉偵測結果
    # faces_fig.savefig(os.path.join(output_dir, "all_faces.png"))
    # plt.close(faces_fig)

    print("\n偵測主要發言者...")
    speakers, speakers_fig = find_main_speakers(image)
    print("主要發言者座標:", speakers)
    
    # 保存主要發言者偵測結果
    speakers_fig.savefig(os.path.join(output_dir, "main_speakers.png"))
    plt.close(speakers_fig)


if __name__ == "__main__":
    main()
