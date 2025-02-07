import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

sam2_checkpoint = "/home/ericakcc/segment-anything-2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
sam2_predictor = SAM2ImagePredictor(sam2_model)

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

def is_overlapping(box1, box2, threshold=0.7):
    # Unpack coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate the overlap area
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    # Calculate the area of each box
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    overlap_area = x_overlap * y_overlap
    
    # Calculate the minimum area of the two boxes
    min_area = min(area1, area2)
    
    # Check if the overlap area is at least the specified percentage of the smaller box's area
    return overlap_area >= threshold * min_area

def filter_boxes(initial_boxes, new_boxes):
    filtered_boxes = []
    
    for box in initial_boxes:
        if not any(is_overlapping(box, new_box) for new_box in new_boxes):
            filtered_boxes.append(box)
            
    return filtered_boxes


def find_all_passerbys(image):
    raw_list, _ = find_all_faces(image)
    speaker_face_list, _ = find_main_speakers(image)
    
    filtered_boxes = filter_boxes(raw_list, speaker_face_list)
    
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


def get_mask_withSAM2(image, bboxes):
    """使用 SAM2 生成路人遮罩"""
    # 先設定圖片
    sam2_predictor.set_image(np.array(image))
    
    masks = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        input_box = np.array([x1, y1, x2, y2])
        
        # 使用正確的 SAM2 預測 API
        mask, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )
        masks.append(mask[0])  # 只取第一個遮罩，因為 multimask_output=False
    return masks

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

    # print("\n偵測主要發言者...")
    # speakers, speakers_fig = find_main_speakers(image)
    # print("主要發言者座標:", speakers)
    
    # # 保存主要發言者偵測結果
    # speakers_fig.savefig(os.path.join(output_dir, "main_speakers.png"))
    # plt.close(speakers_fig)

    print("\n偵測路人...")
    passerbys = find_all_passerbys(image)
    print("路人座標:", passerbys)
    
    # 生成路人遮罩
    masks = get_mask_withSAM2(image, passerbys)
    
    # 將所有遮罩合併成一個，確保使用相同的資料型態
    combined_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask.astype(bool))
    
    # 將圖片轉換為numpy陣列
    image_array = np.array(image)
    
    # 創建一個半透明的藍色遮罩
    overlay = image_array.copy()
    overlay[combined_mask] = [0, 0, 255]  # 藍色
    
    # 混合原圖和遮罩
    alpha = 0.5  # 透明度
    result = image_array.copy()
    result[combined_mask] = (alpha * image_array[combined_mask] + (1 - alpha) * overlay[combined_mask]).astype(np.uint8)
    
    # 儲存結果
    result_image = Image.fromarray(result)
    result_image.save(os.path.join(output_dir, "masked_result.png"))
    print("已儲存遮罩結果到:", os.path.join(output_dir, "masked_result.png"))

if __name__ == "__main__":
    main()
