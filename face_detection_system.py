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

class FaceDetectionSystem:
    def __init__(self, florence_model_name="microsoft/Florence-2-large-ft", 
                 florence_cache_dir="content/my_models/Florence_2",
                 sam2_checkpoint="/home/ericakcc/segment-anything-2/checkpoints/sam2.1_hiera_small.pt",
                 sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml"):
        """
        Initialize the face detection system with Florence-2 and SAM2 models
        """
        # Initialize Florence-2 model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            florence_model_name,
            cache_dir=florence_cache_dir,
            device_map="cuda",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            florence_model_name,
            cache_dir=florence_cache_dir,
            trust_remote_code=True
        )
        
        # Initialize SAM2 model and predictor
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device='cuda')
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def detect_all_faces(self, image):
        """
        Detect all faces in the image
        Returns: list of face boxes and visualization figure
        """
        PROMPT = "<OD>"
        task_type = "<OD>"

        inputs = self.processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            do_sample=False
        )
        
        text_generations = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(text_generations, task=task_type, 
                                                       image_size=(image.width, image.height))

        face_boxes = [bbox for bbox, label in zip(results[task_type]['bboxes'], 
                                                results[task_type]['labels']) if label == 'human face']
        
        # Visualize results
        fig = self._visualize_boxes(image, face_boxes, "human face")
        return face_boxes, fig

    def detect_main_speakers(self, image):
        """
        Detect main speakers' faces in the image
        Returns: list of filtered face boxes and visualization figure
        """
        PROMPT = "<CAPTION_TO_PHRASE_GROUNDING> human face (main speaker)"
        task_type = "<CAPTION_TO_PHRASE_GROUNDING>"

        inputs = self.processor(text=PROMPT, images=image, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            do_sample=False
        )

        text_generations = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(text_generations, task=task_type, 
                                                       image_size=(image.width, image.height))
        
        speaker_faces = [bbox for bbox, label in zip(results[task_type]['bboxes'], 
                                                   results[task_type]['labels']) if label == 'human face']
        
        # Filter face boxes with stricter parameters
        filtered_faces = self._filter_face_boxes(
            speaker_faces, 
            (image.width, image.height),
            max_size_ratio=0.15,
            max_iou=0.2,
            aspect_ratio_range=(0.7, 1.4)
        )
        
        # Visualize results
        fig = self._visualize_boxes(image, filtered_faces, "main speaker")
        return filtered_faces, fig

    def detect_passerbys(self, image):
        """
        Detect passerby faces by excluding main speakers
        Returns: list of passerby face boxes
        """
        all_faces, _ = self.detect_all_faces(image)
        speaker_faces, _ = self.detect_main_speakers(image)
        return self._filter_boxes(all_faces, speaker_faces)

    def generate_masks(self, image, boxes):
        """
        Generate masks for given boxes using SAM2
        Returns: list of masks
        """
        self.sam2_predictor.set_image(np.array(image))
        
        masks = []
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            input_box = np.array([x1, y1, x2, y2])
            
            mask, scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )
            masks.append(mask[0])
        return masks

    def create_masked_image(self, image, masks, color=[0, 0, 255], alpha=0.5):
        """
        Create a masked image with highlighted regions
        Returns: PIL Image with masks applied
        """
        # Combine all masks
        combined_mask = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask.astype(bool))
        
        # Apply mask to image
        image_array = np.array(image)
        overlay = image_array.copy()
        overlay[combined_mask] = color
        
        result = image_array.copy()
        result[combined_mask] = (alpha * image_array[combined_mask] + 
                               (1 - alpha) * overlay[combined_mask]).astype(np.uint8)
        
        return Image.fromarray(result)

    @staticmethod
    def _filter_face_boxes(boxes, image_size, max_size_ratio=0.3, max_iou=0.3, 
                          aspect_ratio_range=(0.5, 2.0)):
        """
        Filter face boxes based on size, overlap, and aspect ratio
        """
        if not boxes:
            return []
        
        image_width, image_height = image_size
        image_area = image_width * image_height
        filtered_boxes = []
        
        # Sort boxes by area (ascending)
        boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height
            
            if (area / image_area <= max_size_ratio and
                aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
                not any(FaceDetectionSystem._calculate_iou(box, kept_box) > max_iou 
                       for kept_box in filtered_boxes)):
                filtered_boxes.append(box)
        
        return filtered_boxes

    @staticmethod
    def _calculate_iou(box1, box2):
        """
        Calculate Intersection over Union between two boxes
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union

    @staticmethod
    def _filter_boxes(initial_boxes, exclude_boxes, overlap_threshold=0.7):
        """
        Filter out boxes that overlap with exclude_boxes
        """
        filtered_boxes = []
        for box in initial_boxes:
            if not any(FaceDetectionSystem._is_overlapping(box, exclude_box, overlap_threshold) 
                      for exclude_box in exclude_boxes):
                filtered_boxes.append(box)
        return filtered_boxes

    @staticmethod
    def _is_overlapping(box1, box2, threshold=0.7):
        """
        Check if two boxes are overlapping above the threshold
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        overlap_area = x_overlap * y_overlap
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        min_area = min(area1, area2)
        
        return overlap_area >= threshold * min_area

    def _visualize_boxes(self, image, boxes, label):
        """
        Create visualization of boxes on image
        """
        fig, ax = plt.subplots()
        ax.imshow(image)
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            rect_box = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                       linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_box)
            plt.text(x1, y1-5, label, color='white', fontsize=10, 
                    bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')
        return fig

def main():
    """
    Example usage of FaceDetectionSystem
    """
    # Initialize system
    face_system = FaceDetectionSystem()
    
    # Create output directory
    output_dir = "test_florence2_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image_path = "video_frames/frame_0100.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    
    # Detect and mask passerbys
    print("\nDetecting passerbys...")
    passerbys = face_system.detect_passerbys(image)
    print("Passerby coordinates:", passerbys)
    
    # Generate and apply masks
    masks = face_system.generate_masks(image, passerbys)
    result_image = face_system.create_masked_image(image, masks)
    
    # Save result
    output_path = os.path.join(output_dir, "masked_result.png")
    result_image.save(output_path)
    print(f"Saved masked result to: {output_path}")

if __name__ == "__main__":
    main() 