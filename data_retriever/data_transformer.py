import numpy as np
import cv2
import albumentations as A
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple, List
from tqdm import tqdm

class MelanomaImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        self.scaler = StandardScaler()
        
        # Basic augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1)
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ], p=0.2),
        ])

    def normalize_color(self, image: np.ndarray) -> np.ndarray:
        """Applies color normalization"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Normalize L channel
        l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        
        # Merge channels
        normalized_lab = cv2.merge([l_norm, a, b])
        
        # Convert back to RGB
        normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
        return normalized_rgb

    def remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Cleans artifacts from the image"""
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Clean small artifacts using morphological operations
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        return cleaned

    def preprocess_image(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """Main preprocessing function"""
        try:
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Color normalization
            image = self.normalize_color(image)
            
            # Artifact removal
            image = self.remove_artifacts(image)
            
            # Normalize pixel values to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            if augment:
                augmented = self.augmentation(image=image)
                image = augmented['image']
            
            return image
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return None

    def process_dataset(self, image_dir: str, augment: bool = False) -> Tuple[List[np.ndarray], List[str]]:
        """Processes the entire dataset"""
        processed_images = []
        filenames = []
        
        if not os.path.exists(image_dir):
            print(f"Error: Image directory not found: {image_dir}")
            return [], []

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in tqdm(image_files, desc="Processing Images"):
            image_path = os.path.join(image_dir, filename)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_image = self.preprocess_image(image, augment=augment)
                
                if processed_image is not None:
                    processed_images.append(processed_image)
                    filenames.append(filename)
            except Exception as e:
                continue
        
        return processed_images, filenames
