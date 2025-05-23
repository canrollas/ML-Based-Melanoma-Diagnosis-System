import os
import pandas as pd
import shutil
import logging
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

class DatasetPreparer:
    def __init__(self, source_dir="data_transformed", target_dir="neural_network_data"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.metadata_path = os.path.join(source_dir, "metadata.csv")
        self.images_dir = os.path.join(source_dir, "images")
        
        # Target directory structure
        self.train_dir = os.path.join(target_dir, "train")
        self.val_dir = os.path.join(target_dir, "validation")
        self.test_dir = os.path.join(target_dir, "test")
        
        # Class directories
        self.class_dirs = {
            "benign": ["benign"],
            "malignant": ["malignant"]
        }

    def setup_directory_structure(self):
        """Creates target directory structure"""
        # Clean and recreate main directory
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        
        # Create directory structure
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            for class_name in self.class_dirs.keys():
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        logging.info(f"Directory structure created: {self.target_dir}")

    def prepare_balanced_dataset(self, target_count=1000, train_ratio=0.7, val_ratio=0.15):
        """Creates balanced dataset and splits into train/val/test"""
        try:
            # Read metadata
            df = pd.read_csv(self.metadata_path)
            logging.info(f"Metadata file read. Total records: {len(df)}")
            
            # Check CSV contents
            logging.info("\nCSV columns:")
            for col in df.columns:
                logging.info(f"Column: {col}")
            
            # Check unique values
            logging.info("\nUnique values in benign_malignant column:")
            unique_vals = df['benign_malignant'].unique()
            logging.info(unique_vals)
            
            # Separate malignant and benign images
            malignant_samples = df[df['benign_malignant'] == 'malignant']['isic_id'].tolist()
            benign_samples = df[df['benign_malignant'] == 'benign']['isic_id'].tolist()
            
            logging.info(f"\nTotal malignant samples: {len(malignant_samples)}")
            logging.info(f"Total benign samples: {len(benign_samples)}")
            
            if len(malignant_samples) == 0 or len(benign_samples) == 0:
                raise ValueError("No malignant or benign samples found!")
            
            # Random selection from benign samples (down sampling)
            if len(benign_samples) > target_count:
                benign_samples = random.sample(benign_samples, target_count)
            
            # Random selection from malignant samples
            if len(malignant_samples) > target_count:
                malignant_samples = random.sample(malignant_samples, target_count)
            
            logging.info(f"\nSelected malignant samples: {len(malignant_samples)}")
            logging.info(f"Selected benign samples: {len(benign_samples)}")
            
            # Split dataset
            for class_name, samples in [("malignant", malignant_samples), ("benign", benign_samples)]:
                if not samples:
                    logging.error(f"No samples found for class {class_name}!")
                    continue
                    
                # Train/val/test split
                train_val_samples, test_samples = train_test_split(samples, test_size=1-train_ratio-val_ratio, random_state=42)
                train_samples, val_samples = train_test_split(train_val_samples, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
                
                logging.info(f"\n{class_name} split distribution:")
                logging.info(f"Train: {len(train_samples)}")
                logging.info(f"Validation: {len(val_samples)}")
                logging.info(f"Test: {len(test_samples)}")
                
                # Copy images
                splits = {
                    "train": (self.train_dir, train_samples),
                    "validation": (self.val_dir, val_samples),
                    "test": (self.test_dir, test_samples)
                }
                
                for split_name, (split_dir, split_samples) in splits.items():
                    target_class_dir = os.path.join(split_dir, class_name)
                    
                    for img_id in tqdm(split_samples, desc=f"Copying {class_name} {split_name} images"):
                        # Find image file
                        source_path = os.path.join(self.images_dir, f"{img_id}.jpg")
                        if not os.path.exists(source_path):
                            source_path = os.path.join(self.images_dir, f"{img_id}.png")
                        
                        if os.path.exists(source_path):
                            target_path = os.path.join(target_class_dir, os.path.basename(source_path))
                            shutil.copy2(source_path, target_path)
                        else:
                            logging.warning(f"Image not found: {img_id}")
            
            # Print statistics
            self._print_statistics()
            
        except Exception as e:
            logging.error(f"Dataset preparation error: {str(e)}")
            raise

    def _print_statistics(self):
        """Prints class distributions for each split"""
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_name = os.path.basename(split_dir)
            logging.info(f"\n{split_name} statistics:")
            
            for class_name in self.class_dirs.keys():
                class_dir = os.path.join(split_dir, class_name)
                count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))])
                logging.info(f"{class_name}: {count} images")

def main():
    # Logging settings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run dataset preparer
    preparer = DatasetPreparer()
    preparer.setup_directory_structure()
    preparer.prepare_balanced_dataset(target_count=1000)  # 1000 images per class

if __name__ == "__main__":
    main()
