import os
from data_transformer import MelanomaImagePreprocessor
from data_final_prepare import DatasetPreparer
import cv2
import logging
from tqdm import tqdm
import numpy as np

def process_cli_images():
    # Logging settings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Directory paths
    cli_images_dir = "cli_images"
    transformed_dir = "cli_transformed"
    final_dir = "cli_neural_net"
    
    # Create directories
    os.makedirs(transformed_dir, exist_ok=True)
    os.makedirs(os.path.join(transformed_dir, "images"), exist_ok=True)
    
    # Create image preprocessor
    preprocessor = MelanomaImagePreprocessor(target_size=(128, 128))
    
    # Process malignant and benign images
    for class_name in ["malignant", "benign"]:
        source_dir = os.path.join(cli_images_dir, class_name)
        if not os.path.exists(source_dir):
            logging.error(f"{source_dir} not found!")
            continue
            
        logging.info(f"Processing {class_name} images...")
        
        # Process images
        processed_images, filenames = preprocessor.process_dataset(source_dir)
        
        # Save processed images
        for img, fname in tqdm(zip(processed_images, filenames), desc=f"Saving {class_name}"):
            if img is not None:
                # Convert from [0,1] range to [0,255] range
                img = (img * 255).astype(np.uint8)
                save_path = os.path.join(transformed_dir, "images", fname)
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Create metadata.csv file
    with open(os.path.join(transformed_dir, "metadata.csv"), "w") as f:
        f.write("isic_id,benign_malignant\n")
        # Malignant images
        for img in os.listdir(os.path.join(cli_images_dir, "malignant")):
            img_id = os.path.splitext(img)[0]
            f.write(f"{img_id},malignant\n")
        # Benign images
        for img in os.listdir(os.path.join(cli_images_dir, "benign")):
            img_id = os.path.splitext(img)[0]
            f.write(f"{img_id},benign\n")
    
    # Split into train/val/test with Dataset preparer
    preparer = DatasetPreparer(
        source_dir=transformed_dir,
        target_dir=final_dir
    )
    preparer.setup_directory_structure()
    preparer.prepare_balanced_dataset(target_count=4500)  # Using all available images without limiting

if __name__ == "__main__":
    process_cli_images() 