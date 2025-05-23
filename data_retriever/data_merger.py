import os
import shutil
import pandas as pd
from tqdm import tqdm
import zipfile

zip_directory = "downloaded_zips"
# List of source directories
source_directories = [
    os.path.join(zip_directory, "ISIC_2017_Training"),
    os.path.join(zip_directory, "ISIC_2016_Training"),
    os.path.join(zip_directory, "ISIC_2016_Test"),
    os.path.join(zip_directory, "ISIC_2017_Test"),
    os.path.join(zip_directory, "ISIC_2018_Task3_Test"),
    os.path.join(zip_directory, "ISIC_2018_Task12_Training"),
    os.path.join(zip_directory, "ISIC_2020_Test"),
]

# Target directory
target_directory = "merged_data"
# Directory where zip files are located


def extract_zips():
    """
    Extracts downloaded zip files
    """
    print("\nExtracting zip files...")
    
    for source_dir in [os.path.basename(d) for d in source_directories]:
        zip_path = os.path.join(zip_directory, f"{source_dir}.zip")
        extract_path = os.path.join(zip_directory, source_dir)
        
        if not os.path.exists(zip_path):
            print(f"Warning: {zip_path} not found, skipping...")
            continue
            
        print(f"Extracting: {source_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Successfully extracted: {source_dir}")
        except Exception as e:
            print(f"Error: Problem occurred while extracting {source_dir}: {str(e)}")

def setup_target_directory(target_dir):
    """
    Creates or cleans the target directory
    """
    # Target directory and subdirectories
    target_image_dir = os.path.join(target_dir, "images")
    
    # Delete if directories exist
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # Create directories
    os.makedirs(target_image_dir)
    
    print(f"Target directory created: {target_dir}")
    return target_image_dir

def merge_images(source_dirs, target_image_dir):
    """
    Merges images from all source directories
    """
    print("\nMerging images...")
    copied_images = set()  # To track copied images
    total_copied = 0
    total_skipped = 0
    
    for source_dir in source_dirs:
        source_image_dir = os.path.join(source_dir, "")
        
        # Skip if source directory doesn't exist
        if not os.path.exists(source_image_dir):
            print(f"Warning: {source_image_dir} not found, skipping...")
            continue
        
        # List all images in directory
        images = [f for f in os.listdir(source_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Copy images with progress bar
        for image in tqdm(images, desc=f"Processing: {os.path.basename(source_dir)}"):
            if image not in copied_images:
                shutil.copy2(
                    os.path.join(source_image_dir, image),
                    os.path.join(target_image_dir, image)
                )
                copied_images.add(image)
                total_copied += 1
            else:
                total_skipped += 1
    
    return total_copied, total_skipped

def merge_metadata(source_dirs, target_dir):
    """
    Merges metadata files from all source directories
    """
    print("\nMerging metadata files...")
    dataframes = []
    
    for source_dir in source_dirs:
        metadata_path = os.path.join(source_dir, "metadata.csv")
        
        if not os.path.exists(metadata_path):
            print(f"Warning: {metadata_path} not found, skipping...")
            continue
            
        try:
            df = pd.read_csv(metadata_path)
            dataframes.append(df)
            print(f"Read: {os.path.basename(source_dir)}/metadata.csv ({len(df)} rows)")
        except Exception as e:
            print(f"Error: Failed to read {os.path.basename(source_dir)}/metadata.csv: {str(e)}")
    
    if not dataframes:
        raise Exception("No metadata files could be read!")
    
    # Merge dataframes and remove duplicate IDs
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['isic_id'], keep='first')
    
    # Save merged metadata
    output_path = os.path.join(target_dir, "metadata.csv")
    merged_df.to_csv(output_path, index=False)
    
    return len(merged_df)

def main():
    print("Starting Data Merge Process...")
    
    # Extract zip files
    extract_zips()
    
    # Prepare target directory
    target_image_dir = setup_target_directory(target_directory)
    
    try:
        # Merge images
        total_copied, total_skipped = merge_images(source_directories, target_image_dir)
        
        # Merge metadata
        total_metadata_rows = merge_metadata(source_directories, target_directory)
        
        # Report results
        print("\nProcess Completed!")
        print(f"- Number of copied images: {total_copied}")
        print(f"- Number of skipped duplicate images: {total_skipped}")
        print(f"- Number of merged metadata rows: {total_metadata_rows}")
        print(f"\nResults saved to '{target_directory}' directory.")
        
    except Exception as e:
        print(f"\nERROR: An error occurred during the process: {str(e)}")
        return

