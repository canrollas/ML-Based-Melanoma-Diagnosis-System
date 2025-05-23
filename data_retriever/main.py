import os
import logging
from datetime import datetime
from data_retriever.downloader import main as download_main
from data_retriever.data_merger import main as merge_main
from data_retriever.data_transformer import MelanomaImagePreprocessor
from data_retriever.data_visualizer import DataVisualizer
from data_retriever.dir_cleaner import main as clean_main
from data_retriever.data_final_prepare import main as prepare_final_data
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm


# Configure logging
def setup_logging():
    """Configure logging with timestamp in filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'pipeline_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )


def create_directory_structure():
    """Create necessary directories for the pipeline"""
    directories = [
        "downloaded_zips",
        "merged_data",
        "merged_data/images",
        "data_transformed",
        "data_transformed/images",
        "data_visual_repr",
        "logs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")


def run_pipeline():
    """Execute the complete data processing pipeline"""
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting Melanoma Detection Pipeline")


        # Create directory structure
        create_directory_structure()

        # Step 1: Download datasets
        logging.info("Step 1: Downloading datasets")
        download_main()

        # Step 2: Merge datasets
        logging.info("Step 2: Merging datasets")
        merge_main()

        # Step 3: Transform images
        logging.info("Step 3: Transforming images")
        preprocessor = MelanomaImagePreprocessor(target_size=(128, 128))
        processed_dataset, original_filenames = preprocessor.process_dataset(
            "merged_data/images",
            augment=True
        )

        # Save processed images
        logging.info("Saving processed images...")
        for img, filename in tqdm(zip(processed_dataset, original_filenames),
                                desc="Saving Images",
                                total=len(processed_dataset)):
            try:
                # Create output directory
                os.makedirs(os.path.join("data_transformed", "images"), exist_ok=True)
                output_path = os.path.join("data_transformed/images", filename)
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(output_path)
            except Exception as e:
                logging.error(f"Saving error ({filename}): {str(e)}")

        # Copy metadata.csv file
        metadata_src = "merged_data/metadata.csv"
        metadata_dst = os.path.join("data_transformed", "metadata.csv")
        shutil.copy2(metadata_src, metadata_dst)
        logging.info("Metadata file copied")

        # Step 4: Generate visualizations
        logging.info("Step 4: Generating visualizations")
        visualizer = DataVisualizer(
            csv_path='merged_data/metadata.csv',
            output_dir='data_visual_repr',
            images_dir='merged_data/images'
        )

        # Run all visualizations
        visualizer.analyze_data()
        visualizer.analyze_missing_values(threshold=50)
        visualizer.analyze_key_features()
        visualizer.analyze_image_dimensions()

        # Step 5: Prepare final dataset for neural network
        logging.info("Step 5: Preparing final dataset for neural network")
        prepare_final_data()


        # Step 6: Clean up directories
        logging.info("Step 6: Cleaning up directories")
        clean_main()

        logging.info("Pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_pipeline() 