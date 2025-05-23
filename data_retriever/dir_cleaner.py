import os
import shutil
import logging

def clean_directories():
    """
    Cleans up the workspace by:
    1. Removing the merged_data directory
    2. Removing extracted directories in downloaded_zips (keeping only zip files)
    """
    try:
        # Remove merged_data directory
        merged_data_path = "merged_data"
        if os.path.exists(merged_data_path):
            shutil.rmtree(merged_data_path)
            logging.info(f"Removed directory: {merged_data_path}")

        # Clean downloaded_zips directory
        zip_directory = "downloaded_zips"
        if os.path.exists(zip_directory):
            # List all items in the directory
            items = os.listdir(zip_directory)
            
            for item in items:
                item_path = os.path.join(zip_directory, item)
                # If it's a directory, remove it
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    logging.info(f"Removed extracted directory: {item_path}")

        logging.info("Directory cleanup completed successfully")

    except Exception as e:
        logging.error(f"Error during directory cleanup: {str(e)}")
        raise

def main():
    clean_directories()

if __name__ == "__main__":
    main()
