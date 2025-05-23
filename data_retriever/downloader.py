import os
import requests
from tqdm import tqdm

# Download links and names of zip files from ISIC Archive
dataset_info = [
    {
        "link": "https://zip.isic-archive.com/download?zsid=eyJ1c2VyIjpudWxsLCJxdWVyeSI6bnVsbCwiY29sbGVjdGlvbnMiOls2MF19:1uIWSu:M-AvOV-A9MRmUYvc1oyKGdAB6-GbFfh1QiIvzpXAefU",
        "name": "ISIC_2017_Training"
    },
    {
        "link": "https://zip.isic-archive.com/download?zsid=eyJ1c2VyIjpudWxsLCJxdWVyeSI6bnVsbCwiY29sbGVjdGlvbnMiOls3NF19:1uIWTi:V0lsPaquXznyw35xib0CikbL-PV44rXS1G2S6-8kF4E",
        "name": "ISIC_2016_Training"
    },
    {
        "link": "https://zip.isic-archive.com/download?zsid=eyJ1c2VyIjpudWxsLCJxdWVyeSI6bnVsbCwiY29sbGVjdGlvbnMiOls2MV19:1uIWUP:11B2e5jW9IRpkWeYgiuszutiVBBdiYUuCm4AGfDjha4",
        "name": "ISIC_2016_Test"
    },
    {
        "link": "https://zip.isic-archive.com/download?zsid=eyJ1c2VyIjpudWxsLCJxdWVyeSI6bnVsbCwiY29sbGVjdGlvbnMiOls2OV19:1uIXgO:TbQF-UH8hxE5yQIOpig7MFmWdN3UwtIkpvrINe4ixeY",
        "name": "ISIC_2017_Test"
    },
    {
        "link": "https://zip.isic-archive.com/download?zsid=eyJ1c2VyIjpudWxsLCJxdWVyeSI6bnVsbCwiY29sbGVjdGlvbnMiOls2N119:1uIZPR:-MD9q7ic3iko8U3zwk70o6fl8DtrjSw2cTzkvC2I_ko",
        "name": "ISIC_2018_Task3_Test"
    },
    {
        "link": "https://zip.isic-archive.com/download?zsid=eyJ1c2VyIjpudWxsLCJxdWVyeSI6bnVsbCwiY29sbGVjdGlvbnMiOls2M119:1uIZTw:iukdXIE3UaczsQHDX7Ryzya8WLfLZQxOxaEjVa9_-lI",
        "name": "ISIC_2018_Task12_Training"
    },
    {
        "link": "https://zip.isic-archive.com/download?zsid=eyJ1c2VyIjpudWxsLCJxdWVyeSI6bnVsbCwiY29sbGVjdGlvbnMiOls2OF19:1uIZcP:fx7ON38Dkv91SP8R7kfxbAmvbT8TxJXJcLRYy0kmKds",
        "name": "ISIC_2020_Test"
    }
]


def download_zip(url, save_path):
    """
    Downloads a zip file from the given URL and saves it to the specified path.
    Shows the download progress with a progress bar.
    """
    try:
        # Start download in stream mode
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Catch any errors

        # Get file size
        total_size = int(response.headers.get('content-length', 0))

        # Save the file
        with open(save_path, 'wb') as file, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

        return True

    except Exception as e:
        print(f"Error: Problem occurred while downloading {url}: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)  # Delete incomplete file
        return False


def main():
    # Directory where downloaded files will be saved
    download_dir = "downloaded_zips"
    os.makedirs(download_dir, exist_ok=True)

    print("Downloading zip files...")

    # Download each zip file
    success_count = 0
    skipped_count = 0
    for i, dataset in enumerate(dataset_info, 1):
        zip_path = os.path.join(download_dir, f"{dataset['name']}.zip")
        
        # Check if file already exists
        if os.path.exists(zip_path):
            print(f"\nSkipping file {i}/{len(dataset_info)}: {dataset['name']} (already exists)")
            skipped_count += 1
            success_count += 1
            continue

        print(f"\nDownloading file {i}/{len(dataset_info)}: {dataset['name']}")
        if download_zip(dataset['link'], zip_path):
            success_count += 1

    print(f"\nProcess completed:")
    print(f"- Total files: {len(dataset_info)}")
    print(f"- Successfully downloaded: {success_count - skipped_count}")
    print(f"- Skipped (already existed): {skipped_count}")
    print(f"- Failed: {len(dataset_info) - success_count}")
    print(f"\nFiles saved to '{download_dir}' directory.")



