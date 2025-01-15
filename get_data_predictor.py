import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

def download_and_extract_kaggle_dataset(dataset_name, download_dir, extract_path):
    """
    Download, extract a dataset from Kaggle, and delete the download directory.

    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., "aleksandrpikul222/nuaaaa").
        download_dir (str): Directory to save the downloaded dataset zip file.
        extract_path (str): Directory to extract the dataset.
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Path for the ZIP file
    zip_file_path = os.path.join(download_dir, f"{dataset_name.split('/')[-1]}.zip")

    # Check if the file is already downloaded
    if not os.path.exists(zip_file_path):
        print(f"Downloading dataset {dataset_name}...")
        # Download dataset
        api.dataset_download_files(dataset_name, path=download_dir, unzip=False)
        print("Download complete.")
    else:
        print(f"Dataset {dataset_name} already downloaded.")

    # Extract dataset
    if not os.path.exists(extract_path):
        print(f"Extracting dataset {dataset_name}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting files"):
                zip_ref.extract(file, extract_path)
        print(f"Extraction complete: {extract_path}")
    else:
        print(f"Dataset {dataset_name} already extracted.")

    # Delete download directory
    if os.path.exists(download_dir):
        print(f"Deleting download directory: {download_dir}...")
        shutil.rmtree(download_dir)
        print(f"Download directory {download_dir} deleted.")

def rename_folders(base_path):
    """
    Rename ClientRaw to Real and ImposterRaw to Fake.

    Args:
        base_path (str): Path to the base folder containing ClientRaw and ImposterRaw.
    """
    client_folder = os.path.join(base_path, "ClientRaw")
    imposter_folder = os.path.join(base_path, "ImposterRaw")

    real_folder = os.path.join(base_path, "Real")
    fake_folder = os.path.join(base_path, "Fake")

    if os.path.exists(client_folder):
        print(f"Renaming {client_folder} to {real_folder}...")
        os.rename(client_folder, real_folder)

    if os.path.exists(imposter_folder):
        print(f"Renaming {imposter_folder} to {fake_folder}...")
        os.rename(imposter_folder, fake_folder)

    print("Folder renaming completed.")

def create_test_folder(base_path, test_folder, num_images=3):
    """
    Create a test folder containing samples from ClientRaw and ImposterRaw folders.
    Takes first 5 images from each numbered subfolder.

    Args:
        base_path (str): Path to the base folder containing ClientRaw and ImposterRaw folders.
        test_folder (str): Path to the test folder to be created.
        num_images (int): Number of images to copy from each subfolder (default: 3).
    """
    client_folder = os.path.join(base_path, "ClientRaw")
    imposter_folder = os.path.join(base_path, "ImposterRaw")

    # Ensure the test folder structure
    real_test_folder = os.path.join(test_folder, "Real")
    fake_test_folder = os.path.join(test_folder, "Fake")
    os.makedirs(real_test_folder, exist_ok=True)
    os.makedirs(fake_test_folder, exist_ok=True)

    # Function to copy images from source to destination
    def copy_folder_images(src_folder, dest_folder):
        if os.path.exists(src_folder):
            print(f"Copying {num_images} images from each folder in {src_folder} to {dest_folder}...")
            for subfolder in tqdm(sorted(os.listdir(src_folder)), desc=f"Processing {os.path.basename(dest_folder)}"):
                subfolder_path = os.path.join(src_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    # Get first 5 images from the subfolder
                    images = sorted(os.listdir(subfolder_path))[:num_images]
                    for image in images:
                        src = os.path.join(subfolder_path, image)
                        dest = os.path.join(dest_folder, f"{subfolder}_{image}")
                        shutil.copy(src, dest)

    # Copy images from ClientRaw to test/Real
    copy_folder_images(client_folder, real_test_folder)
    
    # Copy images from ImposterRaw to test/Fake
    copy_folder_images(imposter_folder, fake_test_folder)

    print(f"Test folder created at: {test_folder}")

if __name__ == "__main__":
    # Dataset identifiers
    dataset_nuaaaa = "aleksandrpikul222/nuaaaa"
    dataset_predictor = "sergiovirahonda/shape-predictor-68-face-landmarksdat"

    # Paths for datasets
    download_dir_nuaaaa = "nuaaaa"
    extract_path_nuaaaa = "data"

    download_dir_predictor = "shape_predictor"
    extract_path_predictor = "model_predictor"

    # Final test folder
    test_folder = "data/test"

    # Download and extract datasets
    download_and_extract_kaggle_dataset(dataset_nuaaaa, download_dir_nuaaaa, extract_path_nuaaaa)
    download_and_extract_kaggle_dataset(dataset_predictor, download_dir_predictor, extract_path_predictor)

    # Create test folder with 5 images from each subfolder
    # Note: We create the test folder before renaming to ensure we copy from the original folder names
    create_test_folder(base_path="data/raw", test_folder=test_folder, num_images=3)

    # Rename folders (if needed)
    rename_folders(extract_path_nuaaaa)

    print("All downloads, extractions, and post-processing completed.")