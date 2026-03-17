import os
import urllib.request
import tarfile

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
DOWNLOAD_DIR = "data"
ARCHIVE_PATH = os.path.join(DOWNLOAD_DIR, "CUB_200_2011.tgz")

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, int(downloaded * 100 / total_size))
        print(f"\rDownloading CUB dataset... {percent}%", end="")

def main():
    # 1. Create the data directory if it doesn't exist
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # 2. Download the file
    if not os.path.exists(ARCHIVE_PATH):
        print(f"Starting download from {CUB_URL}...")
        urllib.request.urlretrieve(CUB_URL, ARCHIVE_PATH, reporthook=download_progress)
        print("\nDownload complete!")
    else:
        print("Archive already exists. Skipping download.")

    # 3. Extract the tar file
    print("Extracting files...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DOWNLOAD_DIR)
    print("Extraction complete")

    # 4. Clean up the heavy archive file
    print("Cleaning up .tgz archive...")
    os.remove(ARCHIVE_PATH)
    
    print(f"CUB images and annotations are in the '{DOWNLOAD_DIR}' folder.")

if __name__ == "__main__":
    main()