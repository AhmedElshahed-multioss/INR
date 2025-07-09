"""

- Run it in terminal: python download_div2k.py

- To import in another script: 
    from download_div2k import main
    main()

"""

import os
import requests
from tqdm import tqdm
from zipfile import ZipFile
from PIL import Image

BASE_DIR = "DIV2K"
DOWNLOAD_URLS = {
    "DIV2K_train_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_train_LR_bicubic_X2.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
    "DIV2K_train_LR_bicubic_X4.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
}
SCALES_TO_GENERATE = [8, 16]


def download_file(url, output_path):
    """Download file from URL with a progress bar."""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path), total=total, unit='B', unit_scale=True
    ) as pbar:
        for data in response.iter_content(1024):
            f.write(data)
            pbar.update(len(data))


def extract_zip(zip_path, extract_to):
    """Extract a ZIP file."""
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def generate_lr_images(hr_dir, out_base, scales):
    """Generate lower-resolution images from HR images."""
    for scale in scales:
        out_dir = os.path.join(out_base, f"DIV2K_train_LR_bicubic/X{scale}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating downscaled images for X{scale}...")

        for filename in tqdm(sorted(os.listdir(hr_dir))):
            if not filename.endswith(".png"):
                continue
            img_path = os.path.join(hr_dir, filename)
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img_down = img.resize((w // scale, h // scale), Image.BICUBIC)

            new_filename = filename.replace(".png", f"x{scale}.png")
            img_down.save(os.path.join(out_dir, new_filename))


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    # Download and extract datasets
    for filename, url in DOWNLOAD_URLS.items():
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            download_file(url, path)
        else:
            print(f"[✓] Already downloaded: {filename}")
        extract_zip(path, BASE_DIR)

    # Generate LR x8 and x16
    hr_dir = os.path.join(BASE_DIR, "DIV2K_train_HR")
    generate_lr_images(hr_dir, BASE_DIR, SCALES_TO_GENERATE)

    print("\nDone! You now have HR and LR x2, x4, x8, x16 images in the DIV2K/ directory.")


if __name__ == "__main__":
    main()
