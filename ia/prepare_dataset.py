import os
import tarfile
import random
import shutil
from tqdm import tqdm
import requests


# 📥 Téléchargement
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
            desc=f"Téléchargement {os.path.basename(dest_path)}",
            total=total, unit='iB', unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


# 📦 Extraction
def extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)


# 🧹 Préparation train/test
def split_dataset(image_root, output_dir, split_ratio=0.8):
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    breeds = os.listdir(image_root)
    for breed in tqdm(breeds, desc="Préparation train/test"):
        breed_dir = os.path.join(image_root, breed)
        images = os.listdir(breed_dir)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        for img_set, split in [(train_images, "train"), (test_images, "test")]:
            dest_dir = os.path.join(output_dir, split, breed)
            os.makedirs(dest_dir, exist_ok=True)
            for img in img_set:
                shutil.copy(os.path.join(breed_dir, img), os.path.join(dest_dir, img))


# 🚀 Main
def main():
    base_dir = "Stanford_Dogs_Dataset"
    os.makedirs(base_dir, exist_ok=True)

    # 1. URLs
    images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    images_tar_path = os.path.join(base_dir, "images.tar")

    # 2. Téléchargement et extraction
    if not os.path.exists(images_tar_path):
        download_file(images_url, images_tar_path)
    extract_tar(images_tar_path, base_dir)

    # 3. Organisation
    image_root = os.path.join(base_dir, "Images")
    output_dir = base_dir  # same dir, but inside: train/ test/
    split_dataset(image_root, output_dir)

    print("✅ Dataset prêt !")


if __name__ == "__main__":
    main()
