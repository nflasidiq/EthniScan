<<<<<<< HEAD
=======
from mtcnn import MTCNN
>>>>>>> 8e13e37ad06f16981b8c7b73b250e1f0b785e66e
import cv2
import os
from pathlib import Path
import numpy as np
<<<<<<< HEAD
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import gaussian_filter

=======
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from scipy.ndimage import gaussian_filter
import random
import matplotlib.pyplot as plt

# Resize image to desired input size
>>>>>>> 8e13e37ad06f16981b8c7b73b250e1f0b785e66e
def preprocessing(image, size=(224, 224)):
    if image is not None:
        image = cv2.resize(image, size)
        return image
    return None

<<<<<<< HEAD
def add_noise(image, noise_factor=0.01):
    noise = np.random.normal(0, 1, image.shape) * noise_factor
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

datagen = ImageDataGenerator(
    rotation_range=15,
    brightness_range=[0.8, 1.2],
=======
# Add random Gaussian noise
def add_noise(image, noise_factor=0.00089):
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

# Apply Gaussian blur
def apply_blur_aug(image, blur_factor=2.5):
    return gaussian_filter(image, sigma=blur_factor)

# ImageDataGenerator for standard augmentations
datagen = ImageDataGenerator(
    rotation_range=19,
    shear_range=10,
    zoom_range=0.26,
    brightness_range=[0.7, 1.3],
>>>>>>> 8e13e37ad06f16981b8c7b73b250e1f0b785e66e
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255
)

<<<<<<< HEAD
=======
# Combine custom and standard augmentations
>>>>>>> 8e13e37ad06f16981b8c7b73b250e1f0b785e66e
def augment_image(image):
    processed_image = preprocessing(image)
    if processed_image is None:
        return None
<<<<<<< HEAD
    processed_image = add_noise(processed_image / 255.0)
    processed_image = (processed_image * 255).astype(np.uint8)
    augmented_images = datagen.flow(np.expand_dims(processed_image, axis=0), batch_size=1)
    return next(augmented_images)[0]

def preprocessing_augmentation_pipeline(input_dir, temp_output_dir, numberOfduplication=1):
    input_dir = Path(input_dir)
    temp_output_dir = Path(temp_output_dir)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
=======

    # Randomly choose a mode
    mode = random.choice(["noise", "blur", "noise_blur", "none"])

    if mode == "noise":
        processed_image = add_noise(processed_image)
    elif mode == "blur":
        processed_image = apply_blur_aug(processed_image)
    elif mode == "noise_blur":
        processed_image = add_noise(processed_image)
        processed_image = apply_blur_aug(processed_image)
    
    # Apply ImageDataGenerator
    augmented_images = datagen.flow(np.expand_dims(processed_image, axis=0), batch_size=1)
    return next(augmented_images)[0]

# Pipeline to apply augmentation to entire folder
def preprocessing_augmentation_pipeline(input_dir, output_dir, numberOfduplication=1):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

>>>>>>> 8e13e37ad06f16981b8c7b73b250e1f0b785e66e
    for i in range(numberOfduplication):
        for label_folder in input_dir.iterdir():
            if not label_folder.is_dir():
                continue
<<<<<<< HEAD
            (temp_output_dir / label_folder.name).mkdir(exist_ok=True)
            for image_path in label_folder.glob("*.*"):
                if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
=======
            (output_dir / label_folder.name).mkdir(exist_ok=True)

            for image_path in label_folder.glob("*.*"):
                if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Skipped: {image_path.name} (can't be read)")
                    continue

>>>>>>> 8e13e37ad06f16981b8c7b73b250e1f0b785e66e
                img = augment_image(img)
                if img is not None:
                    img = (img * 255).astype(np.uint8)
                    image_name = f"{image_path.stem}_aug_{i}.jpg"
<<<<<<< HEAD
                    save_path = temp_output_dir / label_folder.name / image_name
                    cv2.imwrite(str(save_path), img)

def split_by_identity(temp_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    temp_dir = Path(temp_dir)
    output_dir = Path(output_dir)
    for label_folder in temp_dir.iterdir():
        if not label_folder.is_dir():
            continue
        person_groups = {}
        for img_path in label_folder.glob("*.jpg"):
            person_id = img_path.name.split("_")[0]
            person_groups.setdefault(person_id, []).append(img_path)
        person_ids = list(person_groups.keys())
        random.shuffle(person_ids)
        total = len(person_ids)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        subsets = {
            'train': person_ids[:train_end],
            'valid': person_ids[train_end:val_end],
            'test': person_ids[val_end:]
        }
        for subset, ids in subsets.items():
            for pid in ids:
                for img_path in person_groups[pid]:
                    subset_dir = output_dir / subset / label_folder.name
                    subset_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(img_path, subset_dir / img_path.name)

if __name__ == "__main__":
    preprocessing_augmentation_pipeline("dataset/cropped", "dataset/temp_augmented", 4)
    split_by_identity("dataset/temp_augmented", "dataset/preprocessed-augmented")
=======
                    save_path = output_dir / label_folder.name / image_name
                    cv2.imwrite(str(save_path), img)

# Run the pipeline
if __name__ == "__main__":
    preprocessing_augmentation_pipeline("dataset/cropped", "dataset/preprocessed-augmented", 4)
>>>>>>> 8e13e37ad06f16981b8c7b73b250e1f0b785e66e
