from mtcnn import MTCNN
import cv2
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import gaussian_filter
import random
import matplotlib.pyplot as plt

def crop_faces(input_dir, output_dir):
    detector = MTCNN()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label_folder in input_dir.iterdir():
        if label_folder.is_dir():
            (output_dir / label_folder.name).mkdir(exist_ok=True)
            for image_path in label_folder.glob("*.*"):
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Failed to read {image_path}")
                    continue
                results = detector.detect_faces(img)
                if results:
                    x, y, w, h = results[0]['box']
                    face = img[y:y+h, x:x+w]
                    save_path = output_dir / label_folder.name / image_path.name
                    cv2.imwrite(str(save_path), face)
                    print(f"Cropped: {save_path}")
                else:
                    print(f"No face detected: {image_path}")

    
def preprocessing(image, size = (224,224)):
     if image is not None:
                cv2.resize(image, size)
                return image

def add_noise(image, noise_factor=0.00089):
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def apply_blur(image, blur_factor =2.5):
    return gaussian_filter(image, sigma=blur_factor)

datagen = ImageDataGenerator(
    rotation_range=19,  # ±19 degrees rotation
    shear_range=10,  # ±10° horizontal shear
    zoom_range=0.26,  # 26% zoom
    brightness_range=[0.7, 1.3],  # ±30% brightness
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255
)

def augment_image(image, apply_noise=True, apply_blur=True, apply_noise_and_blur=True):
    processed_image = preprocessing(image)
    if processed_image is not None:
        if apply_noise_and_blur:
            processed_image = add_noise(processed_image)
            processed_image = apply_blur(processed_image)
        elif apply_noise:
            processed_image = add_noise(processed_image)
        elif apply_blur:
            processed_image = apply_blur(processed_image)

        augmented_images = datagen.flow(np.expand_dims(processed_image, axis=0))
        return augmented_images
    return None

def preprocessing_augmentation_pipeline(input_dir, output_dir, numberOfduplication = 1):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label_folder in input_dir.iterdir():
        (output_dir / label_folder.name).mkdir(exist_oke=True)
        for image_path in label_folder.glob("*.*"):
            img = cv2.imread(str(image_path))
            img = preprocessing(img)
            img = augment_image(img, apply_noise = random.choice([True, False]), apply_blur = ([True, False]),apply_noise_and_blur=random.choice([True, False]), numberOfduplication = 4)
            save_path = output_dir / label_folder.name / image_path.name
            cv2.imwrite(str(save_path), img)
        return None

if __name__ == "__main__":
    crop_faces("dataset/raw", "dataset/cropped")