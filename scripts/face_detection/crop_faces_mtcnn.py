from mtcnn import MTCNN
import cv2
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
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


if __name__ == "__main__":
    crop_faces("dataset/raw", "dataset/cropped")