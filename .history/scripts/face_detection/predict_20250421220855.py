import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

ethnic_labels = ["Jawa", "Sunda", "Minang"]

def build_ethnicity_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# LOAD pretrained model (setelah training)
model_path = "ethnic_classifier_model.keras"
model = tf.keras.models.load_model(model_path)

def predict_ethnicity(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    idx = np.argmax(pred)
    label = ethnic_labels[idx]
    confidence = round(float(pred[idx]), 4)

    return {
        "predicted_ethnicity": label,
        "confidence": confidence
    }