from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Fracture Detection API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:3000',
        'http://localhost:5173',
        'http://localhost:5174',
        'http://localhost:5175',
        'http://localhost:5176',
        'http://localhost:5177',
        'http://localhost:5178',
        'http://localhost:5179',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5173',
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    grad_cam_image: Optional[str] = None
    grad_cam_layer: Optional[str] = None


# Global model
model = None
last_conv_layer = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'fracture_model.h5'


def infer_class_labels():
    """
    Infer class labels from training directory structure.
    Classes are assigned indices alphabetically by folder name.
    """
    data_dirs = [
        'archive (1)/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train',
        'archive/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/train',
        'train/',
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            classes = sorted([d for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d))])
            if len(classes) >= 2:
                logger.info(f"✓ Found training classes: {classes}")
                logger.info(f"  Class 0: '{classes[0]}'")
                logger.info(f"  Class 1: '{classes[1]}'")
                return classes[:2]
    
    logger.warning("⚠ Could not find training directory. Using default class mapping.")
    return ['fractured', 'not fractured']


CLASS_LABELS = infer_class_labels()


def iter_layers(layer):
    """Yield nested layers recursively so Grad-CAM can find MobileNetV2 blocks."""
    for child in getattr(layer, 'layers', []):
        yield child
        yield from iter_layers(child)


def find_last_conv_layer(model_obj):
    """Return the last layer with a 4D output tensor."""
    last_layer = None
    for layer in iter_layers(model_obj):
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                last_layer = layer
        except Exception:
            continue
    return last_layer


@app.on_event('startup')
async def load_model_on_startup():
    global model, last_conv_layer
    try:
        if not MODEL_PATH.exists():
            logger.warning(f"⚠ Model file not found at {MODEL_PATH}")
            logger.info("Using dummy model for demo")
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"✓ Model loaded from {MODEL_PATH}")
        last_conv_layer = find_last_conv_layer(model)
        if last_conv_layer is not None:
            logger.info(f"✓ Grad-CAM last conv layer: {last_conv_layer.name}")
        else:
            logger.warning("⚠ Could not find a convolutional layer for Grad-CAM")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_model_input_size():
    """Infer the model's expected spatial input size."""
    if model is None:
        return 224, 224

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape and len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
        return int(input_shape[1]), int(input_shape[2])

    return 224, 224


def preprocess_image(image_bytes: bytes, target_size=None) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if target_size is None:
            target_size = get_model_input_size()
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype='float32') / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def sigmoid_label_score(probability: float, predicted_class: int) -> float:
    """Score used for Grad-CAM when the model outputs a single sigmoid probability."""
    return probability if predicted_class == 1 else 1.0 - probability


def jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Create a jet-style RGB heatmap without extra dependencies."""
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * heatmap - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * heatmap - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * heatmap - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def normalize_label(label: str) -> str:
    """Convert raw directory labels into the frontend-friendly display labels."""
    cleaned = label.strip().lower()
    if cleaned == 'fractured':
        return 'Fractured'
    if cleaned == 'not fractured':
        return 'Not Fractured'
    return label.strip().title()


def encode_pil_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


def create_grad_cam(image_bytes: bytes, predicted_class: int) -> Optional[str]:
    """Generate a Grad-CAM overlay image as a base64 data URI."""
    if model is None or last_conv_layer is None:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size
        input_image = preprocess_image(image_bytes)

        backbone = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
        if backbone is None:
            return None

        x = backbone.output
        for layer in model.layers[1:]:
            x = layer(x)

        grad_model = tf.keras.models.Model(
            inputs=backbone.input,
            outputs=[last_conv_layer.output, x],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_image)
            if predictions.shape[-1] == 1:
                class_score = predictions[:, 0]
                if predicted_class == 0:
                    class_score = 1.0 - class_score
            else:
                class_score = predictions[:, predicted_class]

        gradients = tape.gradient(class_score, conv_outputs)
        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)
        if float(max_value) == 0.0:
            return None

        heatmap = heatmap / (max_value + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(original_size, Image.Resampling.BILINEAR)
        heatmap_array = np.array(heatmap_image, dtype='float32') / 255.0
        colored_heatmap = jet_colormap(heatmap_array)

        original_array = np.array(image, dtype='float32') / 255.0
        overlay = np.clip((0.58 * original_array) + (0.42 * colored_heatmap), 0.0, 1.0)
        overlay_image = Image.fromarray(np.uint8(overlay * 255.0))
        return encode_pil_to_data_uri(overlay_image)

    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {e}")
        return None


@app.get('/health')
async def health_check():
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'class_labels': CLASS_LABELS,
        'grad_cam_layer': last_conv_layer.name if last_conv_layer is not None else None,
    }


@app.get('/info')
async def model_info():
    if not model:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    return {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'class_labels': CLASS_LABELS,
        'class_mapping': {'0': CLASS_LABELS[0], '1': CLASS_LABELS[1]},
        'grad_cam_layer': last_conv_layer.name if last_conv_layer is not None else None,
        'note': 'Classes sorted alphabetically from training directory'
    }


@app.post('/predict', response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    if not file.content_type or 'image' not in file.content_type:
        raise HTTPException(status_code=400, detail='File must be an image')
    
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail='File is empty')
        
        x = preprocess_image(contents)
        preds = model.predict(x, verbose=0)
        
        if preds.shape[-1] == 1:
            # Sigmoid: prob > 0.5 → Class 1, < 0.5 → Class 0
            prob = float(preds[0][0])
            if prob > 0.5:
                predicted_class = 1
                confidence = round(prob * 100.0, 1)
            else:
                predicted_class = 0
                confidence = round((1 - prob) * 100.0, 1)
            label = normalize_label(CLASS_LABELS[predicted_class])
            logger.info(f"Sigmoid {prob:.4f} → {label} ({confidence}%)")
        else:
            # Softmax: argmax of class probabilities
            class_probs = preds[0]
            predicted_class = int(np.argmax(class_probs))
            confidence = round(class_probs[predicted_class] * 100.0, 1)
            label = normalize_label(CLASS_LABELS[predicted_class])
            logger.info(f"Softmax → {label} ({confidence}%)")

        grad_cam_image = create_grad_cam(contents, predicted_class)
        
        return PredictionResponse(
            prediction=label,
            confidence=confidence,
            grad_cam_image=grad_cam_image,
            grad_cam_layer=last_conv_layer.name if last_conv_layer is not None else None,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

