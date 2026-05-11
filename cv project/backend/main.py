from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
import logging

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


# Global model
model = None


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


@app.on_event('startup')
async def load_model_on_startup():
    global model
    try:
        model_path = 'fracture_model.h5'
        if not os.path.exists(model_path):
            logger.warning(f"⚠ Model file not found at {model_path}")
            logger.info("Using dummy model for demo")
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"✓ Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype='float32') / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


@app.get('/health')
async def health_check():
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'class_labels': CLASS_LABELS
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
            label = CLASS_LABELS[predicted_class]
            logger.info(f"Sigmoid {prob:.4f} → {label} ({confidence}%)")
        else:
            # Softmax: argmax of class probabilities
            class_probs = preds[0]
            predicted_class = int(np.argmax(class_probs))
            confidence = round(class_probs[predicted_class] * 100.0, 1)
            label = CLASS_LABELS[predicted_class]
            logger.info(f"Softmax → {label} ({confidence}%)")
        
        return PredictionResponse(prediction=label, confidence=confidence)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

