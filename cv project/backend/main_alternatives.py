"""
Alternative backend implementations for different class label orders.
Copy the correct one to backend/main.py based on your model's class mapping.
"""

# ============================================================================
# OPTION A: Standard Order ("fractured" = Class 0, "not fractured" = Class 1)
# Use this if: 'fractured' folder comes before 'not fractured' alphabetically
# ============================================================================

MAIN_PY_OPTION_A = '''
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


model = None


@app.on_event('startup')
async def load_model_on_startup():
    global model
    try:
        model_path = 'fracture_model.h5'
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Using dummy model for demo purposes")
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
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
    return {'status': 'ok', 'model_loaded': model is not None}


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
            # SIGMOID OUTPUT (Binary Classification)
            # Class 0 = "Fractured", Class 1 = "Not Fractured"
            # Sigmoid > 0.5 means closer to Class 1
            prob = float(preds[0][0])
            
            if prob > 0.5:
                label = 'Not Fractured'
                confidence = round(prob * 100.0, 1)
            else:
                label = 'Fractured'
                confidence = round((1 - prob) * 100.0, 1)
        else:
            # SOFTMAX OUTPUT (Categorical)
            class_probs = preds[0]
            predicted_class = int(np.argmax(class_probs))
            confidence = round(class_probs[predicted_class] * 100.0, 1)
            
            # Map class index to label
            # Class 0 = "Fractured", Class 1 = "Not Fractured"
            labels = ['Fractured', 'Not Fractured']
            label = labels[predicted_class]
        
        logger.info(f"Prediction: {label}, Confidence: {confidence}%")
        return PredictionResponse(prediction=label, confidence=confidence)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
'''


# ============================================================================
# OPTION B: Reversed Order ("not fractured" = Class 0, "fractured" = Class 1)
# Use this if: 'not fractured' folder comes before 'fractured' alphabetically
# ============================================================================

MAIN_PY_OPTION_B = '''
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


model = None


@app.on_event('startup')
async def load_model_on_startup():
    global model
    try:
        model_path = 'fracture_model.h5'
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Using dummy model for demo purposes")
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
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
    return {'status': 'ok', 'model_loaded': model is not None}


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
            # SIGMOID OUTPUT (Binary Classification)
            # Class 0 = "Not Fractured", Class 1 = "Fractured"
            # Sigmoid > 0.5 means closer to Class 1
            prob = float(preds[0][0])
            
            if prob > 0.5:
                label = 'Fractured'
                confidence = round(prob * 100.0, 1)
            else:
                label = 'Not Fractured'
                confidence = round((1 - prob) * 100.0, 1)
        else:
            # SOFTMAX OUTPUT (Categorical)
            class_probs = preds[0]
            predicted_class = int(np.argmax(class_probs))
            confidence = round(class_probs[predicted_class] * 100.0, 1)
            
            # Map class index to label
            # Class 0 = "Not Fractured", Class 1 = "Fractured"
            labels = ['Not Fractured', 'Fractured']
            label = labels[predicted_class]
        
        logger.info(f"Prediction: {label}, Confidence: {confidence}%")
        return PredictionResponse(prediction=label, confidence=confidence)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
'''


if __name__ == '__main__':
    print("=" * 80)
    print("CHOOSE THE CORRECT OPTION BASED ON YOUR MODEL'S CLASS ORDER")
    print("=" * 80)
    print("\nRun this first to determine your class order:")
    print("  python backend/inspect_model.py")
    print("\nThen choose:")
    print("  Option A: if 'fractured' < 'not fractured' alphabetically")
    print("  Option B: if 'not fractured' < 'fractured' alphabetically")
    print("\nCopy the correct code to backend/main.py")
