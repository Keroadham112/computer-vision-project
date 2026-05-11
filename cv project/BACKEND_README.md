# FastAPI Fracture Detection Backend

Backend server for the AI-powered fracture detection web app.

## Features

- **TensorFlow Model Loading**: Loads `fracture_model.h5` once on startup
- **Image Preprocessing**: Resizes to 224x224 and normalizes pixel values
- **FastAPI REST API**: Fast and modern Python web framework
- **CORS Enabled**: Supports React development servers
- **Health Check**: `/health` endpoint to verify server status
- **Error Handling**: Comprehensive error messages and logging

## Setup

### 1. Create Python Virtual Environment

```bash
cd "D:\computer vision project\cv project"
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Place Model File

Place your trained model at the project root:
```
D:\computer vision project\cv project\fracture_model.h5
```

If the model file is not found, the backend will start with a dummy model for testing.

### 4. Start Backend Server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at: **http://localhost:8000**

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Prediction

```bash
POST /predict
```

**Request**: Form data with `file` (image file)

**Response**:
```json
{
  "prediction": "Fractured",
  "confidence": 92.4
}
```

**Example with cURL**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/xray.jpg"
```

## Model Output Format

The backend handles two model output formats:

1. **Single output (sigmoid)**: Shape `(batch, 1)`
   - Returns probability of fracture

2. **Two outputs (softmax)**: Shape `(batch, 2)`
   - Assumes `[not_fractured, fractured]`
   - Uses last class probability

Decision threshold: **0.5**
- `prob >= 0.5` → Fractured
- `prob < 0.5` → Not Fractured

## Image Preprocessing

- **Input**: Any image format (JPG, PNG, etc.)
- **Conversion**: RGB (3 channels)
- **Resize**: 224x224 pixels
- **Normalization**: Pixel values divided by 255.0
- **Output shape**: (1, 224, 224, 3)

## CORS Configuration

Allowed origins:
- `http://localhost:3000`
- `http://localhost:5173-5179` (Vite dev servers)
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

## Troubleshooting

### Model not found error
- Ensure `fracture_model.h5` is in the `cv project` root directory
- Backend will use a dummy model if file is missing (for testing)

### CORS error in frontend
- Make sure backend is running on port 8000
- Check frontend `VITE_API_BASE` environment variable

### TensorFlow issues
- Ensure TensorFlow version matches your model
- For older models: `pip install tensorflow==2.13.0`

## Development

To run with auto-reload:
```bash
uvicorn backend.main:app --reload
```

To run in production:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Logs

The backend logs:
- Model loading status
- Prediction results
- Errors and exceptions

Check console output for detailed information.
