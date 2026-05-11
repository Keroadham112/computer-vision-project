AI Fracture Detection — Complete Setup Guide

This project includes a React + TypeScript frontend and a FastAPI backend for bone fracture detection using TensorFlow.

## Project Structure

```
cv project/
├── src/
│   ├── components/          # React components
│   │   ├── ImageUploader.tsx
│   │   ├── ImageUploader.module.css
│   │   ├── StatusCard.tsx
│   │   └── StatusCard.module.css
│   ├── pages/              # Page components
│   │   ├── Home.tsx
│   │   └── Home.module.css
│   ├── services/           # API client
│   │   └── api.ts
│   ├── types/              # TypeScript types
│   │   └── index.ts
│   ├── App.tsx
│   ├── index.css
│   └── main.tsx
├── backend/                # FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   ├── test_api.py
│   └── .env.example
├── fracture_model.h5       # TensorFlow model (place here)
├── package.json
├── vite.config.ts
├── README_AI_APP.md        # This file
├── BACKEND_README.md       # Backend documentation
└── ...
```

## Frontend Setup

### 1. Install Dependencies

```bash
cd "D:\computer vision project\cv project"
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The frontend will be available at: **http://localhost:5178/** (or similar port)

### 3. Build for Production

```bash
npm run build
npm run preview
```

## Backend Setup

### 1. Create Virtual Environment

```bash
cd "D:\computer vision project\cv project"
python -m venv .venv
.\\.venv\\Scripts\\activate
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Place Model File

Put your trained model file at:
```
D:\computer vision project\cv project\fracture_model.h5
```

### 4. Start Backend Server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: **http://localhost:8000**

## Running Both Frontend and Backend

**Terminal 1 - Frontend:**
```bash
cd "D:\computer vision project\cv project"
npm run dev
```

**Terminal 2 - Backend:**
```bash
cd "D:\computer vision project\cv project"
.\\.venv\\Scripts\\activate
uvicorn backend.main:app --reload
```

## API Endpoints

### Health Check
```
GET http://localhost:8000/health
```

### Predict Fracture
```
POST http://localhost:8000/predict
Content-Type: multipart/form-data

Body: file (image)

Response:
{
  "prediction": "Fractured",
  "confidence": 92.4
}
```

## Testing the API

### With cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/xray.jpg"
```

### With Python

```bash
python backend/test_api.py path/to/xray.jpg
```

## Model Requirements

The TensorFlow model should:
- Accept input shape: `(batch, 224, 224, 3)`
- Output shape: `(batch, 1)` (sigmoid) or `(batch, 2)` (softmax)
- Be saved in `.h5` format

## Features

### Frontend
- 🎨 Dark medical dashboard design
- 📁 Drag & drop X-ray upload
- 🖼️ Image preview
- ⚡ Real-time predictions
- 📊 Confidence visualization
- 📱 Responsive layout
- ✨ Glassmorphism UI with gradients

### Backend
- 🤖 TensorFlow model inference
- 📸 Automatic image preprocessing (224x224, normalized)
- 🚀 FastAPI with async support
- 🔐 CORS enabled for dev servers
- ✅ Health check endpoint
- 📝 Comprehensive logging

## Troubleshooting

### Frontend white screen
- Clear browser cache (F5 or Ctrl+Shift+R)
- Check browser console for errors
- Verify Vite dev server is running

### Backend connection error
- Ensure backend is running on port 8000
- Check CORS configuration if getting cross-origin errors
- Verify model file path is correct

### Model loading error
- Check that `fracture_model.h5` exists in the project root
- Verify TensorFlow version compatibility
- Check model file is not corrupted

### Image processing error
- Use supported formats: JPG, PNG, WebP
- Ensure image is valid (not corrupted)
- Check image file size (keep under 10MB)

## Environment Variables

### Frontend (.env)
```
VITE_API_BASE=http://localhost:8000
```

### Backend (.env)
```
MODEL_PATH=fracture_model.h5
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

## Next Steps

1. Train your fracture detection model (CNN with 224x224 input)
2. Save model as `fracture_model.h5`
3. Place in project root
4. Run frontend and backend
5. Test via web interface or API

## Production Deployment

### Frontend
```bash
npm run build
# Deploy the dist/ folder to a static hosting service
```

### Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Use a reverse proxy (nginx) in front of the backend for production.

## References

- [React Documentation](https://react.dev)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [TensorFlow Documentation](https://www.tensorflow.org)
- [Vite Documentation](https://vitejs.dev)
