# 🩻 AI-Powered Fracture Detection System

An end-to-end AI medical imaging project for detecting bone fractures from X-ray images using Deep Learning and Explainable AI techniques.

The project combines:

* TensorFlow / Keras
* MobileNetV2 Transfer Learning
* FastAPI Backend
* React + TypeScript Frontend
* Grad-CAM Visualization

---

# 🚀 Features

✅ Fracture / Not Fracture Classification
✅ Real-time AI Inference
✅ Modern Medical Dashboard UI
✅ Confidence Score Prediction
✅ Upload & Analyze X-ray Images
✅ React + FastAPI Integration
✅ Deep Learning Transfer Learning Pipeline
✅ Grad-CAM Heatmap Visualization
✅ Explainable AI for Medical Imaging

---

# 🧠 AI Model

The model was built using:

* MobileNetV2
* Transfer Learning
* Binary Classification
* TensorFlow / Keras

## Dataset

Bone Fracture Multi-Region X-ray Dataset

Classes:

* Fractured
* Not Fractured

---

# 🔬 Explainability

Grad-CAM visualization is integrated to provide approximate attention maps showing which image regions influenced the model prediction.

This helps improve interpretability and provides visual insight into the model’s decision-making process for medical X-ray analysis.

---

# 📊 Model Performance

| Metric            | Score |
| ----------------- | ----- |
| Test Accuracy     | ~88%  |
| Recall (Fracture) | ~92%  |
| F1-score          | ~89%  |

---

# 🛠️ Tech Stack

## Frontend

* React
* TypeScript
* CSS Modules
* Axios

## Backend

* FastAPI
* TensorFlow
* Pillow
* NumPy

---

# 📂 Project Structure

```bash
project/
│
├── frontend/
│   ├── src/
│   ├── components/
│   ├── pages/
│   └── services/
│
├── backend/
│   ├── main.py
│   ├── fracture_model.h5
│   ├── gradcam.py
│   └── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Keroadham112/computer-vision-project.git
```

---

# 🖥️ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on:

```bash
http://localhost:5173
```

---

# 🔥 Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs on:

```bash
http://localhost:8000
```

---

# 📤 API Endpoint

## POST `/predict`

Upload an X-ray image and receive prediction results.

### Example Response

```json
{
  "prediction": "Fractured",
  "confidence": 96.4,
  "grad_cam_image": "generated_heatmap.png"
}
```

---

# 🧪 How It Works

1. User uploads an X-ray image
2. Image is sent to FastAPI backend
3. TensorFlow model preprocesses image
4. MobileNetV2 predicts fracture probability
5. Grad-CAM generates attention heatmap
6. Prediction & confidence score returned to frontend
7. Results displayed in dashboard UI

---

# 📸 Screenshots

Add screenshots of:

* Dashboard UI
* Upload Section
* Prediction Results
* Grad-CAM Visualization
* Confusion Matrix

---

# 🔬 Future Improvements

* Fine-tuning MobileNetV2
* EfficientNet implementation
* Improved Grad-CAM localization
* Multi-class fracture detection
* Region-aware fracture analysis
* Fracture localization
* Cloud deployment

---

# ⚠️ Disclaimer

This project is intended for educational and research purposes only.

It is not a certified medical diagnostic tool and should not replace professional medical advice.

---

# 👨‍💻 Author

Developed by:
Kerolos Adham

GitHub:
https://github.com/Keroadham112
