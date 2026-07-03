# 🧠 MNIST Digit Classifier

A Deep Learning and Computer Vision project that classifies handwritten digits (0–9) using a trained TensorFlow/Keras model. The application consists of a **FastAPI backend** for model inference and a **Streamlit frontend** for an interactive user experience.

## 🚀 Live Demo

### Frontend (Streamlit)
🔗 https://mnist-digit-classifiergit-vaakx2eoj4kkqfgubsmebw.streamlit.app/

### Backend API (FastAPI)
🔗 https://memon122-mnist-digit-classifier.hf.space/

---

## ✨ Features

- 🧠 MNIST handwritten digit recognition
- ⚡ FastAPI REST API for inference
- 🎨 Interactive Streamlit frontend
- 📤 Upload handwritten digit images
- 📊 Prediction confidence score
- 🚀 Lightweight and fast inference

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- FastAPI
- Streamlit
- NumPy
- Pillow
- Requests

---

## 📂 Project Structure

```text
.
├── app.py                 # Streamlit Frontend
├── main.py                # FastAPI Backend
├── mnist_model.keras      # Trained TensorFlow Model
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/mnist-digit-classifier.git
cd mnist-digit-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run FastAPI

```bash
uvicorn main:app --reload
```

API will be available at:

```
http://localhost:8000
```

Swagger Documentation:

```
http://localhost:8000/docs
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

Frontend will be available at:

```
http://localhost:8501
```

---

## 📡 API Endpoints

### Health Check

```http
GET /
```

### Predict Digit

```http
POST /predict
```

---

## 📤 Example Response

```json
{
    "predicted_digit": 7,
    "confidence": 99.84
}
```

---

## 📷 Supported Formats

- PNG
- JPG
- JPEG

---

## 🧠 Model Information

- Dataset: **MNIST**
- Framework: **TensorFlow / Keras**
- Task: Handwritten Digit Classification
- Classes: **0 – 9**

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Abdul Ahad**

If you found this project helpful, consider giving it a ⭐ on GitHub.
