import io
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mnist_api")

MODEL_PATH = "mnist_model.keras"
IMAGE_SIZE = (28, 28)
INPUT_SHAPE = (1, 784)

ml_models = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ml_models["digit_classifier"] = load_model(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}") from e
    yield
    ml_models.clear()


app = FastAPI(
    title="MNIST Digit Classifier API",
    version="1.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    filename: str
    predicted_digit: int
    confidence: float


class HealthResponse(BaseModel):
    message: str


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("L")
    except UnidentifiedImageError as e:
        raise ValueError("File is not a valid image.") from e

    image = image.resize(IMAGE_SIZE)
    array = np.array(image, dtype="float32") / 255.0
    return array.reshape(INPUT_SHAPE)


@app.get("/", response_model=HealthResponse)
def read_root():
    return HealthResponse(
        message=(
            "Welcome to the MNIST Digit Classification API. "
            "The service is running successfully. "
            "Use the POST /predict endpoint to classify handwritten digit images."
        )
    )

@app.post("/predict/", response_model=PredictionResponse)
async def predict_digit(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image_array = preprocess_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        model = ml_models["digit_classifier"]
        prediction_probs = model.predict(image_array, verbose=0)
        predicted_class = int(np.argmax(prediction_probs))
        confidence = float(np.max(prediction_probs))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    return PredictionResponse(
        filename=file.filename or "unknown",
        predicted_digit=predicted_class,
        confidence=round(confidence, 4),
    )
