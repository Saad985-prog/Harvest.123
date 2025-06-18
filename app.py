from fastapi import FastAPI, UploadFile, File, Request
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import gdown
import os

app = FastAPI(
    title="Harvest Readiness Classifier",
    description="API for classifying if tomatoes or bananas are ready for harvest"
)

templates = Jinja2Templates(directory="templates")

def download_model(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

TOMATO_MODEL_PATH = "tomato_ready_model10.h5"
download_model("1DhXwbRhjWV2qpBDD0D3AmVSIufqlgP9R", TOMATO_MODEL_PATH)
tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)
tomato_img_size = (224, 224)

BANANA_MODEL_PATH = "banana_ripeness_model20.h5"
download_model("1DYOcwfXVVSMy1yYQWAnSGfx_2uAwZxGA", BANANA_MODEL_PATH)
banana_model = tf.keras.models.load_model(BANANA_MODEL_PATH)
banana_img_size = (150, 150)

def preprocess_image(image_bytes, target_size):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.post("/predict/tomato")
async def predict_tomato(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents, tomato_img_size)
        prediction = tomato_model.predict(processed_image)[0][0]
        ready = bool(prediction > 0.5)
        return JSONResponse(content={
            "ready_for_harvest": ready,
            "crop": "tomato",
            "confidence": float(prediction),
            "model_input_size": tomato_img_size
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict/banana")
async def predict_banana(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents, banana_img_size)
        prediction = banana_model.predict(processed_image)[0][0]
        ready = bool(prediction > 0.5)
        return JSONResponse(content={
            "ready_for_harvest": ready,
            "crop": "banana",
            "confidence": float(prediction),
            "model_input_size": banana_img_size
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
async def root():
    return {
        "message": "Harvest Readiness Classifier API",
        "endpoints": {
            "tomato": {
                "path": "/predict/tomato",
                "input_size": tomato_img_size,
                "model": TOMATO_MODEL_PATH
            },
            "banana": {
                "path": "/predict/banana",
                "input_size": banana_img_size,
                "model": BANANA_MODEL_PATH
            }
        }
    }

@app.get("/ui", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
