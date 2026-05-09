from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import mlflow
import mlflow.pytorch

app = FastAPI(title="Object Detection API", description="Upload an image and get object detection results with bounding boxes.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = YOLO('yolov8n.pt')  # Using pre-trained YOLOv8 nano model

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("object_detection")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_np = np.array(image)

    # Run inference
    results = model(image_np)

    # Process results
    annotated_image = results[0].plot()

    # Convert back to PIL Image
    annotated_pil = Image.fromarray(annotated_image)

    # Save to bytes
    img_byte_arr = io.BytesIO()
    annotated_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("model", "yolov8n")
        mlflow.log_param("input_image_size", image_np.shape)
        mlflow.log_metric("detections", len(results[0].boxes))
        # Log model
        mlflow.pytorch.log_model(model, "model")

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.get("/")
async def root():
    return {"message": "Object Detection API is running"}