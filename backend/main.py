from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import mlflow

app = FastAPI(title="Object Detection API", description="Upload an image and get object detection results with bounding boxes.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the app starts
model = YOLO('yolov8n.pt')

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("object_detection")


@app.get("/")
async def root():
    return {"message": "Object Detection API is running"}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)

        # Run object detection
        results = model(image_np)

        # Draw bounding boxes and labels
        annotated_image = results[0].plot()

        # Convert to JPEG bytes
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_bytes = io.BytesIO(buffer.tobytes())

        # Log metrics to MLflow (safe way)
        with mlflow.start_run():
            mlflow.log_param("model", "yolov8n")
            mlflow.log_param("input_image_size", str(image_np.shape))
            mlflow.log_metric("detections", len(results[0].boxes))

        return StreamingResponse(annotated_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")