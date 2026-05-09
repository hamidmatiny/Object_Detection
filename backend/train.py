import mlflow
import mlflow.pytorch
from ultralytics import YOLO

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("object_detection_training")

with mlflow.start_run():
    # Load pre-trained model
    model = YOLO('yolov8n.pt')

    # Log parameters
    mlflow.log_param("model_type", "YOLOv8n")
    mlflow.log_param("pretrained", True)

    # Log the model
    mlflow.pytorch.log_model(model, "model")

    print("Model logged to MLflow")