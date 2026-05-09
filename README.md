# Object Detection Application

A complete production-ready object detection application with ML training, tracking, Docker, FastAPI serving, CI/CD, and deployment.

## Architecture

- **Backend**: FastAPI with YOLOv8 model for object detection
- **Frontend**: React app for uploading images and displaying results
- **ML Tracking**: MLflow for experiment tracking
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: Render

## Local Setup

### Backend

1. Navigate to `backend/` directory
2. Create venv: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Run training: `python train.py`
6. Start server: `uvicorn main:app --reload`

### Frontend

1. Navigate to `frontend/` directory
2. Install: `npm install`
3. Start: `npm start`

## Docker

Build: `docker build -t object-detection .`

Run: `docker run -p 8000:8000 object-detection`

## MLflow

View experiments: `mlflow ui` (from backend directory)

## Deployment on Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Select the repository and branch
4. Set build command: (leave default or `pip install -r backend/requirements.txt`)
5. Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. For frontend, deploy separately as Static Site on Render, build command: `cd frontend && npm install && npm run build`, publish directory: `frontend/build`

## API

POST `/detect` - Upload image file, returns annotated image with bounding boxes