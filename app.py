from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import cv2
import numpy as np
from pathlib import Path
import io
from PIL import Image
from train import UnifiedBodyMeasurementModel
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel
import json
from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import mediapipe as mp
from app.services.body_measurements import HybridMeasurementSystem

app = FastAPI(title="Body Measurement API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Define Pydantic model for camera information
class CameraInfo(BaseModel):
    focal_length: float = 50.0  # mm
    sensor_height: float = 24.0  # mm
    image_height: float = 256.0  # pixels
    distance: float = 2000.0  # mm (2 meters)

# Load model at startup
model = None
measurement_names = [
    'height', 'shoulder-to-crotch', 'waist', 'chest', 'shoulder-breadth',
    'hip', 'ankle', 'arm-length', 'bicep', 'calf', 'forearm',
    'leg-length', 'thigh', 'wrist'
]

# Define measurement ranges in cm
measurement_ranges = {
    'height': (150, 210),    # Typical height range
    'shoulder-to-crotch': (50, 80),  # Torso length
    'waist': (50, 110),      # Waist circumference
    'chest': (70, 130),      # Chest circumference
    'shoulder-breadth': (35, 55),  # Shoulder width
    'hip': (70, 130),        # Hip circumference
    'ankle': (18, 28),       # Ankle circumference
    'arm-length': (50, 80),  # Arm length
    'bicep': (25, 45),       # Bicep circumference
    'calf': (25, 45),        # Calf circumference
    'forearm': (20, 35),     # Forearm circumference
    'leg-length': (65, 95),  # Leg length
    'thigh': (45, 75),       # Thigh circumference
    'wrist': (15, 25)        # Wrist circumference
}

# Add scaling factors for model output
measurement_scales = {
    'height': 1.0,           # Height is usually most accurate
    'shoulder-to-crotch': 0.8,  # Torso measurements
    'waist': 0.8,
    'chest': 0.8,
    'shoulder-breadth': 0.8,
    'hip': 0.8,
    'ankle': 0.7,           # Smaller measurements need less scaling
    'arm-length': 0.8,
    'bicep': 0.7,
    'calf': 0.7,
    'forearm': 0.7,
    'leg-length': 0.8,
    'thigh': 0.7,
    'wrist': 0.7
}

def get_confidence_score(value: float, min_val: float, max_val: float) -> float:
    """Calculate confidence score based on how close the value is to the range center"""
    range_center = (min_val + max_val) / 2
    range_width = max_val - min_val
    distance_from_center = abs(value - range_center)
    confidence = 1 - (distance_from_center / (range_width / 2))
    return max(0, min(1, confidence))

def get_size_category(value: float, min_val: float, max_val: float) -> str:
    """Determine size category based on measurement value"""
    range_width = max_val - min_val
    if value < min_val + range_width * 0.33:
        return "small"
    elif value > max_val - range_width * 0.33:
        return "large"
    else:
        return "medium"

def get_body_type_summary(measurements: Dict[str, float]) -> Dict[str, str]:
    """Generate a summary of body type based on key measurements"""
    chest = measurements['chest']
    waist = measurements['waist']
    hip = measurements['hip']
    height = measurements['height']
    
    # Calculate waist-to-hip ratio
    whr = waist / hip
    
    # Determine body type
    if whr > 0.85:
        body_type = "apple"
    elif whr < 0.75:
        body_type = "pear"
    else:
        body_type = "hourglass"
    
    # Determine size category based on height and measurements
    avg_size = (chest + waist + hip) / 3
    if height < 165:
        size = "petite"
    elif height > 185:
        size = "tall"
    elif avg_size < 85:
        size = "small"
    elif avg_size > 105:
        size = "large"
    else:
        size = "medium"
    
    return {
        "body_type": body_type,
        "size_category": size,
        "waist_to_hip_ratio": round(whr, 2),
        "height_category": "short" if height < 165 else "tall" if height > 185 else "average"
    }

def validate_measurements(measurements: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    """Validate measurements and calculate confidence scores and categories"""
    validated = {}
    confidence_scores = {}
    categories = {}
    
    for name, value in measurements.items():
        # Get range and scale
        min_val, max_val = measurement_ranges[name]
        scale = measurement_scales[name]
        
        # Normalize the raw prediction to a reasonable range
        # First, scale the prediction
        scaled_value = value * scale
        
        # Then, map it to the valid range using a sigmoid-like function
        # This prevents hard clipping while keeping values in a reasonable range
        range_width = max_val - min_val
        normalized_value = min_val + (max_val - min_val) * (1 / (1 + np.exp(-(scaled_value - (min_val + max_val)/2) / (range_width/4))))
        
        validated[name] = float(normalized_value)
        
        # Calculate confidence score based on how close to the center of the range
        range_center = (min_val + max_val) / 2
        distance_from_center = abs(validated[name] - range_center)
        confidence_scores[name] = max(0, min(1, 1 - (distance_from_center / (range_width/2))))
        
        # Determine size category
        if validated[name] < min_val + range_width * 0.33:
            categories[name] = "small"
        elif validated[name] > max_val - range_width * 0.33:
            categories[name] = "large"
        else:
            categories[name] = "medium"
    
    return validated, confidence_scores, categories

@app.on_event("startup")
async def load_model():
    global model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnifiedBodyMeasurementModel()
    model.load_state_dict(torch.load('body_measurement_model.pth', map_location=device))
    model.eval()
    print("Model loaded successfully!")

def preprocess_images(mask_image_bytes, mask_left_image_bytes):
    # Convert bytes to numpy arrays
    mask_array = np.frombuffer(mask_image_bytes, np.uint8)
    mask_left_array = np.frombuffer(mask_left_image_bytes, np.uint8)
    
    # Decode images
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    mask_left = cv2.imdecode(mask_left_array, cv2.IMREAD_GRAYSCALE)
    
    if mask is None or mask_left is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Resize images
    mask = cv2.resize(mask, (256, 256))
    mask_left = cv2.resize(mask_left, (256, 256))
    
    # Stack images
    images = np.stack([mask, mask_left], axis=0)
    images = images.astype(np.float32) / 255.0
    
    return torch.FloatTensor(images).unsqueeze(0)

@app.post("/predict/")
async def predict_measurements(
    mask_image: UploadFile = File(...),
    mask_left_image: UploadFile = File(...),
    camera_info: str = Form(default='{"focal_length": 50.0, "sensor_height": 24.0, "image_height": 256.0}')
):
    """
    Predict body measurements from mask images.
    
    Parameters:
    - mask_image: Front view mask image
    - mask_left_image: Left view mask image
    - camera_info: JSON string containing camera parameters (optional)
        - focal_length: Camera focal length in mm
        - sensor_height: Camera sensor height in mm
        - image_height: Image height in pixels
    
    Returns:
    - Dictionary containing measurements, confidence scores, categories, and body type summary
    """
    try:
        # Parse camera info
        try:
            camera_dict = json.loads(camera_info)
            camera_info_obj = CameraInfo(**camera_dict)
        except json.JSONDecodeError:
            camera_info_obj = CameraInfo()  # Use defaults if parsing fails
        
        # Read image files
        mask_contents = await mask_image.read()
        mask_left_contents = await mask_left_image.read()
        
        # Preprocess images
        images = preprocess_images(mask_contents, mask_left_contents)
        
        # Convert camera info to tensor (only first three parameters)
        camera_tensor = torch.tensor([
            camera_info_obj.focal_length,
            camera_info_obj.sensor_height,
            camera_info_obj.image_height
        ], dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(images, camera_tensor)
            predictions = predictions.cpu().numpy()[0]
        
        # Convert predictions to dictionary
        measurements = {name: float(value) for name, value in zip(measurement_names, predictions)}
        
        # Validate measurements and get confidence scores
        validated_measurements, confidence_scores, categories = validate_measurements(measurements)
        
        # Get body type summary
        body_summary = get_body_type_summary(validated_measurements)
        
        return {
            "measurements": validated_measurements,
            "confidence_scores": confidence_scores,
            "categories": categories,
            "body_summary": body_summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the service is healthy"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 