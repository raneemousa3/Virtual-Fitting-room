from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
from datetime import datetime
from typing import Dict, Optional
import uuid
import json
from app.services.body_measurements import HybridMeasurementSystem as BodyMeasurementService

router = APIRouter()
measurement_service = BodyMeasurementService(device="cpu")  # Use CPU for now, can be changed to "cuda" if GPU available

# Ensure upload directory exists
UPLOAD_DIR = "static/images/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/measurements/upload")
async def upload_image_and_get_measurements(
    file: UploadFile = File(...),
    height: float = Form(...),
    weight: Optional[float] = Form(None),
    garment_size: Optional[str] = Form(None)
):
    try:
        print(f"DEBUG: Received file: {file.filename}, height: {height}")
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Validate height
        if height < 100 or height > 250:
            raise HTTPException(
                status_code=400, 
                detail="Height must be between 100 and 250 cm"
            )

        # Validate file size (max 10MB)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        if file_size > 10 * 1024 * 1024:  # 10MB in bytes
            raise HTTPException(
                status_code=400,
                detail="File size must be less than 10MB"
            )

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{timestamp}_{unique_id}.jpg"
        file_path = os.path.join(UPLOAD_DIR, new_filename)
        print(f"DEBUG: Saving file to: {file_path}")

        # Save the file
        with open(file_path, "wb") as f:
            f.write(content)

        # Process the image
        print(f"DEBUG: Processing image with height: {height}")
        try:
            measurements = measurement_service.process_image(file_path)
            print(f"DEBUG: Got measurements: {measurements}")
            
            # Convert measurements to the expected format
            formatted_measurements = {
                "chest_cm": measurements.get("chest", 0),
                "waist_cm": measurements.get("waist", 0),
                "hips_cm": measurements.get("hips", 0),
                "shoulder_width_cm": measurements.get("shoulder_width", 0),
                "sleeve_cm": measurements.get("sleeve", 0),
                "inseam_cm": measurements.get("inseam", 0)
            }
            
            # Add visualization path if available
            if "visualization" in measurements:
                formatted_measurements["visualization"] = measurements["visualization"]
            
            print(f"DEBUG: Measurement keys: {list(formatted_measurements.keys())}")
            
            # Prepare response
            response = {
                "status": "success",
                "message": "Image processed successfully",
                "measurements": formatted_measurements,
                "fit_rating": None,  # This can be implemented later
                "image_path": f"/static/images/uploads/{os.path.basename(file_path)}"
            }
            
            print(f"DEBUG: Sending response: {response}")
            print(f"DEBUG: Response measurement keys: {list(formatted_measurements.keys())}")
            
            return JSONResponse(content=response)
            
        except Exception as e:
            print(f"DEBUG: Error processing image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing image: {str(e)}"
            )

    except HTTPException as he:
        print(f"DEBUG: HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        print(f"DEBUG: Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 