import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the service
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from services.body_measurements import BodyMeasurementService

def display_image(title, image):
    """Display an image with a title"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Initialize the service
    service = BodyMeasurementService()
    
    # Get the path to an image (you'll need to provide your own image path)
    image_path = input("Enter the path to your image: ")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Get the height from the user
    try:
        height_cm = float(input("Enter your height in cm: "))
    except ValueError:
        print("Error: Height must be a number")
        return
    
    # Process the image and get measurements with visualizations
    try:
        measurements = service.process_image(image_path, height_cm, save_visualization=True)
        
        # Print the measurements
        print("\nBody Measurements:")
        for key, value in measurements.items():
            if key != "visualization":
                print(f"{key}: {value:.2f} cm")
        
        # Load and display the visualizations
        if "visualization" in measurements:
            vis_data = measurements["visualization"]
            
            # Display landmark visualization
            if vis_data["landmarks"] and os.path.exists(vis_data["landmarks"]):
                landmark_img = cv2.imread(vis_data["landmarks"])
                display_image("Pose Landmarks", landmark_img)
                print(f"\nLandmark visualization saved to: {vis_data['landmarks']}")
            
            # Display segmentation mask
            if vis_data["segmentation"] and os.path.exists(vis_data["segmentation"]):
                mask_img = cv2.imread(vis_data["segmentation"])
                display_image("Body Segmentation", mask_img)
                print(f"Segmentation mask saved to: {vis_data['segmentation']}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main() 