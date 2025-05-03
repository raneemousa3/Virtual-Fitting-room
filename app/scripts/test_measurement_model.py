import logging
from pathlib import Path
import sys
import json
import numpy as np

# Add the parent directory to the path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from services.measurement_model import MeasurementModel

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def generate_sample_keypoints() -> np.ndarray:
    """Generate sample pose keypoints for testing"""
    # This is a simplified example - in practice, these would come from pose estimation
    keypoints = np.zeros((17, 3))  # 17 keypoints with x, y, confidence
    
    # Set some example keypoints (normalized coordinates)
    keypoints[5] = [0.5, 0.3, 1.0]  # Right shoulder
    keypoints[6] = [0.5, 0.5, 1.0]  # Right elbow
    keypoints[7] = [0.5, 0.7, 1.0]  # Right wrist
    keypoints[11] = [0.5, 0.4, 1.0]  # Right hip
    keypoints[12] = [0.5, 0.6, 1.0]  # Right knee
    keypoints[13] = [0.5, 0.8, 1.0]  # Right ankle
    
    return keypoints

def main():
    """Main function to test the measurement model"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize model
        logger.info("Initializing measurement model...")
        model = MeasurementModel()
        
        # Load trained model
        logger.info("Loading trained model...")
        model.load_model()
        
        # Generate sample keypoints
        logger.info("Generating sample keypoints...")
        keypoints = generate_sample_keypoints()
        
        # Make prediction
        logger.info("Making prediction...")
        measurements = model.predict_measurements(keypoints)
        
        # Print results
        logger.info("Predicted measurements:")
        for measurement, value in measurements.items():
            logger.info(f"{measurement}: {value:.1f} cm")
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 