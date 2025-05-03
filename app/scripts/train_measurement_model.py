import logging
from pathlib import Path
import sys
import json

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

def load_size_chart_data() -> dict:
    """Load the collected size chart data"""
    data_path = Path("data/size_charts/size_charts.json")
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Size chart data not found at {data_path}")
        raise

def main():
    """Main function to train the measurement model"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load size chart data
        logger.info("Loading size chart data...")
        size_chart_data = load_size_chart_data()
        
        # Initialize and train model
        logger.info("Initializing measurement model...")
        model = MeasurementModel()
        
        logger.info("Training model...")
        model.train(size_chart_data)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 