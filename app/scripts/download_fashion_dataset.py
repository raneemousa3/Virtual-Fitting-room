import os
import kaggle
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def download_dataset():
    """Download the Fashion AI Dataset from Kaggle"""
    logger = logging.getLogger(__name__)
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/fashion_ai")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the dataset
        logger.info("Downloading Fashion AI Dataset...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'paramaggarwal/fashion-product-images-dataset',
            path=str(data_dir),
            unzip=True
        )
        logger.info("Dataset downloaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False

def process_measurements():
    """Process the measurements from the dataset"""
    logger = logging.getLogger(__name__)
    data_dir = Path("data/fashion_ai")
    
    try:
        # Read the styles.csv file which contains measurements
        styles_df = pd.read_csv(data_dir / "styles.csv")
        
        # Extract relevant measurements
        measurements = []
        for _, row in styles_df.iterrows():
            if pd.notna(row.get('measurements')):
                try:
                    # Parse measurements JSON string
                    size_data = json.loads(row['measurements'])
                    
                    # Standardize measurements
                    standardized = {
                        'chest': float(size_data.get('chest', 0)),
                        'waist': float(size_data.get('waist', 0)),
                        'hips': float(size_data.get('hips', 0)),
                        'shoulder': float(size_data.get('shoulder', 0)),
                        'sleeve': float(size_data.get('sleeve', 0)),
                        'inseam': float(size_data.get('inseam', 0))
                    }
                    
                    measurements.append(standardized)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error processing measurements for row {row['id']}: {str(e)}")
                    continue
        
        # Save processed measurements
        output_file = data_dir / "processed_measurements.json"
        with open(output_file, 'w') as f:
            json.dump(measurements, f, indent=2)
        
        logger.info(f"Processed {len(measurements)} measurements and saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing measurements: {str(e)}")
        return False

def main():
    """Main function to download and process the dataset"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Download dataset
        if not download_dataset():
            logger.error("Failed to download dataset")
            return
        
        # Process measurements
        if not process_measurements():
            logger.error("Failed to process measurements")
            return
        
        logger.info("Dataset download and processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main() 