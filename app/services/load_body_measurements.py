from pathlib import Path
import json, cv2     # or `from PIL import Image`
import pandas as pd  # handy for bookkeeping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

# then place kaggle.json (API token) in ~/.kaggle/


# 1️⃣  Point to the dataset root
#    • Kaggle Notebook:   /kaggle/input/body-measurements-dataset
#    • Local machine:     Use the local CSV file
CSV_PATH = Path("/Users/raneemmousa/Downloads/Virtual-Fitting-room-main/body.csv")

# Check if CSV exists
if not CSV_PATH.exists():
    print(f"CSV file not found at {CSV_PATH}")
    print("Please make sure the CSV file is in the correct location.")
    exit(1)

class BodyMeasurementsLoader:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self.validated_data = None
        
    def load_data(self) -> bool:
        """Load and validate the CSV file."""
        if not self.csv_path.exists():
            print(f"CSV file not found at {self.csv_path}")
            return False
            
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Successfully loaded {len(self.df)} records")
            return True
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return False
            
    def preprocess_data(self) -> None:
        """Clean and preprocess the data."""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
            
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[~((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR)))]
            
        print(f"Preprocessed data shape: {self.df.shape}")
        
    def validate_measurements(self) -> Dict[str, List[Tuple[str, str]]]:
        """Validate measurements against expected ranges."""
        validation_results = {
            'warnings': [],
            'errors': []
        }
        
        # Define measurement ranges (in cm)
        ranges = {
            'height': (140, 220),
            'weight': (40, 150),
            'chest': (70, 140),
            'waist': (50, 120),
            'hips': (70, 140)
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in self.df.columns:
                invalid = self.df[col][(self.df[col] < min_val) | (self.df[col] > max_val)]
                if not invalid.empty:
                    validation_results['warnings'].append(
                        (col, f"Found {len(invalid)} measurements outside range [{min_val}, {max_val}]")
                    )
                    
        self.validated_data = self.df.copy()
        return validation_results
        
    def visualize_data(self, save_path: str = None) -> None:
        """Create visualizations of the measurements."""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
            
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Body Measurements Distribution')
        
        # Plot histograms for key measurements
        measurements = ['height', 'weight', 'chest', 'waist']
        for idx, (ax, measurement) in enumerate(zip(axes.flat, measurements)):
            if measurement in self.df.columns:
                sns.histplot(data=self.df, x=measurement, ax=ax)
                ax.set_title(f'{measurement.capitalize()} Distribution')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
    def export_data(self, output_path: str, format: str = 'csv') -> None:
        """Export the processed data."""
        if self.validated_data is None:
            print("No validated data available. Please validate data first.")
            return
            
        try:
            if format.lower() == 'csv':
                self.validated_data.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                self.validated_data.to_json(output_path, orient='records')
            else:
                print(f"Unsupported format: {format}")
                return
                
            print(f"Data exported successfully to {output_path}")
        except Exception as e:
            print(f"Error exporting data: {str(e)}")

def main():
    # Initialize the loader
    loader = BodyMeasurementsLoader("body.csv")
    
    # Load data
    if not loader.load_data():
        return
        
    # Preprocess data
    loader.preprocess_data()
    
    # Validate measurements
    validation_results = loader.validate_measurements()
    
    # Print validation results
    print("\nValidation Results:")
    for category, results in validation_results.items():
        if results:
            print(f"\n{category.capitalize()}:")
            for field, message in results:
                print(f"- {field}: {message}")
                
    # Create visualizations
    loader.visualize_data('measurements_distribution.png')
    
    # Export processed data
    loader.export_data('processed_measurements.csv')

if __name__ == "__main__":
    main()
