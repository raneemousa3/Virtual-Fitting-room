import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

class MeasurementModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        
        # Define measurement targets
        self.target_measurements = [
            'chest',
            'waist',
            'hips',
            'shoulder',
            'sleeve',
            'inseam'
        ]
        
        # Define pose keypoints to use
        self.keypoint_indices = {
            'shoulder_left': 11,
            'shoulder_right': 12,
            'hip_left': 23,
            'hip_right': 24,
            'knee_left': 25,
            'knee_right': 26,
            'ankle_left': 27,
            'ankle_right': 28,
            'wrist_left': 15,
            'wrist_right': 16
        }

    def prepare_training_data(self, size_chart_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from size chart data"""
        X = []  # Features (pose keypoints)
        y = []  # Targets (measurements)
        
        for brand, sizes in size_chart_data.items():
            for size, measurements in sizes.items():
                # Skip if missing required measurements
                if not all(m in measurements for m in self.target_measurements):
                    continue
                
                # Generate synthetic pose keypoints based on measurements
                keypoints = self._generate_synthetic_keypoints(measurements)
                
                # Add to training data
                X.append(keypoints)
                y.append([measurements[m] for m in self.target_measurements])
        
        return np.array(X), np.array(y)

    def _generate_synthetic_keypoints(self, measurements: Dict) -> np.ndarray:
        """Generate synthetic pose keypoints from measurements"""
        # Initialize keypoints array (33 points, x,y coordinates)
        keypoints = np.zeros((33, 2))
        
        # Set keypoints based on measurements
        # Shoulder width
        shoulder_width = measurements.get('shoulder', 0)
        keypoints[self.keypoint_indices['shoulder_left']] = [-shoulder_width/2, 0]
        keypoints[self.keypoint_indices['shoulder_right']] = [shoulder_width/2, 0]
        
        # Hip width
        hip_width = measurements.get('hips', 0) / np.pi  # Approximate
        keypoints[self.keypoint_indices['hip_left']] = [-hip_width/2, -measurements.get('height', 170)/2]
        keypoints[self.keypoint_indices['hip_right']] = [hip_width/2, -measurements.get('height', 170)/2]
        
        # Knee positions
        keypoints[self.keypoint_indices['knee_left']] = [-hip_width/2, -measurements.get('height', 170)*0.75]
        keypoints[self.keypoint_indices['knee_right']] = [hip_width/2, -measurements.get('height', 170)*0.75]
        
        # Ankle positions
        keypoints[self.keypoint_indices['ankle_left']] = [-hip_width/2, -measurements.get('height', 170)]
        keypoints[self.keypoint_indices['ankle_right']] = [hip_width/2, -measurements.get('height', 170)]
        
        # Wrist positions (for sleeve length)
        sleeve_length = measurements.get('sleeve', 0)
        keypoints[self.keypoint_indices['wrist_left']] = [-shoulder_width/2 - sleeve_length, 0]
        keypoints[self.keypoint_indices['wrist_right']] = [shoulder_width/2 + sleeve_length, 0]
        
        return keypoints.flatten()

    def train(self, size_chart_data: Dict):
        """Train the measurement model"""
        try:
            # Prepare training data
            X, y = self.prepare_training_data(size_chart_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.logger.info(f"Model training complete:")
            self.logger.info(f"Train R² score: {train_score:.3f}")
            self.logger.info(f"Test R² score: {test_score:.3f}")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise

    def predict_measurements(self, pose_keypoints: np.ndarray) -> Dict[str, float]:
        """Predict measurements from pose keypoints"""
        try:
            if self.model is None:
                self.load_model()
            
            # Reshape and scale keypoints
            keypoints_flat = pose_keypoints.flatten()
            keypoints_scaled = self.scaler.transform(keypoints_flat.reshape(1, -1))
            
            # Make prediction
            predictions = self.model.predict(keypoints_scaled)[0]
            
            # Convert to dictionary
            measurements = {
                measurement: float(value)
                for measurement, value in zip(self.target_measurements, predictions)
            }
            
            return measurements
            
        except Exception as e:
            self.logger.error(f"Error predicting measurements: {e}")
            return {}

    def save_model(self):
        """Save the trained model"""
        try:
            # Save model
            model_path = self.model_dir / "measurement_model.joblib"
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = self.model_dir / "measurement_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def load_model(self):
        """Load the trained model"""
        try:
            # Load model
            model_path = self.model_dir / "measurement_model.joblib"
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = self.model_dir / "measurement_scaler.joblib"
            self.scaler = joblib.load(scaler_path)
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise 