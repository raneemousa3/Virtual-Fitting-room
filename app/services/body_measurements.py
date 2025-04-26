import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, Tuple, List
import math
import os

class BodyMeasurementService:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            enable_segmentation=True
        )

    def _calculate_pixel_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points in pixels"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _calculate_angle(self, point1: Tuple[float, float], point2: Tuple[float, float], point3: Tuple[float, float]) -> float:
        """Calculate angle between three points in degrees"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def _estimate_circumference(self, width: float, depth_factor: float = 1.0) -> float:
        """Estimate circumference from width using depth factor"""
        # Using a more accurate ellipse approximation
        # C ≈ π * (3(a + b) - sqrt((3a + b)(a + 3b)))
        # where a = width/2 and b = width/2 * depth_factor
        a = width / 2
        b = a * depth_factor
        return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

    def _estimate_real_world_measurements(self, pixel_measurements: Dict[str, float], height_cm: float) -> Dict[str, float]:
        """Convert pixel measurements to real-world measurements using height as reference."""
        # Get shoulder width in pixels as reference
        shoulder_width_pixels = pixel_measurements['shoulder_width']
        
        # Calculate pixel to cm ratio using shoulder width
        # Average shoulder width is approximately 5% of height (reduced from 10%)
        expected_shoulder_width_cm = height_cm * 0.05
        pixel_to_cm_ratio = expected_shoulder_width_cm / shoulder_width_pixels
        
        # Calculate real-world measurements with adjusted scaling factors
        real_measurements = {}
        
        # Shoulder width (already scaled correctly)
        real_measurements['shoulder_width_cm'] = shoulder_width_pixels * pixel_to_cm_ratio
        
        # Chest circumference (using ellipse approximation)
        chest_width = pixel_measurements['chest_width']
        chest_depth = chest_width * 0.2  # Reduced from 0.3 to 0.2
        real_measurements['chest_cm'] = self._estimate_circumference(chest_width, chest_depth) * pixel_to_cm_ratio
        
        # Waist circumference (using ellipse approximation)
        waist_width = pixel_measurements['waist_width']
        waist_depth = waist_width * 0.25  # Reduced from 0.4 to 0.25
        real_measurements['waist_cm'] = self._estimate_circumference(waist_width, waist_depth) * pixel_to_cm_ratio
        
        # Hip circumference (using ellipse approximation)
        hip_width = pixel_measurements['hip_width']
        hip_depth = hip_width * 0.25  # Reduced from 0.35 to 0.25
        real_measurements['hips_cm'] = self._estimate_circumference(hip_width, hip_depth) * pixel_to_cm_ratio
        
        # Length measurements (direct scaling with additional reduction)
        real_measurements['sleeve_cm'] = pixel_measurements['sleeve_length'] * pixel_to_cm_ratio * 0.2  # Reduced from 0.3 to 0.2
        real_measurements['inseam_cm'] = pixel_measurements['inseam'] * pixel_to_cm_ratio * 0.2  # Reduced from 0.3 to 0.2
        
        return real_measurements

    def visualize_landmarks(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Draw pose landmarks on the image"""
        # Create a copy of the image to draw on
        annotated_image = image.copy()
        
        # Draw the pose landmarks
        self.mp_drawing.draw_landmarks(
            annotated_image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        return annotated_image

    def get_body_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Get body segmentation mask from the image"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if results.segmentation_mask is not None:
            # Convert segmentation mask to uint8
            mask = (results.segmentation_mask * 255).astype(np.uint8)
            return mask
        return None

    async def process_image(self, image_path: str, height_cm: float = None, save_visualization: bool = True) -> Dict:
        """Process an image and return body measurements"""
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at path: {image_path}")

            # Convert to RGB (MediaPipe requires RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]

            # Process the image
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                raise ValueError("No pose detected in the image. Please ensure your full body is visible and try again.")

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Calculate pose angle for better scaling
            left_shoulder = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)
            right_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)
            left_hip = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * image_width,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * image_height)
            
            pose_angle = self._calculate_angle(left_shoulder, left_hip, (left_hip[0], left_hip[1] - 100))

            # Calculate shoulder width for scaling
            shoulder_width = self._calculate_pixel_distance(left_shoulder, right_shoulder)

            # Calculate pixel measurements with improved accuracy
            pixel_measurements = {
                "chest_width": shoulder_width * 0.45,  # Reduced from 0.55
                "waist_width": shoulder_width * 0.25,  # Reduced from 0.35
                "hip_width": shoulder_width * 0.35,    # Reduced from 0.45
                "shoulder_width": shoulder_width,
                "sleeve_length": self._calculate_pixel_distance(
                    left_shoulder,
                    (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x * image_width,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y * image_height)
                ) * 0.2,  # Reduced from 0.3
                "inseam": self._calculate_pixel_distance(
                    left_hip,
                    (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height)
                ) * 0.25  # Reduced from 0.35
            }

            # If height is provided, convert to real-world measurements
            if height_cm:
                # Print debug information
                print(f"DEBUG - Height: {height_cm} cm, Image height: {image_height} pixels")
                print(f"DEBUG - Shoulder width: {shoulder_width} pixels")
                print(f"DEBUG - Pose angle: {pose_angle} degrees")
                
                measurements = self._estimate_real_world_measurements(
                    pixel_measurements,
                    height_cm
                )
                
                # Print debug information
                print(f"DEBUG - Real-world measurements: {measurements}")
            else:
                # If no height provided, return pixel measurements
                measurements = pixel_measurements

            # Add height if provided
            if height_cm:
                measurements["height"] = height_cm

            # Save visualization if requested
            if save_visualization:
                # Create visualizations directory if it doesn't exist
                vis_dir = os.path.join(os.path.dirname(image_path), "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                
                # Generate and save landmark visualization
                annotated_image = self.visualize_landmarks(image, results.pose_landmarks)
                vis_path = os.path.join(vis_dir, f"{os.path.basename(image_path)}_landmarks.jpg")
                cv2.imwrite(vis_path, annotated_image)
                
                # Generate and save segmentation mask
                segmentation_mask = self.get_body_segmentation(image)
                if segmentation_mask is not None:
                    mask_path = os.path.join(vis_dir, f"{os.path.basename(image_path)}_mask.jpg")
                    cv2.imwrite(mask_path, segmentation_mask)
                
                # Add visualization paths to the response
                measurements["visualization"] = {
                    "landmarks": vis_path,
                    "segmentation": mask_path if segmentation_mask is not None else None
                }

            return measurements
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    def get_fit_rating(self, measurements: Dict[str, float], garment_size: Dict[str, float]) -> str:
        """Determine fit rating based on measurements and garment size"""
        # Calculate percentage differences
        chest_diff = abs(measurements["chest"] - garment_size["chest"]) / garment_size["chest"] * 100
        waist_diff = abs(measurements["waist"] - garment_size["waist"]) / garment_size["waist"] * 100
        hips_diff = abs(measurements["hips"] - garment_size["hips"]) / garment_size["hips"] * 100

        # More sophisticated fit rating based on percentage differences
        if chest_diff < 3 and waist_diff < 3 and hips_diff < 3:
            return "Perfect fit"
        elif chest_diff < 5 and waist_diff < 5 and hips_diff < 5:
            return "Good fit"
        elif chest_diff < 8 and waist_diff < 8 and hips_diff < 8:
            return "Slightly loose"
        else:
            return "Not a good fit"

    def detect_multiple_poses(self, image):
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            enable_segmentation=True
        )

    def get_body_segmentation(self, image):
        results = self.pose.process(image)
        if results.segmentation_mask is not None:
            return results.segmentation_mask 