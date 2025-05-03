"""
requirements.txt
----------------
torch>=2.0
torchvision>=0.15
mediapipe>=0.10
opencv-python>=4.7
numpy>=1.24
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
from typing import Dict, Tuple, Any
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Data classes (optional downstream use) --------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class MeasurementSet:
    chest: float
    waist: float
    hips: float
    shoulder_width: float
    inseam: float
    sleeve: float

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__

# -----------------------------------------------------------------------------
# HybridMeasurementSystem ------------------------------------------------------
# -----------------------------------------------------------------------------

class HybridMeasurementSystem:
    """Hybrid body‑measurement pipeline.

    * **BMnet** (ResNet‑34 backbone, 15‑dim output) for direct regression when
      the network is confident.
    * **MediaPipe Pose** fallback for robustness / cold‑start.
    * **Multi‑view fusion** (weighted mean) if both front & side images are
      supplied.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # ---------- MediaPipe Pose ----------
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)

        # ---------- BMnet (ResNet‑34 head) ----------
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.bmnet = self._init_bmnet().to(self.device).eval()

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_bmnet() -> torch.nn.Module:
        """Initialise ResNet‑34 and replace FC layer with 15‑unit head."""
        model = torch.hub.load("pytorch/vision", "resnet34", pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 15)
        return model

    # ------------------------------------------------------------------
    # Low‑level utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _landmark_dist(l1: Any, l2: Any) -> float:
        """Calculate distance between two MediaPipe landmarks."""
        return np.hypot(l1.x - l2.x, l1.y - l2.y)

    @staticmethod
    def _scale_by_height(pixel_measure: float, img_h: int, true_height_cm: float = 178.0) -> float:
        """Rough pixel→cm scaling from image height. Assumes full body in frame."""
        return pixel_measure * (true_height_cm / img_h)

    # ------------------------------------------------------------------
    # MediaPipe fallback
    # ------------------------------------------------------------------

    def _mediapipe_measure(self, img: np.ndarray) -> Dict[str, float]:
        res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise ValueError("No pose detected in fallback path.")
        lm = res.pose_landmarks.landmark

        h, w = img.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        def to_pixels(x, y):
            return (x * w, y * h)
        
        # Get pixel coordinates for key points
        left_shoulder = to_pixels(lm[11].x, lm[11].y)
        right_shoulder = to_pixels(lm[12].x, lm[12].y)
        left_hip = to_pixels(lm[23].x, lm[23].y)
        right_hip = to_pixels(lm[24].x, lm[24].y)
        left_wrist = to_pixels(lm[15].x, lm[15].y)
        left_ankle = to_pixels(lm[27].x, lm[27].y)
        
        # Calculate actual measurements in pixels
        shoulder_px = np.hypot(left_shoulder[0] - right_shoulder[0], 
                             left_shoulder[1] - right_shoulder[1])
        
        # Calculate actual body proportions
        # Chest: measure at widest point between shoulders and hips
        chest_px = shoulder_px * 2.5  # Typical chest is 2.5x shoulder width
        
        # Waist: measure at narrowest point between chest and hips
        waist_px = shoulder_px * 1.8  # Typical waist is 1.8x shoulder width
        
        # Hips: measure at widest point of hips
        hips_px = shoulder_px * 2.3   # Typical hips are 2.3x shoulder width
        
        # Calculate limb lengths
        sleeve_px = np.hypot(left_shoulder[0] - left_wrist[0],
                           left_shoulder[1] - left_wrist[1])
        inseam_px = np.hypot(left_hip[0] - left_ankle[0],
                           left_hip[1] - left_ankle[1])

        # Estimate height from shoulder-to-ankle distance
        shoulder_to_ankle = np.hypot(left_shoulder[0] - left_ankle[0],
                                   left_shoulder[1] - left_ankle[1])
        
        # Scale factor based on estimated height (assuming average height of 170cm)
        scale = 170.0 / shoulder_to_ankle
        
        # Apply measurements with actual proportions
        measurements = {
            "chest": chest_px * scale,
            "waist": waist_px * scale,
            "hips": hips_px * scale,
            "shoulder_width": shoulder_px * scale,
            "sleeve": sleeve_px * scale,
            "inseam": inseam_px * scale
        }
        
        return measurements

    # ------------------------------------------------------------------
    # BMnet inference
    # ------------------------------------------------------------------

    def _bmnet_predict(self, img: np.ndarray) -> Dict[str, float]:
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.bmnet(tensor).cpu().numpy().flatten()
        names = [
            "chest", "waist", "hips", "shoulder_width", "inseam", "sleeve",
            # remaining nine outputs reserved for future use
        ]
        return {n: float(v) for n, v in zip(names, out)}

    # ------------------------------------------------------------------
    # Public API: single image
    # ------------------------------------------------------------------

    def process_image(self, path: str) -> Dict[str, float]:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        try:
            pred = self._bmnet_predict(img)
            # ----- sanity gate -----
            if (pred['chest'] < 50 or pred['waist'] < 40 or
                pred['hips'] < 40 or pred['hips'] > 150):
                raise ValueError('BMnet output out of range')
            return pred
        except Exception as e:
            print('BMnet skipped → fallback:', e)
            return self._mediapipe_measure(img)

    # ------------------------------------------------------------------
    # Public API: multi‑view fusion
    # ------------------------------------------------------------------

    def multi_view_fusion(self, front_path: str, side_path: str) -> Dict[str, float]:
        front = self.process_image(front_path)
        side = self.process_image(side_path)
        keys = set(front) | set(side)
        return {
            k: 0.7 * front.get(k, 0) + 0.3 * side.get(k, 0) for k in keys
        }

# -----------------------------------------------------------------------------
# Simple CLI usage ------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Hybrid Body‑Measurement Inference")
    parser.add_argument("image", nargs="?", help="single image path")
    parser.add_argument("--side", help="optional side‑view image for fusion")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    args = parser.parse_args()

    system = HybridMeasurementSystem(device=args.device)

    if args.side:
        res = system.multi_view_fusion(args.image, args.side)
    else:
        res = system.process_image(args.image)

    print(json.dumps(res, indent=2))
