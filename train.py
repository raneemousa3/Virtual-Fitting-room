import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import models
from typing import Dict, Tuple, Any, List, Optional
import time
import json
from tqdm import tqdm
import pandas as pd
import cv2

# Define the BodyAttention module
class BodyAttention(nn.Module):
    """Unified attention module for body measurements."""
    def __init__(self, in_channels, attention_type="all"):
        super(BodyAttention, self).__init__()
        self.attention_type = attention_type
        
        # Spatial attention (for general features)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
        # Vertical attention (for height, torso)
        self.vertical_conv = nn.Conv2d(in_channels, 1, kernel_size=(15, 1), padding=(7, 0))
        
        # Horizontal attention (for shoulder, waist, hips)
        self.horizontal_conv = nn.Conv2d(in_channels, 1, kernel_size=(1, 15), padding=(0, 7))
        
        # Specialized shoulder attention
        self.shoulder_conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 7), padding=(0, 3))
        self.shoulder_conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=(7, 1), padding=(3, 0))
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attn_maps = []
        
        # Apply spatial attention
        if self.attention_type in ["all", "spatial"]:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            spatial = torch.cat([avg_out, max_out], dim=1)
            spatial = self.sigmoid(self.spatial_conv(spatial))
            attn_maps.append(spatial)
        
        # Apply vertical attention
        if self.attention_type in ["all", "vertical", "height", "torso"]:
            vertical = self.sigmoid(self.vertical_conv(x))
            attn_maps.append(vertical)
        
        # Apply horizontal attention
        if self.attention_type in ["all", "horizontal", "shoulder", "waist", "hip"]:
            horizontal = self.sigmoid(self.horizontal_conv(x))
            attn_maps.append(horizontal)
        
        # Apply shoulder-specific attention
        if self.attention_type in ["all", "shoulder"]:
            shoulder = self.shoulder_conv1(x)
            shoulder = self.relu(shoulder)
            shoulder = self.shoulder_conv2(shoulder)
            shoulder = self.sigmoid(shoulder)
            attn_maps.append(shoulder)
            
        # Combine attention maps (element-wise multiplication)
        result = x
        for attn_map in attn_maps:
            result = result * attn_map
            
        return result

# Define the UnifiedBodyMeasurementModel
class UnifiedBodyMeasurementModel(nn.Module):
    """Unified body measurement model combining features from both implementations."""
    
    def __init__(self):
        super(UnifiedBodyMeasurementModel, self).__init__()
        
        # Define measurement weights for loss function
        self.measurement_weights = {
            'height': 2.0,
            'shoulder-to-crotch': 2.0,
            'waist': 1.5,
            'chest': 1.0,
            'shoulder-breadth': 2.0,
            'hip': 1.0,
            'ankle': 1.0,
            'arm-length': 1.0,
            'bicep': 1.0,
            'calf': 1.0,
            'forearm': 1.0,
            'leg-length': 1.0,
            'thigh': 1.0,
            'wrist': 1.0
        }
        
        # Load pretrained ResNet-34 backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Modify first conv layer to accept 2 channels instead of 3
        new_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize new conv weights by averaging over the RGB channels
        with torch.no_grad():
            new_conv1.weight = nn.Parameter(torch.mean(resnet.conv1.weight, dim=1, keepdim=True).repeat(1, 2, 1, 1))
        
        # Replace first conv layer
        resnet.conv1 = new_conv1
        
        # Create backbone without final layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Camera info processing branch
        self.camera_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Feature refinement with attention
        self.attention_height = BodyAttention(512, attention_type="vertical")
        self.attention_shoulder = BodyAttention(512, attention_type="shoulder")
        self.attention_waist = BodyAttention(512, attention_type="horizontal")
        self.attention_general = BodyAttention(512, attention_type="spatial")
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Prediction branches
        # Height and vertical measurements branch
        self.vertical_fc = nn.Sequential(
            nn.Linear(512 + 128, 256),  # 512 from backbone + 128 from camera encoder
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # height, torso-length
        )
        
        # Torso measurements branch
        self.torso_fc = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # waist, chest, shoulder
        )
        
        # Limb measurements branch
        self.limb_fc = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 9)  # Other measurements
        )
        
    def forward(self, x, camera_info):
        # Process through backbone
        x = self.backbone(x)
        
        # Apply attention mechanisms
        x_height = self.attention_height(x)
        x_shoulder = self.attention_shoulder(x)
        x_waist = self.attention_waist(x)
        x_general = self.attention_general(x)
        
        # Pool features
        x_height = self.adaptive_pool(x_height).flatten(1)
        x_shoulder = self.adaptive_pool(x_shoulder).flatten(1)
        x_waist = self.adaptive_pool(x_waist).flatten(1)
        x_general = self.adaptive_pool(x_general).flatten(1)
        
        # Process camera info
        camera_features = self.camera_encoder(camera_info)
        
        # Combine with camera features
        height_features = torch.cat([x_height, camera_features], dim=1)
        torso_features = torch.cat([x_shoulder + x_waist, camera_features], dim=1)
        limb_features = torch.cat([x_general, camera_features], dim=1)
        
        # Get predictions from each branch
        height_preds = self.vertical_fc(height_features)
        torso_preds = self.torso_fc(torso_features)
        limb_preds = self.limb_fc(limb_features)
        
        # Combine all predictions
        measurements = torch.cat([height_preds, torso_preds, limb_preds], dim=1)
        
        return measurements

# Custom dataset for BodyM
class BodyMDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load measurements data
        measurements_df = pd.read_csv(os.path.join(data_dir, 'measurements.csv'))
        mapping_df = pd.read_csv(os.path.join(data_dir, 'subject_to_photo_map.csv'))
        
        # Merge measurements with photo mapping
        self.data = pd.merge(measurements_df, mapping_df, on='subject_id')
        
        # Convert measurements to list format
        self.measurement_columns = [
            'height', 'shoulder-to-crotch', 'waist', 'chest', 'shoulder-breadth',
            'hip', 'ankle', 'arm-length', 'bicep', 'calf', 'forearm',
            'leg-length', 'thigh', 'wrist'
        ]
        
        # Default camera info (can be adjusted)
        self.camera_info = {
            'focal_length': 50.0,  # mm
            'sensor_height': 24.0,  # mm
            'image_height': 256.0   # pixels
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load images
        mask_path = os.path.join(self.data_dir, 'mask', f'{row["photo_id"]}.png')
        mask_left_path = os.path.join(self.data_dir, 'mask_left', f'{row["photo_id"]}.png')
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_left = cv2.imread(mask_left_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None or mask_left is None:
            raise ValueError(f"Could not load images for photo_id {row['photo_id']}")
        
        # Resize images
        mask = cv2.resize(mask, (256, 256))
        mask_left = cv2.resize(mask_left, (256, 256))
        
        # Stack images
        images = np.stack([mask, mask_left], axis=0)
        images = images.astype(np.float32) / 255.0
        
        # Get measurements
        measurements = row[self.measurement_columns].values.astype(np.float32)
        
        # Get camera info
        camera_info = np.array([
            self.camera_info['focal_length'],
            self.camera_info['sensor_height'],
            self.camera_info['image_height']
        ], dtype=np.float32)
        
        return torch.FloatTensor(images), torch.FloatTensor(measurements), torch.FloatTensor(camera_info)

# Custom weighted MSE loss
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
        
    def forward(self, pred, target):
        squared_diff = (pred - target) ** 2
        weighted_squared_diff = squared_diff * self.weights
        return weighted_squared_diff.mean()

# Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            
            # Iterate over data
            for inputs, targets, camera_info in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                targets = targets.to(device)
                camera_info = camera_info.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, camera_info)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict().copy()
            
            history[f'{phase}_loss'].append(epoch_loss)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_measurements(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets, camera_info in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            camera_info = camera_info.to(device)
            
            outputs = model(inputs, camera_info)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics for each measurement
    measurement_names = [
        'height', 'shoulder-to-crotch', 'waist', 'chest', 'shoulder-breadth',
        'hip', 'ankle', 'arm-length', 'bicep', 'calf', 'forearm',
        'leg-length', 'thigh', 'wrist'
    ]
    
    results = {}
    for i, name in enumerate(measurement_names):
        pred = all_preds[:, i]
        target = all_targets[:, i]
        
        # Calculate metrics
        mae = np.mean(np.abs(pred - target))
        mse = np.mean((pred - target) ** 2)
        rmse = np.sqrt(mse)
        mean_target = np.mean(target)
        mean_pred = np.mean(pred)
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'Mean Target': mean_target,
            'Mean Prediction': mean_pred,
            'Error %': (mae / mean_target) * 100
        }
    
    return results

# Main function to run training
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = BodyMDataset(
        data_dir='data/raw/bodym/testA',
        split='train',
        transform=None  # No need for transforms during evaluation
    )
    
    # Split into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloader for validation
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize model
    model = UnifiedBodyMeasurementModel()
    model = model.to(device)
    
    # Load the trained model
    print("Loading trained model...")
    model.load_state_dict(torch.load('body_measurement_model.pth', map_location=device))
    
    # Evaluate model
    print("\nEvaluating model performance on individual measurements:")
    results = evaluate_measurements(model, val_loader, device)
    
    # Print results in a formatted table
    print("\nMeasurement-wise Performance:")
    print("-" * 80)
    print(f"{'Measurement':<20} {'MAE':<10} {'RMSE':<10} {'Error %':<10} {'Mean Target':<15} {'Mean Pred':<15}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['MAE']:<10.2f} {metrics['RMSE']:<10.2f} "
              f"{metrics['Error %']:<10.2f} {metrics['Mean Target']:<15.2f} {metrics['Mean Prediction']:<15.2f}")

if __name__ == "__main__":
    main()
