#!/usr/bin/env python3
"""
Hybrid Xception+XGBoost Bone Marrow Cell Classification Model
Combines deep feature extraction with ensemble learning
- Enhanced Xception for deep feature extraction
- XGBoost for final classification
- Multi-modal support (images + tabular features)
- Cross-validation and hyperparameter optimization
- Clinical interpretability features
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# XGBoost and ML imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, mutual_info_classif

# Scientific computing
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("Warning: Seaborn not available - using basic matplotlib plots")

from tqdm import tqdm

print("Hybrid Xception+XGBoost Bone Marrow Classification Model")
print(f"PyTorch Version: {torch.__version__}")
print(f"XGBoost Version: {xgb.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class BoneMarrowDataset(Dataset):
    """
    Dataset for hybrid Xception+XGBoost approach
    Supports both image and tabular feature loading
    """
    def __init__(self, csv_file, image_dir, transform=None, mode='hybrid'):
        print(f"Loading dataset from {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.mode = mode  # 'image_only', 'features_only', 'hybrid'
        
        # Validate required columns
        required_cols = ['image_name', 'label']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract feature columns (exclude metadata)
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['image_name', 'label']]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Dataset loaded: {len(self.df)} samples, {self.num_classes} classes")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Feature columns: {len(self.feature_columns)}")
        print(f"Mode: {mode}")
        
        # Validate feature structure
        self._validate_features()
        
        # Validate data availability
        self._validate_data()
        
    def _validate_features(self):
        """Validate feature structure and quality"""
        if len(self.feature_columns) == 0:
            print("Warning: No feature columns found")
            return
            
        # Check for missing values
        missing_counts = self.df[self.feature_columns].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Warning: Found {missing_counts.sum()} missing values in features")
            
        # Check feature distributions
        numeric_features = self.df[self.feature_columns].select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            feature_stats = numeric_features.describe()
            print(f"Feature statistics: {len(numeric_features.columns)} numeric features")
            print(f"Feature range: [{feature_stats.loc['min'].min():.3f}, {feature_stats.loc['max'].max():.3f}]")
        
    def _validate_data(self):
        """Validate that required files exist"""
        if self.mode in ['image_only', 'hybrid']:
            missing_images = 0
            sample_size = min(100, len(self.df))
            
            for i in range(sample_size):
                image_name = self.df.iloc[i]['image_name']
                image_found = False
                
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    if (self.image_dir / f"{image_name}{ext}").exists():
                        image_found = True
                        break
                        
                if not image_found:
                    missing_images += 1
                    
            if missing_images > 0:
                print(f"Warning: {missing_images}/{sample_size} sample images not found")
                
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        row = self.df.iloc[idx]
        image_name = row['image_name']
        label = row['label_encoded']
        
        sample = {
            'label': torch.tensor(label, dtype=torch.long),
            'image_name': image_name
        }
        
        # Load tabular features if needed
        if self.mode in ['features_only', 'hybrid'] and len(self.feature_columns) > 0:
            features = row[self.feature_columns].values.astype(np.float32)
            # Handle missing values
            features = np.nan_to_num(features, nan=0.0)
            sample['features'] = torch.tensor(features, dtype=torch.float32)
            
        # Load image if needed
        if self.mode in ['image_only', 'hybrid']:
            image = self._load_image(image_name)
            sample['image'] = image
            
        return sample
    
    def _load_image(self, image_name):
        """Load and preprocess image"""
        # Try different extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            potential_path = self.image_dir / f"{image_name}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path:
            try:
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Resize for Xception (299x299 is optimal)
                    image = cv2.resize(image, (299, 299))
                    
                    if self.transform:
                        image = self.transform(image)
                    else:
                        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                    
                    return image
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        
        # Return dummy image if loading fails
        return torch.zeros(3, 299, 299, dtype=torch.float32)


# Call to create_data_loaders with a CSV file and image directory
csv_file = 'test_split.csv'  # Path to your test_split.csv file
image_dir = './images/'  # Path to where your images are stored
batch_size = 2  # For testing with a small batch size

train_loader, test_loader, label_encoder = create_data_loaders(csv_file, image_dir, mode='hybrid', batch_size=batch_size)

# Instantiate your model
num_classes = len(label_encoder.classes_)
model = HybridXceptionXGBoostModel(num_classes=num_classes)

# Run the model on the dummy dataset
model.train(train_loader, val_loader=test_loader)  # Run training and validation using dummy data

# Evaluate the model
predictions, probabilities, labels, image_names = model.predict(test_loader)

# Print evaluation results
print(f"Predictions: {predictions}")
print(f"True labels: {labels}")
