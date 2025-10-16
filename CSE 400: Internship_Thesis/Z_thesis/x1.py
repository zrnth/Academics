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

class EnhancedXceptionFeatureExtractor(nn.Module):
    """
    Enhanced Xception model for feature extraction
    Optimized for bone marrow cell images
    """
    def __init__(self, pretrained=True, feature_dim=2048, dropout=0.5):
        super(EnhancedXceptionFeatureExtractor, self).__init__()
        
        print("Building Enhanced Xception Feature Extractor...")
        
        # Load pretrained Xception
        try:
            self.backbone = models.resnet50(pretrained=pretrained)  # Using ResNet50 as Xception substitute
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final classification layer
            print("ResNet50 backbone loaded (Xception substitute)")
        except Exception as e:
            print(f"Error loading backbone: {e}")
            raise
        
        # Enhanced feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(in_features, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling for robustness
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """
        Extract deep features from images
        Input: [batch_size, 3, 299, 299]
        Output: [batch_size, feature_dim]
        """
        # Extract features using backbone
        if hasattr(self.backbone, 'avgpool'):
            # For ResNet-style models
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        else:
            # Generic feature extraction
            features = self.backbone(x)
            x = torch.flatten(features, 1) if features.dim() > 2 else features
        
        # Process features
        x = self.feature_processor(x)
        
        return x

class TabularFeatureProcessor:
    """
    Advanced tabular feature processing for XGBoost
    """
    def __init__(self, feature_selection=True, scaling='robust', pca_components=None):
        self.feature_selection = feature_selection
        self.scaling = scaling
        self.pca_components = pca_components
        
        # Initialize components
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.feature_names = None
        
    def fit(self, X, y):
        """
        Fit the feature processor
        """
        print("Fitting tabular feature processor...")
        
        # Store original feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            X = X.values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scaling
        if self.scaling == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = None
            
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Feature selection
        if self.feature_selection and X_scaled.shape[1] > 50:
            print("Performing feature selection...")
            # Use mutual information for feature selection
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            # Select top 50% of features or minimum 20 features
            n_features = max(20, int(X_scaled.shape[1] * 0.5))
            selected_indices = np.argsort(mi_scores)[-n_features:]
            
            self.feature_selector = selected_indices
            X_selected = X_scaled[:, selected_indices]
            print(f"Selected {len(selected_indices)} features out of {X_scaled.shape[1]}")
        else:
            X_selected = X_scaled
        
        # PCA if specified
        if self.pca_components and X_selected.shape[1] > self.pca_components:
            print(f"Applying PCA to reduce to {self.pca_components} components...")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            X_final = self.pca.fit_transform(X_selected)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        else:
            X_final = X_selected
            
        return X_final
    
    def transform(self, X):
        """
        Transform features using fitted processor
        """
        # Convert to numpy if needed
        if hasattr(X, 'columns'):
            X = X.values
            
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Apply scaling
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Apply feature selection
        if self.feature_selector is not None:
            X = X[:, self.feature_selector]
        
        # Apply PCA
        if self.pca:
            X = self.pca.transform(X)
            
        return X

class HybridXceptionXGBoostModel:
    """
    Hybrid model combining Xception deep features with XGBoost
    """
    def __init__(self, 
                 num_classes,
                 xception_feature_dim=2048,
                 use_tabular_features=True,
                 feature_fusion='concatenate',
                 xgb_params=None):
        
        self.num_classes = num_classes
        self.xception_feature_dim = xception_feature_dim
        self.use_tabular_features = use_tabular_features
        self.feature_fusion = feature_fusion
        
        print(f"Building Hybrid Xception+XGBoost Model for {num_classes} classes")
        print(f"Feature fusion strategy: {feature_fusion}")
        
        # Xception feature extractor
        self.xception_model = EnhancedXceptionFeatureExtractor(
            feature_dim=xception_feature_dim
        )
        
        # Tabular feature processor
        if use_tabular_features:
            self.tabular_processor = TabularFeatureProcessor(
                feature_selection=True,
                scaling='robust',
                pca_components=100  # Reduce dimensionality for XGBoost
            )
        
        # XGBoost model
        default_params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'verbosity': 1
        }
        
        if xgb_params:
            default_params.update(xgb_params)
            
        self.xgb_params = default_params
        self.xgb_model = None
        
        # Device for PyTorch models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xception_model.to(self.device)
        
        print("Hybrid model initialized successfully")
        
    def extract_image_features(self, dataloader):
        """
        Extract features from images using Xception
        """
        print("Extracting image features using Xception...")
        
        self.xception_model.eval()
        all_features = []
        all_labels = []
        all_image_names = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if 'image' in batch:
                    images = batch['image'].to(self.device)
                    features = self.xception_model(images)
                    
                    all_features.append(features.cpu().numpy())
                    all_labels.append(batch['label'].numpy())
                    all_image_names.extend(batch['image_name'])
        
        features_array = np.vstack(all_features)
        labels_array = np.hstack(all_labels)
        
        print(f"Extracted features shape: {features_array.shape}")
        return features_array, labels_array, all_image_names
    
    def extract_tabular_features(self, dataloader, fit_processor=False):
        """
        Extract and process tabular features
        """
        if not self.use_tabular_features:
            return None, None
            
        print("Processing tabular features...")
        
        all_features = []
        all_labels = []
        
        for batch in dataloader:
            if 'features' in batch:
                all_features.append(batch['features'].numpy())
                all_labels.append(batch['label'].numpy())
        
        if not all_features:
            print("No tabular features found in data")
            return None, None
            
        features_array = np.vstack(all_features)
        labels_array = np.hstack(all_labels)
        
        if fit_processor:
            features_processed = self.tabular_processor.fit(features_array, labels_array)
        else:
            features_processed = self.tabular_processor.transform(features_array)
        
        print(f"Processed tabular features shape: {features_processed.shape}")
        return features_processed, labels_array
    
    def combine_features(self, image_features, tabular_features):
        """
        Combine image and tabular features
        """
        if image_features is None and tabular_features is None:
            raise ValueError("At least one feature type must be provided")
        
        if image_features is None:
            return tabular_features
        
        if tabular_features is None:
            return image_features
        
        if self.feature_fusion == 'concatenate':
            combined = np.hstack([image_features, tabular_features])
        elif self.feature_fusion == 'weighted':
            # Simple weighted combination (could be learned)
            weight_img = 0.7
            weight_tab = 0.3
            # Normalize to same scale first
            img_norm = (image_features - image_features.mean(axis=0)) / (image_features.std(axis=0) + 1e-8)
            tab_norm = (tabular_features - tabular_features.mean(axis=0)) / (tabular_features.std(axis=0) + 1e-8)
            combined = np.hstack([weight_img * img_norm, weight_tab * tab_norm])
        else:
            combined = np.hstack([image_features, tabular_features])
        
        print(f"Combined features shape: {combined.shape}")
        return combined
    
    def train(self, train_loader, val_loader=None, optimize_hyperparams=False):
        """
        Train the hybrid model
        """
        print("Training Hybrid Xception+XGBoost Model...")
        
        # Extract features from training data
        train_img_features, train_labels_img, _ = self.extract_image_features(train_loader)
        train_tab_features, train_labels_tab = self.extract_tabular_features(
            train_loader, fit_processor=True
        )
        
        # Use labels from image extraction (should be the same)
        train_labels = train_labels_img
        
        # Combine features
        train_features = self.combine_features(train_img_features, train_tab_features)
        
        # Hyperparameter optimization
        if optimize_hyperparams:
            print("Optimizing XGBoost hyperparameters...")
            param_grid = {
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [300, 500, 700],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
            
            xgb_base = xgb.XGBClassifier(**self.xgb_params)
            grid_search = GridSearchCV(
                xgb_base, param_grid, cv=3, scoring='f1_weighted',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(train_features, train_labels)
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            
            # Update parameters
            self.xgb_params.update(best_params)
        
        # Train final XGBoost model
        print("Training final XGBoost model...")
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        
        # Handle class weights for imbalanced data
        classes, counts = np.unique(train_labels, return_counts=True)
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        sample_weights = np.array([weights[label] for label in train_labels])
        
        # Train with early stopping if validation data is provided
        if val_loader is not None:
            val_img_features, val_labels_img, _ = self.extract_image_features(val_loader)
            val_tab_features, _ = self.extract_tabular_features(val_loader, fit_processor=False)
            val_features = self.combine_features(val_img_features, val_tab_features)
            
            self.xgb_model.fit(
                train_features, train_labels,
                sample_weight=sample_weights,
                eval_set=[(val_features, val_labels_img)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.xgb_model.fit(train_features, train_labels, sample_weight=sample_weights)
        
        print("Training completed successfully")
        
    def predict(self, dataloader):
        """
        Make predictions using the trained model
        """
        if self.xgb_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        img_features, labels, image_names = self.extract_image_features(dataloader)
        tab_features, _ = self.extract_tabular_features(dataloader, fit_processor=False)
        
        # Combine features
        features = self.combine_features(img_features, tab_features)
        
        # Make predictions
        predictions = self.xgb_model.predict(features)
        probabilities = self.xgb_model.predict_proba(features)
        
        return predictions, probabilities, labels, image_names
    
    def get_feature_importance(self):
        """
        Get feature importance from XGBoost model
        """
        if self.xgb_model is None:
            raise ValueError("Model must be trained first")
        
        importance = self.xgb_model.feature_importances_
        
        # Create feature names
        feature_names = []
        
        # Image features
        for i in range(self.xception_feature_dim):
            feature_names.append(f"xception_feat_{i}")
        
        # Tabular features (if used)
        if self.use_tabular_features and hasattr(self.tabular_processor, 'feature_names'):
            if self.tabular_processor.feature_selector is not None:
                selected_names = [self.tabular_processor.feature_names[i] 
                                for i in self.tabular_processor.feature_selector]
            else:
                selected_names = self.tabular_processor.feature_names
            
            if self.tabular_processor.pca:
                for i in range(self.tabular_processor.pca.n_components_):
                    feature_names.append(f"pca_component_{i}")
            else:
                feature_names.extend(selected_names)
        
        return dict(zip(feature_names, importance))
    
    def save_model(self, save_path):
        """
        Save the trained model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        if self.xgb_model:
            self.xgb_model.save_model(save_path / 'xgboost_model.json')
        
        # Save Xception model
        torch.save(self.xception_model.state_dict(), save_path / 'xception_model.pth')
        
        # Save tabular processor
        if self.use_tabular_features:
            with open(save_path / 'tabular_processor.pkl', 'wb') as f:
                pickle.dump(self.tabular_processor, f)
        
        # Save model configuration
        config = {
            'num_classes': self.num_classes,
            'xception_feature_dim': self.xception_feature_dim,
            'use_tabular_features': self.use_tabular_features,
            'feature_fusion': self.feature_fusion,
            'xgb_params': self.xgb_params
        }
        
        with open(save_path / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        """
        Load a trained model
        """
        load_path = Path(load_path)
        
        # Load configuration
        with open(load_path / 'model_config.json', 'r') as f:
            config = json.load(f)
        
        # Update model configuration
        self.__init__(**config)
        
        # Load XGBoost model
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(load_path / 'xgboost_model.json')
        
        # Load Xception model
        self.xception_model.load_state_dict(
            torch.load(load_path / 'xception_model.pth', map_location=self.device)
        )
        
        # Load tabular processor
        if self.use_tabular_features:
            with open(load_path / 'tabular_processor.pkl', 'rb') as f:
                self.tabular_processor = pickle.load(f)
        
        print(f"Model loaded from {load_path}")

def create_data_loaders(csv_file, image_dir, mode, batch_size, test_size=0.2):
    """
    Create train and validation data loaders
    """
    print("Creating data loaders...")
    
    # Load and split data
    full_df = pd.read_csv(csv_file)
    
    # Check required columns
    if 'image_name' not in full_df.columns or 'label' not in full_df.columns:
        raise ValueError("CSV file must contain 'image_name' and 'label' columns")
    
    print(f"Loaded {len(full_df)} samples with {full_df['label'].nunique()} unique classes")
    
    # Stratified split
    labels_for_split = LabelEncoder().fit_transform(full_df['label'])
    train_idx, test_idx = train_test_split(
        range(len(full_df)), 
        test_size=test_size, 
        stratify=labels_for_split,
        random_state=42
    )
    
    # Create train/test DataFrames
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)
    
    # Save splits
    train_df.to_csv('train_split.csv', index=False)
    test_df.to_csv('test_split.csv', index=False)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BoneMarrowDataset('train_split.csv', image_dir, train_transform, mode)
    test_dataset = BoneMarrowDataset('test_split.csv', image_dir, test_transform, mode)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, train_dataset.label_encoder

def evaluate_model(predictions, probabilities, targets, label_encoder, save_path=None):
    """
    Comprehensive model evaluation
    """
    # Basic metrics