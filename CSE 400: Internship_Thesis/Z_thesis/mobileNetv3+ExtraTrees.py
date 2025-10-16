#!/usr/bin/env python3
"""
Complete MobileNetV3 + Extra Trees Hybrid Bone Marrow Cell Classification Model
Advanced hybrid approach combining efficient deep learning with ensemble learning:
- MobileNetV3 for lightweight deep feature extraction from images
- Extra Trees for classification on combined features (deep + handcrafted + embeddings)
- Comprehensive XAI with SHAP and feature importance analysis
- Optimized for large-scale bone marrow datasets (150k+ images) and mobile deployment
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Extra Trees and ML imports
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (classification_report, f1_score, accuracy_score, 
                           confusion_matrix, precision_recall_fscore_support,
                           roc_auc_score, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance

# XAI imports
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# Visualization
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("Warning: Seaborn not available - using basic matplotlib plots")

from tqdm import tqdm
import joblib
from collections import defaultdict
import time

print("Complete MobileNetV3 + Extra Trees Hybrid Model")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class BoneMarrowDataset(Dataset):
    """
    Enhanced dataset for bone marrow cell classification
    Handles images, embeddings, and handcrafted features
    """
    def __init__(self, csv_file, image_dir, embeddings_dir, transform=None, mode='hybrid'):
        print(f"Loading dataset from {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.transform = transform
        self.mode = mode  # 'hybrid', 'features_only', 'images_only'
        
        # Extract feature columns (exclude metadata)
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['image_name', 'label']]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Dataset loaded: {len(self.df)} samples, {self.num_classes} classes")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Mode: {mode}")
        print(f"Handcrafted features: {len(self.feature_columns)}")
        
        # Validate feature structure
        self._validate_feature_structure()
        
        # Cache for performance
        self._image_cache = {}
        self._embedding_cache = {}
        
    def _validate_feature_structure(self):
        """Validate that features match the expected bone marrow structure"""
        expected_regions = ['cell', 'nucleus', 'cytoplasm']
        expected_features_per_region = 48
        
        region_counts = {}
        for region in expected_regions:
            region_features = [col for col in self.feature_columns if col.startswith(f'{region}_')]
            region_counts[region] = len(region_features)
        
        print(f"Feature validation:")
        all_good = True
        for region, count in region_counts.items():
            status = "OK" if count == expected_features_per_region else "WARNING"
            print(f"   {status} {region}: {count} features (expected {expected_features_per_region})")
            if count != expected_features_per_region:
                all_good = False
        
        total_features = sum(region_counts.values())
        status = "OK" if total_features == 144 else "WARNING"
        print(f"   {status} Total: {total_features} features (expected 144)")
        
        if all_good:
            print("Perfect! Feature structure matches the paper exactly.")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single sample with caching for performance"""
        row = self.df.iloc[idx]
        image_name = row['image_name']
        label = row['label_encoded']
        
        sample = {
            'image_name': image_name,
            'label': torch.tensor(label, dtype=torch.long),
            'handcrafted_features': torch.tensor(
                row[self.feature_columns].values.astype(np.float32), 
                dtype=torch.float32
            )
        }
        
        # Load image if needed
        if self.mode in ['hybrid', 'images_only']:
            image = self._load_image_cached(image_name)
            sample['image'] = image
            
        # Load embedding if needed
        if self.mode in ['hybrid', 'embeddings_only']:
            embedding = self._load_embedding_cached(image_name, row)
            sample['embedding'] = embedding
            
        return sample
    
    def _load_image_cached(self, image_name):
        """Load image with caching"""
        if image_name in self._image_cache:
            return self._image_cache[image_name]
        
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
                    # Resize to 224x224 for MobileNetV3
                    image = cv2.resize(image, (224, 224))
                    
                    if self.transform:
                        image = self.transform(image)
                    else:
                        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                    
                    # Cache the processed image
                    self._image_cache[image_name] = image
                    return image
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        
        # Return dummy image if loading fails
        dummy_image = torch.zeros(3, 224, 224, dtype=torch.float32)
        self._image_cache[image_name] = dummy_image
        return dummy_image
    
    def _load_embedding_cached(self, image_name, row):
        """Load embedding with caching and fallback creation"""
        if image_name in self._embedding_cache:
            return self._embedding_cache[image_name]
        
        embedding_path = self.embeddings_dir / f"{image_name}_embedding_3ch.npy"
        
        if embedding_path.exists():
            try:
                embedding = np.load(embedding_path).astype(np.float32)
                if embedding.shape != (12, 12, 3):
                    embedding = self._create_embedding_from_features(row)
            except Exception as e:
                print(f"Error loading embedding {embedding_path}: {e}")
                embedding = self._create_embedding_from_features(row)
        else:
            embedding = self._create_embedding_from_features(row)
        
        # Normalize and cache
        embedding = embedding.astype(np.float32)
        if embedding.std() > 1e-8:
            embedding = (embedding - embedding.mean()) / embedding.std()
        
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        self._embedding_cache[image_name] = embedding_tensor
        return embedding_tensor
    
    def _create_embedding_from_features(self, row):
        """Create 12x12x3 embedding from 144 handcrafted features"""
        features = row[self.feature_columns].values.astype(np.float32)
        
        if len(features) != 144:
            if len(features) < 144:
                padded = np.zeros(144, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            else:
                features = features[:144]
        
        # Split into regions
        cell_features = features[0:48]
        nucleus_features = features[48:96]
        cytoplasm_features = features[96:144]
        
        # Create 12x12x3 embedding
        embedding = np.zeros((12, 12, 3), dtype=np.float32)
        
        for channel, region_features in enumerate([cell_features, nucleus_features, cytoplasm_features]):
            for i in range(12):
                for j in range(12):
                    position_idx = i * 12 + j
                    if position_idx < 48:
                        embedding[i, j, channel] = region_features[position_idx]
                    else:
                        feature_idx = position_idx % 48
                        embedding[i, j, channel] = region_features[feature_idx]
        
        return embedding

class MobileNetV3FeatureExtractor(nn.Module):
    """
    MobileNetV3-based feature extractor optimized for efficiency and bone marrow images
    """
    def __init__(self, feature_dim=512, pretrained=True, model_size='large'):
        super(MobileNetV3FeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        self.model_size = model_size
        
        print(f"Building MobileNetV3-{model_size} Feature Extractor")
        
        # Load pretrained MobileNetV3
        if model_size == 'large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            backbone_features = 960  # MobileNetV3-Large output features
        elif model_size == 'small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            backbone_features = 576  # MobileNetV3-Small output features
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Efficient feature extraction head (inspired by MobileNet philosophy)
        self.feature_head = nn.Sequential(
            # Depthwise separable reduction
            nn.Conv2d(backbone_features, backbone_features, 1, groups=1, bias=False),
            nn.BatchNorm2d(backbone_features),
            nn.Hardswish(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Efficient linear layers
            nn.Linear(backbone_features, feature_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            
            # Final feature layer
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Squeeze-and-Excitation inspired feature refinement
        self.se_block = nn.Sequential(
            nn.Linear(backbone_features, backbone_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_features // 4, backbone_features),
            nn.Hardsigmoid(inplace=True)
        )
        
        # Fine-tuning strategy for MobileNetV3
        self._setup_fine_tuning()
        
        print(f"MobileNetV3-{model_size} Feature Extractor built: {backbone_features} -> {feature_dim}")
        
    def _setup_fine_tuning(self):
        """Setup fine-tuning strategy for MobileNetV3"""
        # Freeze early layers, fine-tune later inverted residual blocks
        for name, param in self.backbone.named_parameters():
            if any(early_layer in name for early_layer in ['features.0', 'features.1', 'features.2']):
                param.requires_grad = False
            elif any(late_layer in name for late_layer in ['features.12', 'features.13', 'features.14', 'features.15']):
                param.requires_grad = True
            else:
                param.requires_grad = True
    
    def forward(self, x):
        """Forward pass through MobileNetV3 with SE attention"""
        # Extract features using MobileNetV3 backbone
        features = self.backbone.features(x)
        
        # Apply global average pooling for SE block
        pooled = nn.AdaptiveAvgPool2d(1)(features)
        pooled_flat = torch.flatten(pooled, 1)
        
        # Apply SE attention
        se_weights = self.se_block(pooled_flat).unsqueeze(-1).unsqueeze(-1)
        attended_features = features * se_weights
        
        # Pass through feature head
        deep_features = self.feature_head(attended_features)
        
        return deep_features

class HybridFeatureExtractor:
    """
    Complete hybrid feature extraction system
    Combines MobileNetV3 deep features with handcrafted features and embeddings
    """
    def __init__(self, deep_feature_dim=512, device='cuda', model_size='large'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.deep_feature_dim = deep_feature_dim
        self.model_size = model_size
        
        # Initialize MobileNetV3
        self.mobilenetv3 = MobileNetV3FeatureExtractor(
            feature_dim=deep_feature_dim,
            pretrained=True,
            model_size=model_size
        ).to(self.device)
        
        # Feature names for interpretability
        self.feature_names = None
        
        print(f"Hybrid Feature Extractor initialized on {self.device}")
        
    def extract_deep_features(self, dataloader, save_path=None):
        """Extract deep features from images using MobileNetV3"""
        print("Extracting deep features using MobileNetV3...")
        
        self.mobilenetv3.eval()
        all_deep_features = []
        all_handcrafted_features = []
        all_embeddings = []
        all_labels = []
        all_image_names = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                try:
                    # Move data to device
                    images = batch['image'].to(self.device)
                    handcrafted = batch['handcrafted_features'].numpy()
                    labels = batch['label'].numpy()
                    image_names = batch['image_name']
                    
                    # Extract deep features
                    deep_features = self.mobilenetv3(images)
                    deep_features_np = deep_features.cpu().numpy()
                    
                    # Collect all features
                    all_deep_features.append(deep_features_np)
                    all_handcrafted_features.append(handcrafted)
                    all_labels.extend(labels)
                    all_image_names.extend(image_names)
                    
                    # Handle embeddings if available
                    if 'embedding' in batch:
                        embeddings = batch['embedding'].numpy()
                        # Flatten embeddings: (batch, 12, 12, 3) -> (batch, 432)
                        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
                        all_embeddings.append(embeddings_flat)
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Combine all features
        combined_deep_features = np.vstack(all_deep_features)
        combined_handcrafted = np.vstack(all_handcrafted_features)
        combined_labels = np.array(all_labels)
        
        print(f"Extracted features:")
        print(f"   Deep features: {combined_deep_features.shape}")
        print(f"   Handcrafted features: {combined_handcrafted.shape}")
        
        # Combine embeddings if available
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            print(f"   Embeddings: {combined_embeddings.shape}")
            
            # Combine all feature types
            final_features = np.concatenate([
                combined_deep_features,
                combined_handcrafted,
                combined_embeddings
            ], axis=1)
        else:
            # Combine only deep and handcrafted features
            final_features = np.concatenate([
                combined_deep_features,
                combined_handcrafted
            ], axis=1)
            combined_embeddings = None
        
        # Create feature names
        self._create_feature_names(
            deep_dim=combined_deep_features.shape[1],
            handcrafted_dim=combined_handcrafted.shape[1],
            embedding_dim=combined_embeddings.shape[1] if combined_embeddings is not None else 0
        )
        
        print(f"   Final combined features: {final_features.shape}")
        print(f"   Total feature names: {len(self.feature_names)}")
        
        # Save features if requested
        if save_path:
            feature_data = {
                'features': final_features,
                'labels': combined_labels,
                'image_names': all_image_names,
                'feature_names': self.feature_names,
                'deep_features': combined_deep_features,
                'handcrafted_features': combined_handcrafted,
                'embeddings': combined_embeddings
            }
            joblib.dump(feature_data, save_path)
            print(f"Features saved to {save_path}")
        
        return {
            'features': final_features,
            'labels': combined_labels,
            'image_names': all_image_names,
            'feature_names': self.feature_names,
            'components': {
                'deep': combined_deep_features,
                'handcrafted': combined_handcrafted,
                'embeddings': combined_embeddings
            }
        }
    
    def _create_feature_names(self, deep_dim, handcrafted_dim, embedding_dim):
        """Create interpretable feature names"""
        feature_names = []
        
        # Deep feature names
        feature_names.extend([f'mobilenetv3_{self.model_size}_feature_{i}' for i in range(deep_dim)])
        
        # Handcrafted feature names (organized by region)
        regions = ['cell', 'nucleus', 'cytoplasm']
        for region in regions:
            # Shape features (12 per region)
            shape_features = ['convexity', 'compactness', 'elongation', 'eccentricity', 
                            'roundness', 'solidity', 'area', 'perimeter',
                            'aspect_ratio', 'extent', 'equivalent_diameter', 'orientation']
            feature_names.extend([f'{region}_{feat}' for feat in shape_features])
            
            # Color features (15 per region: 5 features × 3 channels)
            color_channels = ['red', 'green', 'blue']
            color_stats = ['mean', 'variance', 'skewness', 'kurtosis', 'entropy']
            for channel in color_channels:
                for stat in color_stats:
                    feature_names.append(f'{region}_{channel}_{stat}')
            
            # Texture features (20 per region: 5 features × 4 angles)
            texture_features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            angles = ['0deg', '45deg', '90deg', '135deg']
            for feature in texture_features:
                for angle in angles:
                    feature_names.append(f'{region}_{feature}_{angle}')
            
            # Fractal feature (1 per region)
            feature_names.append(f'{region}_fractal_dimension')
        
        # Embedding feature names
        if embedding_dim > 0:
            feature_names.extend([f'embedding_feature_{i}' for i in range(embedding_dim)])
        
        self.feature_names = feature_names

class ExtraTreesClassifierWrapper:
    """
    Optimized Extra Trees classifier for bone marrow cell classification
    """
    def __init__(self, num_classes=21, class_weights=None):
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Optimized parameters for bone marrow classification
        self.params = {
            'n_estimators': 300,
            'criterion': 'gini',
            'max_depth': 25,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': False,  # Extra Trees uses all samples
            'oob_score': False,  # Not applicable when bootstrap=False
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1,
            'warm_start': False,
            'class_weight': 'balanced' if class_weights is None else class_weights,
            'max_samples': None  # Use all samples
        }
        
        print(f"Extra Trees Classifier initialized for {num_classes} classes")
        print("Extra Trees features: Extremely randomized trees with random thresholds")
        
    def train(self, X_train, X_test, y_train, y_test, feature_names=None, 
              use_early_stopping=False, cv_folds=5):
        """Train Extra Trees with cross-validation"""
        print("Training Extra Trees Classifier...")
        
        # Feature scaling (optional for tree-based methods, but can help)
        print("Scaling features...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Cross-validation for robust evaluation
        if cv_folds > 1:
            print(f"Performing {cv_folds}-fold cross-validation...")
            cv_scores = []
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
                print(f"   Fold {fold+1}/{cv_folds}")
                
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Train fold model
                fold_model = ExtraTreesClassifier(**self.params)
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Validate
                val_pred = fold_model.predict(X_fold_val)
                fold_score = accuracy_score(y_fold_val, val_pred)
                cv_scores.append(fold_score)
                
                print(f"     Trees trained: {fold_model.n_estimators}")
                
            print(f"CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train final model
        print("Training final model on full training set...")
        
        # Initialize Extra Trees model
        self.model = ExtraTreesClassifier(**self.params)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Print training information
        print(f"Extra Trees trained with {self.model.n_estimators} extremely randomized trees")
        print(f"Max features per split: {self.model.max_features}")
        print(f"Random threshold selection: Enabled (Extra Trees feature)")
        
        # Final evaluation
        test_pred = self.model.predict(X_test_scaled)
        test_pred_proba = self.model.predict_proba(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'cv_scores': cv_scores if cv_folds > 1 else None,
            'test_accuracy': test_accuracy,
            'y_test': y_test,
            'y_pred': test_pred,
            'y_pred_proba': test_pred_proba
        }
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities
    
    def get_feature_importance(self, importance_type='feature_importances_'):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        if importance_type == 'feature_importances_':
            importance = self.model.feature_importances_
        else:
            # For permutation importance (computationally expensive)
            importance = permutation_importance(self.model, X_test, y_test).importances_mean
            
        return importance
    
    def analyze_tree_diversity(self):
        """Analyze diversity of trees in Extra Trees ensemble"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        tree_info = {
            'n_estimators': self.model.n_estimators,
            'tree_depths': [],
            'tree_leaves': [],
            'feature_usage': defaultdict(int)
        }
        
        # Analyze individual trees
        for i, tree in enumerate(self.model.estimators_):
            depth = tree.tree_.max_depth
            n_leaves = tree.tree_.n_leaves
            
            tree_info['tree_depths'].append(depth)
            tree_info['tree_leaves'].append(n_leaves)
            
            # Count feature usage
            features_used = tree.tree_.feature
            unique_features = np.unique(features_used[features_used >= 0])
            for feat_idx in unique_features:
                tree_info['feature_usage'][feat_idx] += 1
        
        # Calculate diversity statistics
        tree_info['avg_depth'] = np.mean(tree_info['tree_depths'])
        tree_info['std_depth'] = np.std(tree_info['tree_depths'])
        tree_info['avg_leaves'] = np.mean(tree_info['tree_leaves'])
        tree_info['std_leaves'] = np.std(tree_info['tree_leaves'])
        
        return tree_info

class XAIAnalyzer:
    """
    Comprehensive XAI analysis for the hybrid model
    """
    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        self.lime_explainer = None
        
        print("XAI Analyzer initialized for Extra Trees")
        
    def setup_shap_explainer(self, X_background, max_background_samples=100):
        """Setup SHAP explainer with background data"""
        print("Setting up SHAP explainer for Extra Trees...")
        
        # Limit background samples for efficiency
        if len(X_background) > max_background_samples:
            background_idx = np.random.choice(len(X_background), max_background_samples, replace=False)
            X_bg = X_background[background_idx]
        else:
            X_bg = X_background
        
        # Scale background data
        X_bg_scaled = self.model.scaler.transform(X_bg)
        
        # Create TreeExplainer for Extra Trees
        self.shap_explainer = shap.TreeExplainer(self.model.model)
        
        print("SHAP explainer ready")
        
    def setup_lime_explainer(self, X_training):
        """Setup LIME explainer"""
        print("Setting up LIME explainer...")
        
        X_training_scaled = self.model.scaler.transform(X_training)
        
        self.lime_explainer = LimeTabularExplainer(
            X_training_scaled,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=True
        )
        
        print("LIME explainer ready")
        
    def analyze_feature_importance(self, save_path=None, top_k=30):
        """Analyze and visualize feature importance"""
        print("Analyzing feature importance...")
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Create DataFrame
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = feature_imp_df.head(top_k)
        
        if SNS_AVAILABLE:
            sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        else:
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
        
        plt.title(f'Top {top_k} Most Important Features (Extra Trees)', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Importance (Gini)', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_imp_df
    
    def analyze_region_importance(self, feature_imp_df, save_path=None):
        """Analyze importance by cell regions"""
        print("Analyzing region-wise importance...")
        
        # Group by feature types
        importance_by_type = {
            'mobilenetv3': 0,
            'cell': 0,
            'nucleus': 0,
            'cytoplasm': 0,
            'embedding': 0
        }
        
        for _, row in feature_imp_df.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            
            if feature_name.startswith('mobilenetv3_'):
                importance_by_type['mobilenetv3'] += importance
            elif feature_name.startswith('cell_'):
                importance_by_type['cell'] += importance
            elif feature_name.startswith('nucleus_'):
                importance_by_type['nucleus'] += importance
            elif feature_name.startswith('cytoplasm_'):
                importance_by_type['cytoplasm'] += importance
            elif feature_name.startswith('embedding_'):
                importance_by_type['embedding'] += importance
        
        # Plot region importance
        plt.figure(figsize=(10, 6))
        types = list(importance_by_type.keys())
        importances = list(importance_by_type.values())
        
        colors = ['gold', 'lightcoral', 'skyblue', 'lightgreen', 'plum']
        bars = plt.bar(types, importances, color=colors, alpha=0.7)
        
        plt.title('Feature Importance by Type/Region', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Type/Region', fontsize=12)
        plt.ylabel('Total Importance', fontsize=12)
        
        # Add value labels
        for bar, imp in zip(bars, importances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importances)*0.01, 
                    f'{imp:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_by_type
    
    def generate_shap_explanations(self, X_samples, sample_indices=None, max_samples=50):
        """Generate SHAP explanations for samples"""
        if self.shap_explainer is None:
            print("ERROR: SHAP explainer not setup. Call setup_shap_explainer() first.")
            return None
        
        print("Generating SHAP explanations...")
        
        # Limit samples for performance
        if len(X_samples) > max_samples:
            if sample_indices is not None:
                selected_indices = sample_indices[:max_samples]
                X_explain = X_samples[selected_indices]
            else:
                X_explain = X_samples[:max_samples]
        else:
            X_explain = X_samples
        
        # Scale samples
        X_explain_scaled = self.model.scaler.transform(X_explain)
        
        # Generate SHAP values
        shap_values = self.shap_explainer.shap_values(X_explain_scaled)
        
        return shap_values, X_explain_scaled
    
    def plot_shap_summary(self, shap_values, X_samples, save_path=None, max_display=20):
        """Plot SHAP summary for all classes"""
        print("Creating SHAP summary plots...")
        
        if isinstance(shap_values, list):
            # Multi-class case
            for i, class_name in enumerate(self.class_names):
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values[i],
                    X_samples,
                    feature_names=self.feature_names,
                    show=False,
                    max_display=max_display
                )
                plt.title(f'SHAP Summary - {class_name}', fontsize=14, fontweight='bold')
                
                if save_path:
                    class_save_path = save_path.replace('.png', f'_class_{class_name}.png')
                    plt.savefig(class_save_path, dpi=300, bbox_inches='tight')
                plt.show()
        else:
            # Binary case
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_samples, feature_names=self.feature_names, 
                            show=False, max_display=max_display)
            plt.title('SHAP Summary', fontsize=14, fontweight='bold')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def explain_single_prediction(self, X_sample, sample_idx=0, top_k=15):
        """Provide detailed explanation for a single prediction"""
        if self.shap_explainer is None:
            print("ERROR: SHAP explainer not setup.")
            return None
        
        print(f"Explaining prediction for sample {sample_idx}")
        
        # Get prediction
        predictions, prediction_proba = self.model.predict(X_sample.reshape(1, -1))
        predicted_class = predictions[0]
        confidence = np.max(prediction_proba[0])
        
        # Get SHAP values
        X_scaled = self.model.scaler.transform(X_sample.reshape(1, -1))
        shap_values = self.shap_explainer.shap_values(X_scaled)
        
        if isinstance(shap_values, list):
            sample_shap = shap_values[predicted_class][0]
        else:
            sample_shap = shap_values[0]
        
        # Get top contributing features
        feature_contributions = list(zip(self.feature_names, sample_shap, X_sample))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"Predicted Class: {self.class_names[predicted_class]} (confidence: {confidence:.3f})")
        print(f"Top {top_k} Contributing Features:")
        print("-" * 80)
        
        for i, (feature, shap_val, feature_val) in enumerate(feature_contributions[:top_k]):
            direction = "Increases" if shap_val > 0 else "Decreases"
            print(f"{i+1:2d}. {feature:35s} | SHAP: {shap_val:8.4f} | {direction} | Value: {feature_val:.4f}")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_features': feature_contributions[:top_k]
        }
    
    def analyze_tree_diversity(self, save_path=None):
        """Analyze Extra Trees ensemble diversity"""
        if self.model.model is None:
            print("ERROR: Model not trained.")
            return None
        
        print("Analyzing Extra Trees diversity...")
        
        # Get diversity analysis
        diversity_info = self.model.analyze_tree_diversity()
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Tree depth distribution
        ax1.hist(diversity_info['tree_depths'], bins=20, alpha=0.7, color='skyblue')
        ax1.set_title('Distribution of Tree Depths')
        ax1.set_xlabel('Max Depth')
        ax1.set_ylabel('Number of Trees')
        ax1.axvline(diversity_info['avg_depth'], color='red', linestyle='--', 
                   label=f'Mean: {diversity_info["avg_depth"]:.1f}')
        ax1.legend()
        
        # Tree leaves distribution
        ax2.hist(diversity_info['tree_leaves'], bins=20, alpha=0.7, color='lightgreen')
        ax2.set_title('Distribution of Number of Leaves')
        ax2.set_xlabel('Number of Leaves')
        ax2.set_ylabel('Number of Trees')
        ax2.axvline(diversity_info['avg_leaves'], color='red', linestyle='--',
                   label=f'Mean: {diversity_info["avg_leaves"]:.1f}')
        ax2.legend()
        
        # Feature usage frequency (top 20 features)
        feature_usage = dict(sorted(diversity_info['feature_usage'].items(), 
                                  key=lambda x: x[1], reverse=True)[:20])
        features_used = [self.feature_names[idx] if idx < len(self.feature_names) else f'Feature_{idx}' 
                        for idx in feature_usage.keys()]
        usage_counts = list(feature_usage.values())
        
        ax3.barh(range(len(features_used)), usage_counts, alpha=0.7, color='coral')
        ax3.set_yticks(range(len(features_used)))
        ax3.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in features_used], fontsize=8)
        ax3.set_title('Top 20 Feature Usage Frequency')
        ax3.set_xlabel('Number of Trees Using Feature')
        
        # Summary statistics
        ax4.text(0.1, 0.8, f"Extra Trees Ensemble Analysis", fontsize=14, weight='bold')
        ax4.text(0.1, 0.7, f"Number of Trees: {diversity_info['n_estimators']}", fontsize=12)
        ax4.text(0.1, 0.6, f"Avg Tree Depth: {diversity_info['avg_depth']:.1f} ± {diversity_info['std_depth']:.1f}", fontsize=12)
        ax4.text(0.1, 0.5, f"Avg Leaves: {diversity_info['avg_leaves']:.1f} ± {diversity_info['std_leaves']:.1f}", fontsize=12)
        ax4.text(0.1, 0.4, f"Features Used: {len(diversity_info['feature_usage'])}", fontsize=12)
        ax4.text(0.1, 0.3, f"Random Thresholds: Enabled", fontsize=12)
        ax4.text(0.1, 0.2, f"Bootstrap Sampling: Disabled", fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return diversity_info

def evaluate_model_comprehensive(y_true, y_pred, y_pred_proba, class_names, save_dir=None):
    """Comprehensive model evaluation with multiple metrics"""
    print("Comprehensive model evaluation...")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Per-class metrics
    classification_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Multi-class AUC (if probability predictions available)
    try:
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc_score = None
    
    print(f"Evaluation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision (weighted): {precision:.4f}")
    print(f"   Recall (weighted): {recall:.4f}")
    print(f"   F1-Score (weighted): {f1:.4f}")
    print(f"   F1-Score (macro): {f1_macro:.4f}")
    if auc_score:
        print(f"   AUC (weighted): {auc_score:.4f}")
    
    # Plot confusion matrix
    if save_dir:
        plt.figure(figsize=(12, 10))
        if SNS_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center')
            plt.xticks(range(len(class_names)), class_names, rotation=45)
            plt.yticks(range(len(class_names)), class_names)
        
        plt.title('Confusion Matrix - MobileNetV3 + Extra Trees', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_weighted': f1,
        'f1_macro': f1_macro,
        'auc_weighted': auc_score,
        'classification_report': classification_rep,
        'confusion_matrix': cm
    }

def create_data_loaders(csv_file, image_dir, embeddings_dir, batch_size=32, test_size=0.2, mode='hybrid'):
    """Create optimized data loaders for the hybrid model"""
    print("Creating optimized data loaders...")
    
    # Load and split data
    full_df = pd.read_csv(csv_file)
    print(f"Dataset: {len(full_df)} samples, {full_df['label'].nunique()} classes")
    
    # Stratified split
    labels_for_split = LabelEncoder().fit_transform(full_df['label'])
    train_idx, test_idx = train_test_split(
        range(len(full_df)), test_size=test_size, 
        stratify=labels_for_split, random_state=42
    )
    
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)
    
    # Save splits
    train_df.to_csv('train_split.csv', index=False)
    test_df.to_csv('test_split.csv', index=False)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Transforms for images (224x224 for MobileNetV3)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BoneMarrowDataset('train_split.csv', image_dir, embeddings_dir, 
                                    train_transform, mode)
    test_dataset = BoneMarrowDataset('test_split.csv', image_dir, embeddings_dir, 
                                   test_transform, mode)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=torch.cuda.is_available())
    
    return train_loader, test_loader, train_dataset.label_encoder

def main():
    """Main training and evaluation pipeline"""
    parser = argparse.ArgumentParser(description='MobileNetV3 + Extra Trees Hybrid Classification')
    parser.add_argument('--csv_file', required=True, help='Features CSV file')
    parser.add_argument('--image_dir', required=True, help='Images directory')
    parser.add_argument('--embeddings_dir', required=True, help='Embeddings directory')
    parser.add_argument('--save_dir', default='mobilenetv3_extratrees_results', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--deep_feature_dim', type=int, default=512, help='Deep feature dimension')
    parser.add_argument('--mode', default='hybrid', choices=['hybrid', 'features_only', 'images_only'],
                       help='Training mode')
    parser.add_argument('--cv_folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--n_estimators', type=int, default=300, help='Number of trees in Extra Trees')
    parser.add_argument('--mobilenet_size', default='large', choices=['large', 'small'],
                       help='MobileNetV3 model size')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    start_time = time.time()
    
    # Create data loaders
    train_loader, test_loader, label_encoder = create_data_loaders(
        args.csv_file, args.image_dir, args.embeddings_dir, 
        batch_size=args.batch_size, mode=args.mode
    )
    
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_
    print(f"Classes ({num_classes}): {list(class_names)}")
    
    # Initialize hybrid feature extractor
    feature_extractor = HybridFeatureExtractor(
        deep_feature_dim=args.deep_feature_dim,
        device=device,
        model_size=args.mobilenet_size
    )
    
    # Extract features
    print("Phase 1: Feature Extraction")
    train_features = feature_extractor.extract_deep_features(
        train_loader, 
        save_path=os.path.join(args.save_dir, 'train_features.pkl')
    )
    
    test_features = feature_extractor.extract_deep_features(
        test_loader,
        save_path=os.path.join(args.save_dir, 'test_features.pkl')
    )
    
    # Calculate class weights for imbalanced data
    class_weights_dict = compute_class_weight('balanced', 
                                            classes=np.unique(train_features['labels']), 
                                            y=train_features['labels'])
    class_weights = dict(zip(np.unique(train_features['labels']), class_weights_dict))
    
    # Initialize and train Extra Trees
    print("Phase 2: Extra Trees Training")
    et_classifier = ExtraTreesClassifierWrapper(num_classes=num_classes, class_weights=class_weights)
    
    # Update number of estimators if specified
    et_classifier.params['n_estimators'] = args.n_estimators
    
    training_results = et_classifier.train(
        X_train=train_features['features'],
        X_test=test_features['features'],
        y_train=train_features['labels'],
        y_test=test_features['labels'],
        feature_names=train_features['feature_names'],
        cv_folds=args.cv_folds
    )
    
    # Comprehensive evaluation
    print("Phase 3: Comprehensive Evaluation")
    eval_results = evaluate_model_comprehensive(
        y_true=training_results['y_test'],
        y_pred=training_results['y_pred'],
        y_pred_proba=training_results['y_pred_proba'],
        class_names=class_names,
        save_dir=args.save_dir
    )
    
    # XAI Analysis
    print("Phase 4: XAI Analysis")
    xai_analyzer = XAIAnalyzer(et_classifier, train_features['feature_names'], class_names)
    
    # Feature importance analysis
    feature_imp_df = xai_analyzer.analyze_feature_importance(
        save_path=os.path.join(args.save_dir, 'feature_importance.png'),
        top_k=30
    )
    
    # Region importance analysis
    region_importance = xai_analyzer.analyze_region_importance(
        feature_imp_df,
        save_path=os.path.join(args.save_dir, 'region_importance.png')
    )
    
    # Extra Trees specific analysis
    diversity_analysis = xai_analyzer.analyze_tree_diversity(
        save_path=os.path.join(args.save_dir, 'tree_diversity.png')
    )
    
    # SHAP analysis
    xai_analyzer.setup_shap_explainer(train_features['features'])
    shap_values, X_explain = xai_analyzer.generate_shap_explanations(
        test_features['features'], max_samples=50
    )
    
    if shap_values is not None:
        xai_analyzer.plot_shap_summary(
            shap_values, X_explain,
            save_path=os.path.join(args.save_dir, 'shap_summary.png')
        )
    
    # Example single prediction explanation
    if len(test_features['features']) > 0:
        single_explanation = xai_analyzer.explain_single_prediction(
            test_features['features'][0], sample_idx=0
        )
    
    # Save models and results
    print("Saving models and results...")
    
    # Save Extra Trees model
    joblib.dump(et_classifier.model, os.path.join(args.save_dir, 'extra_trees_model.pkl'))
    joblib.dump(et_classifier.scaler, os.path.join(args.save_dir, 'feature_scaler.pkl'))
    
    # Save MobileNetV3 feature extractor
    torch.save(feature_extractor.mobilenetv3.state_dict(), 
               os.path.join(args.save_dir, f'mobilenetv3_{args.mobilenet_size}_feature_extractor.pth'))
    
    # Save other components
    joblib.dump(label_encoder, os.path.join(args.save_dir, 'label_encoder.pkl'))
    joblib.dump(train_features['feature_names'], os.path.join(args.save_dir, 'feature_names.pkl'))
    
    # Save comprehensive results
    final_results = {
        'training_args': vars(args),
        'training_results': {
            'cv_scores': training_results['cv_scores'],
            'test_accuracy': training_results['test_accuracy']
        },
        'evaluation_results': eval_results,
        'feature_importance': feature_imp_df.to_dict('records'),
        'region_importance': region_importance,
        'diversity_analysis': diversity_analysis,
        'model_info': {
            'num_classes': num_classes,
            'class_names': list(class_names),
            'total_features': len(train_features['feature_names']),
            'deep_features': args.deep_feature_dim,
            'handcrafted_features': 144,
            'embedding_features': 432 if args.mode == 'hybrid' else 0,
            'n_estimators': args.n_estimators,
            'mobilenet_size': args.mobilenet_size
        }
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    final_results = convert_numpy_types(final_results)
    
    with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Training summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Final Results:")
    print(f"   Accuracy: {eval_results['accuracy']:.4f}")
    print(f"   F1-Score (weighted): {eval_results['f1_weighted']:.4f}")
    print(f"   F1-Score (macro): {eval_results['f1_macro']:.4f}")
    if eval_results['auc_weighted']:
        print(f"   AUC (weighted): {eval_results['auc_weighted']:.4f}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {args.save_dir}")
    print(f"Model features:")
    print(f"   MobileNetV3-{args.mobilenet_size}: Efficient mobile-optimized CNN")
    print(f"   Extra Trees: {args.n_estimators} extremely randomized trees")
    print(f"   Deep features: {args.deep_feature_dim}")
    print(f"   Total features: {len(train_features['feature_names'])}")
    print(f"   Classes: {num_classes}")
    
    # Feature breakdown
    print(f"Feature contribution (by importance):")
    for feature_type, importance in region_importance.items():
        percentage = (importance / sum(region_importance.values())) * 100
        print(f"   {feature_type}: {percentage:.1f}%")
    
    # Efficiency metrics
    print(f"Efficiency metrics:")
    print(f"   Avg tree depth: {diversity_analysis['avg_depth']:.1f}")
    print(f"   Avg tree leaves: {diversity_analysis['avg_leaves']:.1f}")
    print(f"   Model size optimized for: Mobile/Edge deployment")

if __name__ == "__main__":
    main()