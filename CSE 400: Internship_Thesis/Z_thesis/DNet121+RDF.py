#!/usr/bin/env python3
"""
Complete DenseNet121 + Random Forest Hybrid Bone Marrow Cell Classification Model
Advanced hybrid approach combining deep learning feature extraction with ensemble learning:
- DenseNet121 for deep feature extraction from images with dense connections
- Random Forest for classification on combined features (deep + handcrafted + embeddings)
- Comprehensive XAI with SHAP and feature importance analysis
- Optimized for large-scale bone marrow datasets (150k+ images)
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

# Random Forest and ML imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (classification_report, f1_score, accuracy_score, 
                           confusion_matrix, precision_recall_fscore_support,
                           roc_auc_score, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight

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

print("Complete DenseNet121 + Random Forest Hybrid Model")
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
                    # Resize to 224x224 for DenseNet121
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

class DenseNet121FeatureExtractor(nn.Module):
    """
    DenseNet121-based feature extractor optimized for bone marrow images
    """
    def __init__(self, feature_dim=512, pretrained=True):
        super(DenseNet121FeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        
        print(f"Building DenseNet121 Feature Extractor")
        
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        backbone_features = self.backbone.classifier.in_features  # 1024 for DenseNet121
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add custom feature extraction head with dense-style connections
        self.feature_head = nn.Sequential(
            # First dense block
            nn.Linear(backbone_features, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # Second dense block
            nn.Linear(768, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Final feature layer
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Additional dense connections (inspired by DenseNet philosophy)
        self.dense_connection = nn.Sequential(
            nn.Linear(backbone_features + feature_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fine-tuning strategy for DenseNet
        self._setup_fine_tuning()
        
        print(f"DenseNet121 Feature Extractor built: {backbone_features} -> {feature_dim}")
        
    def _setup_fine_tuning(self):
        """Setup fine-tuning strategy for DenseNet121"""
        # Freeze early dense blocks, fine-tune later ones
        for name, param in self.backbone.named_parameters():
            if 'denseblock1' in name or 'denseblock2' in name:
                param.requires_grad = False
            elif 'denseblock3' in name or 'denseblock4' in name:
                param.requires_grad = True
            elif 'transition' in name:
                param.requires_grad = True
    
    def forward(self, x):
        """Forward pass through DenseNet121 with dense connections"""
        # Extract features using DenseNet121 backbone
        backbone_features = self.backbone(x)
        
        # Ensure features are flattened
        if backbone_features.dim() > 2:
            backbone_features = torch.flatten(backbone_features, 1)
        
        # Pass through feature head
        head_features = self.feature_head(backbone_features)
        
        # Apply dense connection (concatenate and transform)
        combined = torch.cat([backbone_features, head_features], dim=1)
        dense_features = self.dense_connection(combined)
        
        return dense_features

class HybridFeatureExtractor:
    """
    Complete hybrid feature extraction system
    Combines DenseNet121 deep features with handcrafted features and embeddings
    """
    def __init__(self, deep_feature_dim=512, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.deep_feature_dim = deep_feature_dim
        
        # Initialize DenseNet121
        self.densenet121 = DenseNet121FeatureExtractor(
            feature_dim=deep_feature_dim,
            pretrained=True
        ).to(self.device)
        
        # Feature names for interpretability
        self.feature_names = None
        
        print(f"Hybrid Feature Extractor initialized on {self.device}")
        
    def extract_deep_features(self, dataloader, save_path=None):
        """Extract deep features from images using DenseNet121"""
        print("Extracting deep features using DenseNet121...")
        
        self.densenet121.eval()
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
                    deep_features = self.densenet121(images)
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
        feature_names.extend([f'densenet121_feature_{i}' for i in range(deep_dim)])
        
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

class RandomForestClassifierWrapper:
    """
    Optimized Random Forest classifier for bone marrow cell classification
    """
    def __init__(self, num_classes=21, class_weights=None):
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Optimized parameters for bone marrow classification
        self.params = {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1,
            'warm_start': False,
            'class_weight': 'balanced' if class_weights is None else class_weights
        }
        
        print(f"Random Forest Classifier initialized for {num_classes} classes")
        
    def train(self, X_train, X_test, y_train, y_test, feature_names=None, 
              use_early_stopping=False, cv_folds=5):
        """Train Random Forest with cross-validation"""
        print("Training Random Forest Classifier...")
        
        # Feature scaling (optional for Random Forest, but can help)
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
                fold_model = RandomForestClassifier(**self.params)
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Validate
                val_pred = fold_model.predict(X_fold_val)
                fold_score = accuracy_score(y_fold_val, val_pred)
                cv_scores.append(fold_score)
                
                # Print OOB score if available
                if hasattr(fold_model, 'oob_score_'):
                    print(f"     OOB Score: {fold_model.oob_score_:.4f}")
                
            print(f"CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train final model
        print("Training final model on full training set...")
        
        # Initialize Random Forest model
        self.model = RandomForestClassifier(**self.params)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Print training information
        print(f"Random Forest trained with {self.model.n_estimators} trees")
        if hasattr(self.model, 'oob_score_'):
            print(f"Out-of-bag score: {self.model.oob_score_:.4f}")
        
        # Final evaluation
        test_pred = self.model.predict(X_test_scaled)
        test_pred_proba = self.model.predict_proba(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'cv_scores': cv_scores if cv_folds > 1 else None,
            'test_accuracy': test_accuracy,
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None,
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
            from sklearn.inspection import permutation_importance
            importance = permutation_importance(self.model, X_test, y_test).importances_mean
            
        return importance

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
        
        print("XAI Analyzer initialized")
        
    def setup_shap_explainer(self, X_background, max_background_samples=100):
        """Setup SHAP explainer with background data"""
        print("Setting up SHAP explainer...")
        
        # Limit background samples for efficiency
        if len(X_background) > max_background_samples:
            background_idx = np.random.choice(len(X_background), max_background_samples, replace=False)
            X_bg = X_background[background_idx]
        else:
            X_bg = X_background
        
        # Scale background data
        X_bg_scaled = self.model.scaler.transform(X_bg)
        
        # Create TreeExplainer for Random Forest
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
        
        plt.title(f'Top {top_k} Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
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
            'densenet121': 0,
            'cell': 0,
            'nucleus': 0,
            'cytoplasm': 0,
            'embedding': 0
        }
        
        for _, row in feature_imp_df.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            
            if feature_name.startswith('densenet121_'):
                importance_by_type['densenet121'] += importance
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
    
    def analyze_tree_feature_usage(self, tree_idx=0):
        """Analyze feature usage in individual trees (Random Forest specific)"""
        if self.model.model is None:
            print("ERROR: Model not trained.")
            return None
        
        print(f"Analyzing tree {tree_idx} feature usage...")
        
        # Get individual tree
        tree = self.model.model.estimators_[tree_idx]
        
        # Get features used in this tree
        features_used = tree.tree_.feature
        unique_features = np.unique(features_used[features_used >= 0])  # Remove -2 (leaf nodes)
        
        feature_usage = {}
        for feat_idx in unique_features:
            if feat_idx < len(self.feature_names):
                feature_usage[self.feature_names[feat_idx]] = np.sum(features_used == feat_idx)
        
        # Sort by usage frequency
        sorted_usage = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Top features used in tree {tree_idx}:")
        for i, (feature, count) in enumerate(sorted_usage[:10]):
            print(f"{i+1:2d}. {feature:35s} | Used {count} times")
        
        return sorted_usage

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
        
        plt.title('Confusion Matrix - DenseNet121 + Random Forest', fontsize=14, fontweight='bold')
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
    
    # Transforms for images (224x224 for DenseNet121)
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
    parser = argparse.ArgumentParser(description='DenseNet121 + Random Forest Hybrid Classification')
    parser.add_argument('--csv_file', required=True, help='Features CSV file')
    parser.add_argument('--image_dir', required=True, help='Images directory')
    parser.add_argument('--embeddings_dir', required=True, help='Embeddings directory')
    parser.add_argument('--save_dir', default='densenet121_randomforest_results', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--deep_feature_dim', type=int, default=512, help='Deep feature dimension')
    parser.add_argument('--mode', default='hybrid', choices=['hybrid', 'features_only', 'images_only'],
                       help='Training mode')
    parser.add_argument('--cv_folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--n_estimators', type=int, default=500, help='Number of trees in Random Forest')
    
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
        device=device
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
    
    # Initialize and train Random Forest
    print("Phase 2: Random Forest Training")
    rf_classifier = RandomForestClassifierWrapper(num_classes=num_classes, class_weights=class_weights)
    
    # Update number of estimators if specified
    rf_classifier.params['n_estimators'] = args.n_estimators
    
    training_results = rf_classifier.train(
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
    xai_analyzer = XAIAnalyzer(rf_classifier, train_features['feature_names'], class_names)
    
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
    
    # Random Forest specific analysis
    tree_analysis = xai_analyzer.analyze_tree_feature_usage(tree_idx=0)
    
    # Save models and results
    print("Saving models and results...")
    
    # Save Random Forest model
    joblib.dump(rf_classifier.model, os.path.join(args.save_dir, 'random_forest_model.pkl'))
    joblib.dump(rf_classifier.scaler, os.path.join(args.save_dir, 'feature_scaler.pkl'))
    
    # Save DenseNet121 feature extractor
    torch.save(feature_extractor.densenet121.state_dict(), 
               os.path.join(args.save_dir, 'densenet121_feature_extractor.pth'))
    
    # Save other components
    joblib.dump(label_encoder, os.path.join(args.save_dir, 'label_encoder.pkl'))
    joblib.dump(train_features['feature_names'], os.path.join(args.save_dir, 'feature_names.pkl'))
    
    # Save comprehensive results
    final_results = {
        'training_args': vars(args),
        'training_results': {
            'cv_scores': training_results['cv_scores'],
            'test_accuracy': training_results['test_accuracy'],
            'oob_score': training_results['oob_score']
        },
        'evaluation_results': eval_results,
        'feature_importance': feature_imp_df.to_dict('records'),
        'region_importance': region_importance,
        'model_info': {
            'num_classes': num_classes,
            'class_names': list(class_names),
            'total_features': len(train_features['feature_names']),
            'deep_features': args.deep_feature_dim,
            'handcrafted_features': 144,
            'embedding_features': 432 if args.mode == 'hybrid' else 0,
            'n_estimators': args.n_estimators
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
    if training_results['oob_score']:
        print(f"   OOB Score: {training_results['oob_score']:.4f}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {args.save_dir}")
    print(f"Model features:")
    print(f"   DenseNet121: Dense connections with feature reuse")
    print(f"   Random Forest: {args.n_estimators} trees with bootstrap aggregating")
    print(f"   Deep features: {args.deep_feature_dim}")
    print(f"   Total features: {len(train_features['feature_names'])}")
    print(f"   Classes: {num_classes}")
    
    # Feature breakdown
    print(f"Feature contribution (by importance):")
    for feature_type, importance in region_importance.items():
        percentage = (importance / sum(region_importance.values())) * 100
        print(f"   {feature_type}: {percentage:.1f}%")

if __name__ == "__main__":
    main()