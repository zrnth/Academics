#!/usr/bin/env python3
"""
Hybrid Xception + XGBoost Bone Marrow Cell Classification Model
Combines:
- Enhanced Xception for region-attention embeddings
- XGBoost for traditional feature classification
- Cross-modal fusion and uncertainty estimation
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import math
import pickle
from typing import Dict, List, Tuple, Optional

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available - training will continue without logging")

# Torchvision for image processing
from torchvision import transforms

# XGBoost imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available! Please install: pip install xgboost")
    exit(1)

# Scientific computing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Visualization
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("⚠️  Seaborn not available - using basic matplotlib plots")

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Hybrid Xception + XGBoost Bone Marrow Classification Model")
print(f"PyTorch Version: {torch.__version__}")
print(f"XGBoost Version: {xgb.__version__}")
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
    Dataset for hybrid Xception + XGBoost model
    Handles both region-attention embeddings and traditional features
    """
    def __init__(self, csv_file, image_dir, embeddings_dir, transform=None, mode='both'):
        print(f"📊 Loading dataset from {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.transform = transform
        self.mode = mode
        
        # Extract feature columns (exclude metadata)
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['image_name', 'label']]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"✅ Dataset loaded: {len(self.df)} samples, {self.num_classes} classes")
        print(f"📋 Classes: {list(self.label_encoder.classes_)}")
        print(f"🔧 Mode: {mode}")
        print(f"🔢 Feature columns: {len(self.feature_columns)}")
        
        # Validate feature structure
        self._validate_feature_structure()
        
        # Validate data availability
        self._validate_data()
        
    def _validate_feature_structure(self):
        """Validate that features match the expected CSV structure"""
        expected_regions = ['cell', 'nucleus', 'cytoplasm']
        expected_features_per_region = 48
        
        # Count features per region
        region_counts = {}
        for region in expected_regions:
            region_features = [col for col in self.feature_columns if col.startswith(f'{region}_')]
            region_counts[region] = len(region_features)
        
        print(f"🔍 Feature validation:")
        all_good = True
        for region, count in region_counts.items():
            status = "✅" if count == expected_features_per_region else "⚠️"
            print(f"   {status} {region}: {count} features (expected {expected_features_per_region})")
            if count != expected_features_per_region:
                all_good = False
        
        total_features = sum(region_counts.values())
        status = "✅" if total_features == 144 else "⚠️"
        print(f"   {status} Total: {total_features} features (expected 144)")
        
        if all_good:
            print("🎯 Perfect! Your CSV structure matches the paper exactly.")
        else:
            print("⚠️  Warning: Feature structure doesn't match expected pattern.")
            print("   This might still work, but performance could be affected.")
        
    def _validate_data(self):
        """Validate that required files exist"""
        missing_embeddings = 0
        
        sample_size = min(100, len(self.df))  # Check first 100 samples
        for i in range(sample_size):
            row = self.df.iloc[i]
            image_name = row['image_name']
            
            # Check embeddings
            if self.mode in ['both', 'embedding_only']:
                embedding_path = self.embeddings_dir / f"{image_name}_embedding_3ch.npy"
                if not embedding_path.exists():
                    missing_embeddings += 1
            
        if missing_embeddings > 0:
            print(f"⚠️  Warning: {missing_embeddings}/{sample_size} sample embeddings not found")
            
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
        
        # Load embedding if needed
        if self.mode in ['both', 'embedding_only']:
            embedding = self._load_embedding(image_name, row)
            sample['embedding'] = embedding
            
        # Load features for XGBoost
        if self.mode in ['both', 'features_only']:
            features = row[self.feature_columns].values.astype(np.float32)
            sample['features'] = torch.tensor(features, dtype=torch.float32)
            
        return sample
    
    def _load_embedding(self, image_name, row):
        """Load region-attention embedding with proper error handling"""
        embedding_path = self.embeddings_dir / f"{image_name}_embedding_3ch.npy"
        
        if embedding_path.exists():
            try:
                embedding = np.load(embedding_path).astype(np.float32)
                # Ensure correct shape [12, 12, 3]
                if embedding.shape != (12, 12, 3):
                    print(f"Warning: Embedding {image_name} has shape {embedding.shape}, expected (12, 12, 3)")
                    if embedding.size == 432:  # 12*12*3 = 432
                        embedding = embedding.reshape(12, 12, 3)
                    else:
                        embedding = self._create_embedding_from_features(row)
            except Exception as e:
                print(f"Error loading embedding {embedding_path}: {e}")
                embedding = self._create_embedding_from_features(row)
        else:
            # Create embedding from features if file doesn't exist
            embedding = self._create_embedding_from_features(row)
        
        # Normalize embedding
        embedding = embedding.astype(np.float32)
        if embedding.std() > 1e-8:
            embedding = (embedding - embedding.mean()) / embedding.std()
        
        return torch.tensor(embedding, dtype=torch.float32)
    
    def _create_embedding_from_features(self, row):
        """Create 12x12x3 embedding from 144 features"""
        features = row[self.feature_columns].values.astype(np.float32)
        
        # Verify exactly 144 features
        if len(features) != 144:
            print(f"Warning: Expected 144 features, got {len(features)}")
            # Pad or truncate to exactly 144
            if len(features) < 144:
                padded = np.zeros(144, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            else:
                features = features[:144]
        
        # Split into the three regions
        cell_features = features[0:48]
        nucleus_features = features[48:96]
        cytoplasm_features = features[96:144]
        
        # Create 12x12x3 embedding
        embedding = np.zeros((12, 12, 3), dtype=np.float32)
        
        # Fill each channel with its 48 features
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

class MultiHeadRegionAttention(nn.Module):
    """Multi-head attention for region-attention embeddings"""
    def __init__(self, embed_dim=144, num_heads=3, dropout=0.1):
        super(MultiHeadRegionAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """Forward pass with proper tensor handling"""
        batch_size, height, width, channels = x.shape
        seq_len = height * width  # 144 spatial locations
        x_seq = x.view(batch_size, seq_len, channels)  # [batch, 144, 3]
        
        # Project to higher dimension for attention
        if channels != self.embed_dim:
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(channels, self.embed_dim).to(x.device)
            x_seq = self.input_projection(x_seq)
        
        # Apply layer normalization
        x_norm = self.layer_norm(x_seq)
        
        # Generate Q, K, V
        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final linear transformation
        output = self.fc_out(attended)
        
        # Residual connection
        output = output + x_seq
        
        # Project back to original channel dimension if needed
        if self.embed_dim != channels:
            if not hasattr(self, 'output_projection'):
                self.output_projection = nn.Linear(self.embed_dim, channels).to(x.device)
            output = self.output_projection(output)
        
        # Reshape back to spatial format
        output = output.view(batch_size, height, width, channels)
        
        return output

class SpatialChannelAttention(nn.Module):
    """Spatial and Channel Attention Module (SCAM)"""
    def __init__(self, in_channels, reduction=16):
        super(SpatialChannelAttention, self).__init__()
        
        self.in_channels = in_channels
        
        # Channel attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(1, in_channels // reduction)
        
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        """Forward pass with proper error handling"""
        batch_size, channels, height, width = x.shape
        
        # Channel attention
        avg_pool = self.global_avg_pool(x).view(batch_size, channels)
        max_pool = self.global_max_pool(x).view(batch_size, channels)
        
        avg_out = self.channel_mlp(avg_pool)
        max_out = self.channel_mlp(max_pool)
        
        channel_att = torch.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        x_channel = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))
        x_spatial = x_channel * spatial_att
        
        return x_spatial

class EnhancedXception(nn.Module):
    """Enhanced Xception with attention modules for region-attention embeddings"""
    def __init__(self, num_classes=21, input_shape=(3, 12, 12)):
        super(EnhancedXception, self).__init__()
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Multi-head region attention
        self.region_attention = MultiHeadRegionAttention(embed_dim=144, num_heads=3)
        
        # Entry flow - adapted for 12x12 input
        self.entry_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.entry_bn1 = nn.BatchNorm2d(32)
        self.entry_scam1 = SpatialChannelAttention(32)
        
        self.entry_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.entry_bn2 = nn.BatchNorm2d(64)
        self.entry_scam2 = SpatialChannelAttention(64)
        
        # Depthwise separable convolutions
        self.depthwise1 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise1 = nn.Conv2d(64, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.scam1 = SpatialChannelAttention(128)
        
        self.depthwise2 = nn.Conv2d(128, 128, 3, padding=1, groups=128)
        self.pointwise2 = nn.Conv2d(128, 256, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.scam2 = SpatialChannelAttention(256)
        
        self.depthwise3 = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.pointwise3 = nn.Conv2d(256, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.scam3 = SpatialChannelAttention(512)
        
        # Global pooling and feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 1024)  # Reduced size for fusion
        
    def forward(self, x):
        """Forward pass with proper tensor handling"""
        # Ensure input is in CHW format for conv layers
        if x.dim() == 4 and x.shape[-1] == 3:
            x_chw = x.permute(0, 3, 1, 2)  # [batch, 3, 12, 12]
            x_hwc = x  # Keep original for attention
        else:
            x_chw = x
            x_hwc = x.permute(0, 2, 3, 1)  # [batch, 12, 12, 3]
        
        # Apply region attention
        x_attended = self.region_attention(x_hwc)  # [batch, 12, 12, 3]
        
        # Convert back to CHW for conv layers
        x = x_attended.permute(0, 3, 1, 2)  # [batch, 3, 12, 12]
        
        # Entry flow
        x = F.relu(self.entry_bn1(self.entry_conv1(x)))
        x = self.entry_scam1(x)
        x = F.relu(self.entry_bn2(self.entry_conv2(x)))
        x = self.entry_scam2(x)
        
        # Depthwise separable convolutions with attention
        x = F.relu(self.bn1(self.pointwise1(self.depthwise1(x))))
        x = self.scam1(x)
        
        x = F.relu(self.bn2(self.pointwise2(self.depthwise2(x))))
        x = self.scam2(x)
        
        x = F.relu(self.bn3(self.pointwise3(self.depthwise3(x))))
        x = self.scam3(x)
        
        # Global pooling and feature extraction
        x = self.global_pool(x)  # [batch, 512, 1, 1]
        x = torch.flatten(x, 1)  # [batch, 512]
        x = self.dropout(x)
        x = self.fc(x)  # [batch, 1024]
        
        return x

class XGBoostWrapper:
    """
    XGBoost wrapper for traditional feature classification
    """
    def __init__(self, num_classes=21, **xgb_params):
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Default XGBoost parameters optimized for bone marrow classification
        self.default_params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Update with user parameters
        self.default_params.update(xgb_params)
        
        print(f"🌲 XGBoost configured for {num_classes} classes")
        
    def fit(self, X, y, eval_set=None, early_stopping_rounds=10):
        """Train XGBoost model"""
        print("🌲 Training XGBoost model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create XGBoost model
        self.model = xgb.XGBClassifier(**self.default_params)
        
        # Prepare evaluation set if provided
        eval_set_scaled = None
        if eval_set is not None:
            X_val_scaled = self.scaler.transform(eval_set[0])
            eval_set_scaled = [(X_val_scaled, eval_set[1])]
        
        # Train model
        if eval_set_scaled:
            self.model.fit(X_scaled, y,eval_set=eval_set_scaled,verbose=False)
        else:
            self.model.fit(X_scaled, y, verbose=False)


        
        self.is_fitted = True
        print("✅ XGBoost training completed")
        
        return self
    
    def predict(self, X):
        """Predict classes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        
        # Ensure correct shape for both binary and multiclass
        if self.num_classes == 2 and probs.shape[1] == 1:
            # Binary case with single column output
            probs = np.column_stack([1 - probs.flatten(), probs.flatten()])
        
        return probs
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.feature_importances_
    
    def save(self, filepath):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'num_classes': self.num_classes,
            'is_fitted': self.is_fitted,
            'params': self.default_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load model and scaler"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.num_classes = model_data['num_classes']
        self.is_fitted = model_data['is_fitted']
        self.default_params = model_data['params']

class FusionModule(nn.Module):
    """
    Fusion module for combining Xception and XGBoost features
    """
    def __init__(self, xception_dim=1024, xgboost_dim=21, fusion_dim=512, num_classes=21):
        super(FusionModule, self).__init__()
        
        self.xception_dim = xception_dim
        self.xgboost_dim = xgboost_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # Feature alignment layers
        self.xception_align = nn.Sequential(
            nn.Linear(xception_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.xgboost_align = nn.Sequential(
            nn.Linear(xgboost_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Attention-based fusion
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, xception_features, xgboost_probs):
        """
        Fuse features from both models
        """
        # Align feature dimensions
        xception_aligned = self.xception_align(xception_features)  # [batch, fusion_dim]
        xgboost_aligned = self.xgboost_align(xgboost_probs)       # [batch, fusion_dim]
        
        # Concatenate for attention computation
        combined = torch.cat([xception_aligned, xgboost_aligned], dim=1)  # [batch, fusion_dim*2]
        
        # Compute attention weights
        attention_weights = self.attention(combined)  # [batch, 2]
        
        # Apply attention weights
        weighted_xception = xception_aligned * attention_weights[:, 0:1]
        weighted_xgboost = xgboost_aligned * attention_weights[:, 1:2]
        
        # Fused features
        fused_features = weighted_xception + weighted_xgboost  # [batch, fusion_dim]
        
        # Final predictions
        logits = self.classifier(fused_features)
        uncertainty = self.uncertainty_estimator(fused_features)
        
        return logits, uncertainty, attention_weights

class HybridXceptionXGBoost(nn.Module):
    """
    Complete hybrid model combining Enhanced Xception and XGBoost
    """
    def __init__(self, num_classes=21, xgb_params=None):
        super(HybridXceptionXGBoost, self).__init__()
        
        self.num_classes = num_classes
        print(f"🔧 Building Hybrid Xception + XGBoost Model for {num_classes} classes")
        
        # Xception pathway for embeddings
        self.xception_net = EnhancedXception(num_classes=num_classes)
        
        # XGBoost wrapper for traditional features
        if xgb_params is None:
            xgb_params = {}
        self.xgboost_wrapper = XGBoostWrapper(num_classes=num_classes, **xgb_params)
        
        # Fusion module
        self.fusion = FusionModule(
            xception_dim=1024,
            xgboost_dim=num_classes,
            fusion_dim=512,
            num_classes=num_classes
        )
        
        print("✅ Hybrid Xception + XGBoost Model built successfully")
        
    def train_xgboost(self, train_features, train_labels, val_features=None, val_labels=None):
        """Train the XGBoost component"""
        eval_set = None
        if val_features is not None and val_labels is not None:
            eval_set = (val_features, val_labels)
        
        self.xgboost_wrapper.fit(train_features, train_labels, eval_set=eval_set)
        
    def forward(self, embedding=None, features=None, mode='both'):
        """
        Forward pass with flexible input modes
        """
        if mode == 'embedding_only' and embedding is not None:
            # Only Xception pathway
            xception_features = self.xception_net(embedding)
            # Simple classifier for Xception features
            logits = torch.mm(xception_features, 
                            torch.randn(xception_features.shape[1], self.num_classes).to(xception_features.device))
            uncertainty = torch.sigmoid(torch.sum(xception_features, dim=1, keepdim=True)) * 0.5
            return logits, uncertainty
            
        elif mode == 'features_only' and features is not None:
            # Only XGBoost pathway (requires CPU processing)
            features_np = features.detach().cpu().numpy()
            if not self.xgboost_wrapper.is_fitted:
                raise ValueError("XGBoost model must be trained before inference")
            
            xgboost_probs = self.xgboost_wrapper.predict_proba(features_np)
            xgboost_probs_tensor = torch.tensor(xgboost_probs, dtype=torch.float32).to(features.device)
            
            logits = torch.log(xgboost_probs_tensor + 1e-8)  # Convert probs to logits
            uncertainty = torch.sum(-xgboost_probs_tensor * torch.log(xgboost_probs_tensor + 1e-8), dim=1, keepdim=True) / np.log(self.num_classes)
            return logits, uncertainty
            
        elif mode == 'both' and embedding is not None and features is not None:
            # Both pathways with fusion
            # Get Xception features
            xception_features = self.xception_net(embedding)
            
            # Get XGBoost predictions
            features_np = features.detach().cpu().numpy()
            if not self.xgboost_wrapper.is_fitted:
                raise ValueError("XGBoost model must be trained before inference")
            
            xgboost_probs = self.xgboost_wrapper.predict_proba(features_np)
            xgboost_probs_tensor = torch.tensor(xgboost_probs, dtype=torch.float32).to(features.device)
            
            # Fusion
            logits, uncertainty, attention_weights = self.fusion(xception_features, xgboost_probs_tensor)
            return logits, uncertainty, attention_weights
            
        else:
            raise ValueError(f"Invalid mode {mode} or missing required inputs")

class HybridLoss(nn.Module):
    """
    Combined loss function for the hybrid Xception + XGBoost model
    """
    def __init__(self, num_classes=21, class_weights=None, 
                 alpha_ce=1.0, alpha_uncertainty=0.1, alpha_consistency=0.1):
        super(HybridLoss, self).__init__()
        
        self.alpha_ce = alpha_ce
        self.alpha_uncertainty = alpha_uncertainty
        self.alpha_consistency = alpha_consistency
        
        # Primary classification loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Uncertainty regularization
        self.uncertainty_loss = nn.MSELoss()
        
    def forward(self, logits, uncertainty, targets, consistency_target=None):
        """
        Compute hybrid loss with proper weighting
        """
        # Classification loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Uncertainty regularization
        predicted = torch.argmax(logits, dim=1)
        correct_mask = (predicted == targets).float()
        
        # Target uncertainty: low (0.1) for correct, high (0.9) for incorrect
        target_uncertainty = 0.1 + 0.8 * (1 - correct_mask)
        uncertainty_reg = self.uncertainty_loss(uncertainty.squeeze(), target_uncertainty)
        
        # Total loss
        total_loss = (self.alpha_ce * ce_loss + 
                     self.alpha_uncertainty * uncertainty_reg)
        
        # Consistency loss (if provided)
        if consistency_target is not None:
            consistency_loss = F.mse_loss(torch.softmax(logits, dim=1), consistency_target)
            total_loss += self.alpha_consistency * consistency_loss
            return total_loss, ce_loss, uncertainty_reg, consistency_loss
        
        return total_loss, ce_loss, uncertainty_reg

def convert_numpy_types(obj):
    """Recursively convert numpy types to JSON serializable Python types"""
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

def calculate_class_weights(labels):
    """Calculate class weights for handling imbalanced dataset"""
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

def train_xgboost_component(model, train_loader, val_loader):
    """Train the XGBoost component separately"""
    print("🌲 Training XGBoost component...")
    
    # Collect all features and labels
    train_features_list = []
    train_labels_list = []
    val_features_list = []
    val_labels_list = []
    
    # Extract training features
    for batch in tqdm(train_loader, desc="Extracting train features"):
        if 'features' in batch:
            train_features_list.append(batch['features'].numpy())
            train_labels_list.append(batch['label'].numpy())
    
    # Extract validation features
    for batch in tqdm(val_loader, desc="Extracting val features"):
        if 'features' in batch:
            val_features_list.append(batch['features'].numpy())
            val_labels_list.append(batch['label'].numpy())
    
    # Combine features
    train_features = np.vstack(train_features_list)
    train_labels = np.hstack(train_labels_list)
    val_features = np.vstack(val_features_list) if val_features_list else None
    val_labels = np.hstack(val_labels_list) if val_labels_list else None
    
    print(f"📊 Training XGBoost on {len(train_features)} samples")
    if val_features is not None:
        print(f"📊 Validation set: {len(val_features)} samples")
    
    # Train XGBoost
    model.train_xgboost(train_features, train_labels, val_features, val_labels)
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, mode='both', epoch=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_uncertainty_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"🔥 Training Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move data to device
            targets = batch['label'].to(device)
            
            # Prepare inputs based on mode
            inputs = {}
            if mode in ['both', 'embedding_only'] and 'embedding' in batch:
                inputs['embedding'] = batch['embedding'].to(device)
            if mode in ['both', 'features_only'] and 'features' in batch:
                inputs['features'] = batch['features'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if mode == 'both':
                logits, uncertainty, attention_weights = model(mode=mode, **inputs)
            else:
                logits, uncertainty = model(mode=mode, **inputs)
            
            # Calculate loss
            loss, ce_loss, uncertainty_loss = criterion(logits, uncertainty, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_uncertainty_loss += uncertainty_loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'CE': f'{ce_loss.item():.4f}',
                    'Unc': f'{uncertainty_loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
                
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
    
    return {
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'uncertainty_loss': total_uncertainty_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }

def validate_epoch(model, dataloader, criterion, device, mode='both', epoch=0):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_uncertainty_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"✅ Validation Epoch {epoch+1}")):
            try:
                # Move data to device
                targets = batch['label'].to(device)
                
                # Prepare inputs based on mode
                inputs = {}
                if mode in ['both', 'embedding_only'] and 'embedding' in batch:
                    inputs['embedding'] = batch['embedding'].to(device)
                if mode in ['both', 'features_only'] and 'features' in batch:
                    inputs['features'] = batch['features'].to(device)
                
                # Forward pass
                if mode == 'both':
                    logits, uncertainty, attention_weights = model(mode=mode, **inputs)
                    all_attention_weights.extend(attention_weights.cpu().numpy())
                else:
                    logits, uncertainty = model(mode=mode, **inputs)
                
                # Calculate loss
                loss, ce_loss, uncertainty_loss = criterion(logits, uncertainty, targets)
                
                # Statistics
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_uncertainty_loss += uncertainty_loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store predictions for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy().flatten())
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    result = {
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'uncertainty_loss': total_uncertainty_loss / len(dataloader),
        'accuracy': 100. * correct / total,
        'predictions': all_predictions,
        'targets': all_targets,
        'uncertainties': all_uncertainties
    }
    
    if all_attention_weights:
        result['attention_weights'] = all_attention_weights
    
    return result

def create_data_loaders(csv_file, image_dir, embeddings_dir, mode, batch_size, test_size=0.4):
    """Create train and validation data loaders"""
    print("📦 Creating data loaders...")
    
    # Load full dataset to split
    full_df = pd.read_csv(csv_file)
    
    # Check required columns
    if 'image_name' not in full_df.columns or 'label' not in full_df.columns:
        raise ValueError("CSV file must contain 'image_name' and 'label' columns")
    
    print(f"Loaded {len(full_df)} samples with {full_df['label'].nunique()} unique classes")
    print(f"Label distribution:\n{full_df['label'].value_counts()}")
    
    # Stratified split using original labels
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
    train_df.to_csv('train_split_xgb.csv', index=False)
    test_df.to_csv('test_split_xgb.csv', index=False)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = BoneMarrowDataset('train_split_xgb.csv', image_dir, embeddings_dir, 
                                    None, mode)
    test_dataset = BoneMarrowDataset('test_split_xgb.csv', image_dir, embeddings_dir, 
                                   None, mode)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=torch.cuda.is_available())
    
    return train_loader, test_loader, train_dataset.label_encoder

def evaluate_model(predictions, targets, uncertainties, label_encoder, attention_weights=None, save_path=None):
    """Comprehensive model evaluation with uncertainty and attention analysis"""
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(targets, predictions, 
                                 target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Uncertainty statistics
    mean_uncertainty = np.mean(uncertainties)
    correct_mask = np.array(predictions) == np.array(targets)
    correct_uncertainty = np.mean(np.array(uncertainties)[correct_mask])
    incorrect_uncertainty = np.mean(np.array(uncertainties)[~correct_mask])
    
    # Attention statistics (if available)
    attention_stats = {}
    if attention_weights is not None:
        attention_weights = np.array(attention_weights)
        attention_stats = {
            'mean_xception_weight': np.mean(attention_weights[:, 0]),
            'mean_xgboost_weight': np.mean(attention_weights[:, 1]),
            'std_xception_weight': np.std(attention_weights[:, 0]),
            'std_xgboost_weight': np.std(attention_weights[:, 1])
        }
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    if save_path:
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Confusion matrix
        if SNS_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
        else:
            im = axes[0,0].imshow(cm, interpolation='nearest', cmap='Blues')
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[0,0].text(j, i, str(cm[i, j]), ha='center', va='center')
        
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # Uncertainty distribution
        axes[0,1].hist(np.array(uncertainties)[correct_mask], alpha=0.7, label='Correct', bins=30)
        axes[0,1].hist(np.array(uncertainties)[~correct_mask], alpha=0.7, label='Incorrect', bins=30)
        axes[0,1].set_title('Uncertainty Distribution')
        axes[0,1].set_xlabel('Uncertainty')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend()
        
        # Per-class F1 scores
        f1_scores = [report[cls]['f1-score'] for cls in class_names]
        axes[1,0].bar(range(len(class_names)), f1_scores, alpha=0.7)
        axes[1,0].set_title('Per-Class F1 Scores')
        axes[1,0].set_xlabel('Classes')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].set_xticks(range(len(class_names)))
        axes[1,0].set_xticklabels(class_names, rotation=45)
        
        # Attention weights (if available)
        if attention_weights is not None:
            attention_weights = np.array(attention_weights)
            axes[1,1].boxplot([attention_weights[:, 0], attention_weights[:, 1]], 
                             labels=['Xception', 'XGBoost'])
            axes[1,1].set_title('Attention Weight Distribution')
            axes[1,1].set_ylabel('Attention Weight')
        else:
            axes[1,1].text(0.5, 0.5, 'No Attention Weights\nAvailable', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Attention Analysis')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    result = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm,
        'mean_uncertainty': mean_uncertainty,
        'correct_uncertainty': correct_uncertainty,
        'incorrect_uncertainty': incorrect_uncertainty
    }
    
    result.update(attention_stats)
    return result

def plot_feature_importance(model, feature_names, save_path):
    """Plot XGBoost feature importance"""
    if not model.xgboost_wrapper.is_fitted:
        print("⚠️  XGBoost model not fitted, skipping feature importance plot")
        return
    
    importance = model.xgboost_wrapper.get_feature_importance()
    
    # Get top 20 features
    top_indices = np.argsort(importance)[-20:]
    top_importance = importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_importance)), top_importance)
    plt.yticks(range(len(top_importance)), top_names)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function for hybrid Xception + XGBoost model"""
    parser = argparse.ArgumentParser(description='Hybrid Xception + XGBoost Bone Marrow Classification')
    parser.add_argument('--csv_file', required=True, help='Features CSV file')
    parser.add_argument('--image_dir', required=True, help='Images directory')
    parser.add_argument('--embeddings_dir', required=True, help='Embeddings directory')
    parser.add_argument('--save_dir', default='results_xgb_hybrid', help='Save directory')
    parser.add_argument('--mode', default='both', choices=['both', 'embedding_only', 'features_only'])
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # XGBoost specific parameters
    parser.add_argument('--xgb_max_depth', type=int, default=6, help='XGBoost max depth')
    parser.add_argument('--xgb_learning_rate', type=float, default=0.1, help='XGBoost learning rate')
    parser.add_argument('--xgb_n_estimators', type=int, default=200, help='XGBoost n_estimators')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Using device: {device}")
    
    # Optional TensorBoard setup
    writer = None
    if TENSORBOARD_AVAILABLE:
        try:
            writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard'))
            print("📊 TensorBoard logging enabled")
        except Exception as e:
            print(f"⚠️  TensorBoard setup failed: {e}")
            writer = None
    
    # Create data loaders
    train_loader, test_loader, label_encoder = create_data_loaders(
        args.csv_file, args.image_dir, args.embeddings_dir, 
        args.mode, args.batch_size
    )
    
    num_classes = len(label_encoder.classes_)
    print(f"📊 Training on {num_classes} classes")
    
    # XGBoost parameters
    xgb_params = {
        'max_depth': args.xgb_max_depth,
        'learning_rate': args.xgb_learning_rate,
        'n_estimators': args.xgb_n_estimators,
    }
    
    # Create model
    model = HybridXceptionXGBoost(num_classes=num_classes, xgb_params=xgb_params).to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Neural model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train XGBoost component first (if using features)
    if args.mode in ['both', 'features_only']:
        model = train_xgboost_component(model, train_loader, test_loader)
        
        # Save XGBoost model
        model.xgboost_wrapper.save(os.path.join(args.save_dir, 'xgboost_model.pkl'))
        print("💾 XGBoost model saved")
    
    # Calculate class weights for imbalanced dataset
    train_df = pd.read_csv('train_split_xgb.csv')
    train_label_encoder = LabelEncoder()
    train_encoded_labels = train_label_encoder.fit_transform(train_df['label'])
    class_weights_dict = calculate_class_weights(train_encoded_labels)
    class_weights = [class_weights_dict[i] for i in range(num_classes)]
    
    # Loss and optimizer (only for neural components)
    criterion = HybridLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        alpha_ce=1.0,
        alpha_uncertainty=0.1,
        alpha_consistency=0.1
    ).to(device)
    
    # Only optimize neural network parameters
    neural_params = list(model.xception_net.parameters()) + list(model.fusion.parameters())
    optimizer = optim.AdamW(neural_params, lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"🚀 Starting training for {args.num_epochs} epochs")
    
    best_f1 = 0.0
    early_stop_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_ce_loss': [], 'train_uncertainty_loss': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_ce_loss': [], 'val_uncertainty_loss': []
    }
    
    for epoch in range(args.num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.num_epochs}")
        
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, args.mode, epoch)
        
        # Validation
        val_metrics = validate_epoch(model, test_loader, criterion, device, args.mode, epoch)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != current_lr:
            print(f"📉 Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
        
        # Evaluate detailed metrics
        eval_metrics = evaluate_model(
            val_metrics['predictions'], 
            val_metrics['targets'],
            val_metrics['uncertainties'],
            label_encoder,
            val_metrics.get('attention_weights'),
            save_path=os.path.join(args.save_dir, f'evaluation_epoch_{epoch+1}.png')
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_ce_loss'].append(train_metrics['ce_loss'])
        history['train_uncertainty_loss'].append(train_metrics['uncertainty_loss'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(eval_metrics['f1_weighted'])
        history['val_ce_loss'].append(val_metrics['ce_loss'])
        history['val_uncertainty_loss'].append(val_metrics['uncertainty_loss'])
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            writer.add_scalar('F1_Score/Validation', eval_metrics['f1_weighted'], epoch)
            writer.add_scalar('Uncertainty/Mean', eval_metrics['mean_uncertainty'], epoch)
            
            # Log attention weights if available
            if 'mean_xception_weight' in eval_metrics:
                writer.add_scalar('Attention/Xception_Weight', eval_metrics['mean_xception_weight'], epoch)
                writer.add_scalar('Attention/XGBoost_Weight', eval_metrics['mean_xgboost_weight'], epoch)
        
        # Print metrics
        print(f"Train: Loss={train_metrics['loss']:.4f}, CE={train_metrics['ce_loss']:.4f}, "
              f"Unc={train_metrics['uncertainty_loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
        print(f"Val: Loss={val_metrics['loss']:.4f}, CE={val_metrics['ce_loss']:.4f}, "
              f"Unc={val_metrics['uncertainty_loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
        print(f"F1: {eval_metrics['f1_weighted']:.4f}, Uncertainty: {eval_metrics['mean_uncertainty']:.4f}")
        
        if 'mean_xception_weight' in eval_metrics:
            print(f"Attention - Xception: {eval_metrics['mean_xception_weight']:.3f}, "
                  f"XGBoost: {eval_metrics['mean_xgboost_weight']:.3f}")
        
        # Save best model
        if eval_metrics['f1_weighted'] > best_f1:
            best_f1 = eval_metrics['f1_weighted']
            early_stop_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'label_encoder': label_encoder,
                'args': args,
                'eval_metrics': eval_metrics
            }, os.path.join(args.save_dir, 'best_hybrid_model.pth'))
            
            # Save detailed metrics
            with open(os.path.join(args.save_dir, 'best_metrics.json'), 'w') as f:
                serializable_metrics = convert_numpy_types(eval_metrics)
                json.dump(serializable_metrics, f, indent=2)
            
            print(f"🎯 New best F1 score: {best_f1:.4f}")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= args.patience:
            print(f"⏹️  Early stopping triggered after {args.patience} epochs without improvement")
            break
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    # Save final models and history
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_hybrid_model.pth'))
    
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot feature importance for XGBoost
    if args.mode in ['both', 'features_only']:
        feature_columns = [col for col in pd.read_csv(args.csv_file).columns 
                          if col not in ['image_name', 'label']]
        plot_feature_importance(model, feature_columns, 
                              os.path.join(args.save_dir, 'xgboost_feature_importance.png'))
    
    # Create comprehensive training summary plot
    plt.figure(figsize=(24, 12))
    
    # Loss plots
    plt.subplot(3, 4, 1)
    plt.plot(history['train_loss'], label='Train Total', linewidth=2)
    plt.plot(history['val_loss'], label='Val Total', linewidth=2)
    plt.title('Total Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 2)
    plt.plot(history['train_ce_loss'], label='Train CE', linewidth=2)
    plt.plot(history['val_ce_loss'], label='Val CE', linewidth=2)
    plt.title('Classification Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 3)
    plt.plot(history['train_uncertainty_loss'], label='Train Unc', linewidth=2)
    plt.plot(history['val_uncertainty_loss'], label='Val Unc', linewidth=2)
    plt.title('Uncertainty Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 4)
    plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
    plt.title('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 5)
    plt.plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
    plt.title('F1 Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Per-class F1 scores
    plt.subplot(3, 4, 6)
    final_report = eval_metrics['classification_report']
    classes = label_encoder.classes_
    f1_scores = [final_report[cls]['f1-score'] for cls in classes]
    
    plt.bar(range(len(classes)), f1_scores, alpha=0.7)
    plt.title('Per-Class F1 Scores', fontsize=12)
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.xticks(range(len(classes)), classes, rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Learning curves comparison
    plt.subplot(3, 4, 7)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.title('Learning Curves', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uncertainty analysis
    plt.subplot(3, 4, 8)
    plt.text(0.1, 0.9, f"Mean Uncertainty: {eval_metrics['mean_uncertainty']:.4f}", fontsize=10)
    plt.text(0.1, 0.8, f"Correct Uncertainty: {eval_metrics['correct_uncertainty']:.4f}", fontsize=10)
    plt.text(0.1, 0.7, f"Incorrect Uncertainty: {eval_metrics['incorrect_uncertainty']:.4f}", fontsize=10)
    if 'mean_xception_weight' in eval_metrics:
        plt.text(0.1, 0.6, f"Avg Xception Weight: {eval_metrics['mean_xception_weight']:.3f}", fontsize=10)
        plt.text(0.1, 0.5, f"Avg XGBoost Weight: {eval_metrics['mean_xgboost_weight']:.3f}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Uncertainty & Attention Stats', fontsize=12)
    
    # Model architecture info
    plt.subplot(3, 4, 9)
    plt.text(0.1, 0.9, f"Hybrid Xception + XGBoost", fontsize=10, weight='bold')
    plt.text(0.1, 0.8, f"Classes: {num_classes}", fontsize=10)
    plt.text(0.1, 0.7, f"Mode: {args.mode}", fontsize=10)
    plt.text(0.1, 0.6, f"Neural Params: {total_params:,}", fontsize=10)
    plt.text(0.1, 0.5, f"XGB Max Depth: {args.xgb_max_depth}", fontsize=10)
    plt.text(0.1, 0.4, f"XGB N Estimators: {args.xgb_n_estimators}", fontsize=10)
    plt.text(0.1, 0.3, f"XGB Learning Rate: {args.xgb_learning_rate}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Model Architecture', fontsize=12)
    
    # Performance summary
    plt.subplot(3, 4, 10)
    plt.text(0.1, 0.9, f"Best F1: {best_f1:.4f}", fontsize=12, weight='bold', color='green')
    plt.text(0.1, 0.8, f"Final Accuracy: {eval_metrics['accuracy']:.4f}", fontsize=11)
    plt.text(0.1, 0.7, f"Macro F1: {eval_metrics['f1_macro']:.4f}", fontsize=11)
    plt.text(0.1, 0.6, f"Weighted F1: {eval_metrics['f1_weighted']:.4f}", fontsize=11)
    plt.text(0.1, 0.5, f"Epochs Trained: {epoch+1}/{args.num_epochs}", fontsize=10)
    plt.text(0.1, 0.4, f"Batch Size: {args.batch_size}", fontsize=10)
    plt.text(0.1, 0.3, f"Learning Rate: {args.learning_rate}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Performance Summary', fontsize=12)
    
    # Training efficiency
    plt.subplot(3, 4, 11)
    epochs_range = range(1, len(history['train_loss']) + 1)
    train_acc_curve = history['train_acc']
    val_acc_curve = history['val_acc']
    
    plt.plot(epochs_range, train_acc_curve, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc_curve, 'r-', label='Validation Accuracy', linewidth=2)
    plt.fill_between(epochs_range, train_acc_curve, val_acc_curve, alpha=0.3)
    plt.title('Training Efficiency', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final metrics comparison
    plt.subplot(3, 4, 12)
    metrics_names = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    metrics_values = [eval_metrics['accuracy'], eval_metrics['f1_macro'], eval_metrics['f1_weighted']]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    plt.title('Final Metrics Comparison', fontsize=12)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'comprehensive_training_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a detailed model comparison plot
    if args.mode == 'both':
        plt.figure(figsize=(12, 8))
        
        # Attention weights distribution over time (if available)
        if 'attention_weights' in val_metrics:
            attention_weights = np.array(val_metrics['attention_weights'])
            
            plt.subplot(2, 2, 1)
            plt.hist(attention_weights[:, 0], alpha=0.7, label='Xception Weights', bins=30)
            plt.hist(attention_weights[:, 1], alpha=0.7, label='XGBoost Weights', bins=30)
            plt.title('Attention Weight Distribution')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.scatter(attention_weights[:, 0], attention_weights[:, 1], alpha=0.6)
            plt.xlabel('Xception Weight')
            plt.ylabel('XGBoost Weight')
            plt.title('Attention Weight Correlation')
            plt.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Perfect Anti-correlation')
            plt.legend()
            
            # Attention weights vs accuracy
            plt.subplot(2, 2, 3)
            correct_predictions = np.array(val_metrics['predictions']) == np.array(val_metrics['targets'])
            
            correct_xception_weights = attention_weights[correct_predictions, 0]
            incorrect_xception_weights = attention_weights[~correct_predictions, 0]
            
            plt.boxplot([correct_xception_weights, incorrect_xception_weights], 
                       labels=['Correct', 'Incorrect'])
            plt.title('Xception Attention vs Prediction Accuracy')
            plt.ylabel('Xception Weight')
            
            plt.subplot(2, 2, 4)
            correct_xgboost_weights = attention_weights[correct_predictions, 1]
            incorrect_xgboost_weights = attention_weights[~correct_predictions, 1]
            
            plt.boxplot([correct_xgboost_weights, incorrect_xgboost_weights], 
                       labels=['Correct', 'Incorrect'])
            plt.title('XGBoost Attention vs Prediction Accuracy')
            plt.ylabel('XGBoost Weight')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'attention_analysis.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\n✅ Training completed!")
    print(f"🎯 Best F1 Score: {best_f1:.4f}")
    print(f"📊 Final Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"🔮 Mean Uncertainty: {eval_metrics['mean_uncertainty']:.4f}")
    
    if 'mean_xception_weight' in eval_metrics:
        print(f"⚖️  Average Attention - Xception: {eval_metrics['mean_xception_weight']:.3f}, "
              f"XGBoost: {eval_metrics['mean_xgboost_weight']:.3f}")
    
    print(f"💾 Results saved to: {args.save_dir}")
    
    if TENSORBOARD_AVAILABLE:
        print(f"📊 View TensorBoard: tensorboard --logdir {args.save_dir}/tensorboard")
    
    # Print final summary
    print("\n" + "="*60)
    print("🎊 HYBRID XCEPTION + XGBOOST TRAINING COMPLETE")
    print("="*60)
    print(f"📈 Best Performance:")
    print(f"   • F1 Score (Weighted): {eval_metrics['f1_weighted']:.4f}")
    print(f"   • F1 Score (Macro): {eval_metrics['f1_macro']:.4f}")
    print(f"   • Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"📊 Model Complexity:")
    print(f"   • Neural Network Parameters: {total_params:,}")
    print(f"   • XGBoost Trees: {args.xgb_n_estimators}")
    print(f"🧠 Uncertainty Insights:")
    print(f"   • Avg Uncertainty (Correct): {eval_metrics['correct_uncertainty']:.4f}")
    print(f"   • Avg Uncertainty (Incorrect): {eval_metrics['incorrect_uncertainty']:.4f}")
    
    if args.mode == 'both' and 'mean_xception_weight' in eval_metrics:
        print(f"⚖️  Fusion Balance:")
        print(f"   • Xception Contribution: {eval_metrics['mean_xception_weight']:.1%}")
        print(f"   • XGBoost Contribution: {eval_metrics['mean_xgboost_weight']:.1%}")
    
    print("="*60)

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Hybrid Xception + XGBoost Bone Marrow Cell Classification Model
Combines:
- Enhanced Xception for region-attention embeddings
- XGBoost for traditional feature classification
- Cross-modal fusion and uncertainty estimation
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import math
import pickle
from typing import Dict, List, Tuple, Optional

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("WARNING: TensorBoard not available - training will continue without logging")

# Torchvision for image processing
from torchvision import transforms

# XGBoost imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ERROR: XGBoost not available! Please install: pip install xgboost")
    exit(1)

# Scientific computing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Visualization
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False
    print("WARNING: Seaborn not available - using basic matplotlib plots")

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Starting Hybrid Xception + XGBoost Bone Marrow Classification Model")
print(f"PyTorch Version: {torch.__version__}")
print(f"XGBoost Version: {xgb.__version__}")
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
    Dataset for hybrid Xception + XGBoost model
    Handles both region-attention embeddings and traditional features
    """
    def __init__(self, csv_file, image_dir, embeddings_dir, transform=None, mode='both'):
        print(f"Loading dataset from {csv_file}")
        self.df = pd.read_csv(args.csv_file)
        self.image_dir = Path(image_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.transform = transform
        self.mode = mode
        
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
        print(f"Feature columns: {len(self.feature_columns)}")
        
        # Validate feature structure
        self._validate_feature_structure()
        
        # Validate data availability
        self._validate_data()
        
    def _validate_feature_structure(self):
        """Validate that features match the expected CSV structure"""
        expected_regions = ['cell', 'nucleus', 'cytoplasm']
        expected_features_per_region = 48
        
        # Count features per region
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
            print("Perfect! Your CSV structure matches the paper exactly.")
        else:
            print("WARNING: Feature structure doesn't match expected pattern.")
            print("   This might still work, but performance could be affected.")
        
    def _validate_data(self):
        """Validate that required files exist"""
        missing_embeddings = 0
        
        sample_size = min(100, len(self.df))  # Check first 100 samples
        for i in range(sample_size):
            row = self.df.iloc[i]
            image_name = row['image_name']
            
            # Check embeddings
            if self.mode in ['both', 'embedding_only']:
                embedding_path = self.embeddings_dir / f"{image_name}_embedding_3ch.npy"
                if not embedding_path.exists():
                    missing_embeddings += 1
            
        if missing_embeddings > 0:
            print(f"WARNING: {missing_embeddings}/{sample_size} sample embeddings not found")
            
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
        
        # Load embedding if needed
        if self.mode in ['both', 'embedding_only']:
            embedding = self._load_embedding(image_name, row)
            sample['embedding'] = embedding
            
        # Load features for XGBoost
        if self.mode in ['both', 'features_only']:
            features = row[self.feature_columns].values.astype(np.float32)
            sample['features'] = torch.tensor(features, dtype=torch.float32)
            
        return sample
    
    def _load_embedding(self, image_name, row):
        """Load region-attention embedding with proper error handling"""
        embedding_path = self.embeddings_dir / f"{image_name}_embedding_3ch.npy"
        
        if embedding_path.exists():
            try:
                embedding = np.load(embedding_path).astype(np.float32)
                # Ensure correct shape [12, 12, 3]
                if embedding.shape != (12, 12, 3):
                    print(f"Warning: Embedding {image_name} has shape {embedding.shape}, expected (12, 12, 3)")
                    if embedding.size == 432:  # 12*12*3 = 432
                        embedding = embedding.reshape(12, 12, 3)
                    else:
                        embedding = self._create_embedding_from_features(row)
            except Exception as e:
                print(f"Error loading embedding {embedding_path}: {e}")
                embedding = self._create_embedding_from_features(row)
        else:
            # Create embedding from features if file doesn't exist
            embedding = self._create_embedding_from_features(row)
        
        # Normalize embedding
        embedding = embedding.astype(np.float32)
        if embedding.std() > 1e-8:
            embedding = (embedding - embedding.mean()) / embedding.std()
        
        return torch.tensor(embedding, dtype=torch.float32)
    
    def _create_embedding_from_features(self, row):
        """Create 12x12x3 embedding from 144 features"""
        features = row[self.feature_columns].values.astype(np.float32)
        
        # Verify exactly 144 features
        if len(features) != 144:
            print(f"Warning: Expected 144 features, got {len(features)}")
            # Pad or truncate to exactly 144
            if len(features) < 144:
                padded = np.zeros(144, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            else:
                features = features[:144]
        
        # Split into the three regions
        cell_features = features[0:48]
        nucleus_features = features[48:96]
        cytoplasm_features = features[96:144]
        
        # Create 12x12x3 embedding
        embedding = np.zeros((12, 12, 3), dtype=np.float32)
        
        # Fill each channel with its 48 features
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

class MultiHeadRegionAttention(nn.Module):
    """Multi-head attention for region-attention embeddings"""
    def __init__(self, embed_dim=144, num_heads=3, dropout=0.1):
        super(MultiHeadRegionAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """Forward pass with proper tensor handling"""
        batch_size, height, width, channels = x.shape
        seq_len = height * width  # 144 spatial locations
        x_seq = x.view(batch_size, seq_len, channels)  # [batch, 144, 3]
        
        # Project to higher dimension for attention
        if channels != self.embed_dim:
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(channels, self.embed_dim).to(x.device)
            x_seq = self.input_projection(x_seq)
        
        # Apply layer normalization
        x_norm = self.layer_norm(x_seq)
        
        # Generate Q, K, V
        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final linear transformation
        output = self.fc_out(attended)
        
        # Residual connection
        output = output + x_seq
        
        # Project back to original channel dimension if needed
        if self.embed_dim != channels:
            if not hasattr(self, 'output_projection'):
                self.output_projection = nn.Linear(self.embed_dim, channels).to(x.device)
            output = self.output_projection(output)
        
        # Reshape back to spatial format
        output = output.view(batch_size, height, width, channels)
        
        return output

class SpatialChannelAttention(nn.Module):
    """Spatial and Channel Attention Module (SCAM)"""
    def __init__(self, in_channels, reduction=16):
        super(SpatialChannelAttention, self).__init__()
        
        self.in_channels = in_channels
        
        # Channel attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(1, in_channels // reduction)
        
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        """Forward pass with proper error handling"""
        batch_size, channels, height, width = x.shape
        
        # Channel attention
        avg_pool = self.global_avg_pool(x).view(batch_size, channels)
        max_pool = self.global_max_pool(x).view(batch_size, channels)
        
        avg_out = self.channel_mlp(avg_pool)
        max_out = self.channel_mlp(max_pool)
        
        channel_att = torch.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        x_channel = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))
        x_spatial = x_channel * spatial_att
        
        return x_spatial

class EnhancedXception(nn.Module):
    """Enhanced Xception with attention modules for region-attention embeddings"""
    def __init__(self, num_classes=21, input_shape=(3, 12, 12)):
        super(EnhancedXception, self).__init__()
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Multi-head region attention
        self.region_attention = MultiHeadRegionAttention(embed_dim=144, num_heads=3)
        
        # Entry flow - adapted for 12x12 input
        self.entry_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.entry_bn1 = nn.BatchNorm2d(32)
        self.entry_scam1 = SpatialChannelAttention(32)
        
        self.entry_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.entry_bn2 = nn.BatchNorm2d(64)
        self.entry_scam2 = SpatialChannelAttention(64)
        
        # Depthwise separable convolutions
        self.depthwise1 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise1 = nn.Conv2d(64, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.scam1 = SpatialChannelAttention(128)
        
        self.depthwise2 = nn.Conv2d(128, 128, 3, padding=1, groups=128)
        self.pointwise2 = nn.Conv2d(128, 256, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.scam2 = SpatialChannelAttention(256)
        
        self.depthwise3 = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.pointwise3 = nn.Conv2d(256, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.scam3 = SpatialChannelAttention(512)
        
        # Global pooling and feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 1024)  # Reduced size for fusion
        
    def forward(self, x):
        """Forward pass with proper tensor handling"""
        # Ensure input is in CHW format for conv layers
        if x.dim() == 4 and x.shape[-1] == 3:
            x_chw = x.permute(0, 3, 1, 2)  # [batch, 3, 12, 12]
            x_hwc = x  # Keep original for attention
        else:
            x_chw = x
            x_hwc = x.permute(0, 2, 3, 1)  # [batch, 12, 12, 3]
        
        # Apply region attention
        x_attended = self.region_attention(x_hwc)  # [batch, 12, 12, 3]
        
        # Convert back to CHW for conv layers
        x = x_attended.permute(0, 3, 1, 2)  # [batch, 3, 12, 12]
        
        # Entry flow
        x = F.relu(self.entry_bn1(self.entry_conv1(x)))
        x = self.entry_scam1(x)
        x = F.relu(self.entry_bn2(self.entry_conv2(x)))
        x = self.entry_scam2(x)
        
        # Depthwise separable convolutions with attention
        x = F.relu(self.bn1(self.pointwise1(self.depthwise1(x))))
        x = self.scam1(x)
        
        x = F.relu(self.bn2(self.pointwise2(self.depthwise2(x))))
        x = self.scam2(x)
        
        x = F.relu(self.bn3(self.pointwise3(self.depthwise3(x))))
        x = self.scam3(x)
        
        # Global pooling and feature extraction
        x = self.global_pool(x)  # [batch, 512, 1, 1]
        x = torch.flatten(x, 1)  # [batch, 512]
        x = self.dropout(x)
        x = self.fc(x)  # [batch, 1024]
        
        return x

class XGBoostWrapper:
    """
    XGBoost wrapper for traditional feature classification
    """
    def __init__(self, num_classes=21, **xgb_params):
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Default XGBoost parameters optimized for bone marrow classification
        self.default_params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Update with user parameters
        self.default_params.update(xgb_params)
        
        print(f"XGBoost configured for {num_classes} classes")
        
    def fit(self, X, y, eval_set=None, early_stopping_rounds=10):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create XGBoost model
        self.model = xgb.XGBClassifier(**self.default_params)
        
        # Prepare evaluation set if provided
        eval_set_scaled = None
        if eval_set is not None:
            X_val_scaled = self.scaler.transform(eval_set[0])
            eval_set_scaled = [(X_val_scaled, eval_set[1])]
        
        # Train model
        self.model.fit(
            X_scaled, y,
            eval_set=eval_set_scaled,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        self.is_fitted = True
        print("XGBoost training completed")
        
        return self
    
    def predict(self, X):
        """Predict classes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        
        # Ensure correct shape for both binary and multiclass
        if self.num_classes == 2 and probs.shape[1] == 1:
            # Binary case with single column output
            probs = np.column_stack([1 - probs.flatten(), probs.flatten()])
        
        return probs
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.feature_importances_
    
    def save(self, filepath):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'num_classes': self.num_classes,
            'is_fitted': self.is_fitted,
            'params': self.default_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load model and scaler"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.num_classes = model_data['num_classes']
        self.is_fitted = model_data['is_fitted']
        self.default_params = model_data['params']

class FusionModule(nn.Module):
    """
    Fusion module for combining Xception and XGBoost features
    """
    def __init__(self, xception_dim=1024, xgboost_dim=21, fusion_dim=512, num_classes=21):
        super(FusionModule, self).__init__()
        
        self.xception_dim = xception_dim
        self.xgboost_dim = xgboost_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # Feature alignment layers
        self.xception_align = nn.Sequential(
            nn.Linear(xception_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.xgboost_align = nn.Sequential(
            nn.Linear(xgboost_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Attention-based fusion
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, xception_features, xgboost_probs):
        """
        Fuse features from both models
        """
        # Align feature dimensions
        xception_aligned = self.xception_align(xception_features)  # [batch, fusion_dim]
        xgboost_aligned = self.xgboost_align(xgboost_probs)       # [batch, fusion_dim]
        
        # Concatenate for attention computation
        combined = torch.cat([xception_aligned, xgboost_aligned], dim=1)  # [batch, fusion_dim*2]
        
        # Compute attention weights
        attention_weights = self.attention(combined)  # [batch, 2]
        
        # Apply attention weights
        weighted_xception = xception_aligned * attention_weights[:, 0:1]
        weighted_xgboost = xgboost_aligned * attention_weights[:, 1:2]
        
        # Fused features
        fused_features = weighted_xception + weighted_xgboost  # [batch, fusion_dim]
        
        # Final predictions
        logits = self.classifier(fused_features)
        uncertainty = self.uncertainty_estimator(fused_features)
        
        return logits, uncertainty, attention_weights

class HybridXceptionXGBoost(nn.Module):
    """
    Complete hybrid model combining Enhanced Xception and XGBoost
    """
    def __init__(self, num_classes=21, xgb_params=None):
        super(HybridXceptionXGBoost, self).__init__()
        
        self.num_classes = num_classes
        print(f"Building Hybrid Xception + XGBoost Model for {num_classes} classes")
        
        # Xception pathway for embeddings
        self.xception_net = EnhancedXception(num_classes=num_classes)
        
        # XGBoost wrapper for traditional features
        if xgb_params is None:
            xgb_params = {}
        self.xgboost_wrapper = XGBoostWrapper(num_classes=num_classes, **xgb_params)
        
        # Fusion module
        self.fusion = FusionModule(
            xception_dim=1024,
            xgboost_dim=num_classes,
            fusion_dim=512,
            num_classes=num_classes
        )
        
        print("Hybrid Xception + XGBoost Model built successfully")
        
    def train_xgboost(self, train_features, train_labels, val_features=None, val_labels=None):
        """Train the XGBoost component"""
        eval_set = None
        if val_features is not None and val_labels is not None:
            eval_set = (val_features, val_labels)
        
        self.xgboost_wrapper.fit(train_features, train_labels, eval_set=eval_set)
        
    def forward(self, embedding=None, features=None, mode='both'):
        """
        Forward pass with flexible input modes
        """
        if mode == 'embedding_only' and embedding is not None:
            # Only Xception pathway
            xception_features = self.xception_net(embedding)
            # Simple classifier for Xception features
            logits = torch.mm(xception_features, 
                            torch.randn(xception_features.shape[1], self.num_classes).to(xception_features.device))
            uncertainty = torch.sigmoid(torch.sum(xception_features, dim=1, keepdim=True)) * 0.5
            return logits, uncertainty
            
        elif mode == 'features_only' and features is not None:
            # Only XGBoost pathway (requires CPU processing)
            features_np = features.detach().cpu().numpy()
            if not self.xgboost_wrapper.is_fitted:
                raise ValueError("XGBoost model must be trained before inference")
            
            xgboost_probs = self.xgboost_wrapper.predict_proba(features_np)
            xgboost_probs_tensor = torch.tensor(xgboost_probs, dtype=torch.float32).to(features.device)
            
            logits = torch.log(xgboost_probs_tensor + 1e-8)  # Convert probs to logits
            uncertainty = torch.sum(-xgboost_probs_tensor * torch.log(xgboost_probs_tensor + 1e-8), dim=1, keepdim=True) / np.log(self.num_classes)
            return logits, uncertainty
            
        elif mode == 'both' and embedding is not None and features is not None:
            # Both pathways with fusion
            # Get XGBoost predictions
            features_np = features.detach().cpu().numpy()
            if not self.xgboost_wrapper.is_fitted:
                raise ValueError("XGBoost model must be trained before inference")
            
            xgboost_probs = self.xgboost_wrapper.predict_proba(features_np)
            xgboost_probs_tensor = torch.tensor(xgboost_probs, dtype=torch.float32).to(features.device)
            
            # Fusion
            logits, uncertainty, attention_weights = self.fusion(xception_features, xgboost_probs_tensor)
            return logits, uncertainty, attention_weights
            
        else:
            raise ValueError(f"Invalid mode {mode} or missing required inputs")

class HybridLoss(nn.Module):
    """
    Combined loss function for the hybrid Xception + XGBoost model
    """
    def __init__(self, num_classes=21, class_weights=None, 
                 alpha_ce=1.0, alpha_uncertainty=0.1, alpha_consistency=0.1):
        super(HybridLoss, self).__init__()
        
        self.alpha_ce = alpha_ce
        self.alpha_uncertainty = alpha_uncertainty
        self.alpha_consistency = alpha_consistency
        
        # Primary classification loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Uncertainty regularization
        self.uncertainty_loss = nn.MSELoss()
        
    def forward(self, logits, uncertainty, targets, consistency_target=None):
        """
        Compute hybrid loss with proper weighting
        """
        # Classification loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Uncertainty regularization
        predicted = torch.argmax(logits, dim=1)
        correct_mask = (predicted == targets).float()
        
        # Target uncertainty: low (0.1) for correct, high (0.9) for incorrect
        target_uncertainty = 0.1 + 0.8 * (1 - correct_mask)
        uncertainty_reg = self.uncertainty_loss(uncertainty.squeeze(), target_uncertainty)
        
        # Total loss
        total_loss = (self.alpha_ce * ce_loss + 
                     self.alpha_uncertainty * uncertainty_reg)
        
        # Consistency loss (if provided)
        if consistency_target is not None:
            consistency_loss = F.mse_loss(torch.softmax(logits, dim=1), consistency_target)
            total_loss += self.alpha_consistency * consistency_loss
            return total_loss, ce_loss, uncertainty_reg, consistency_loss
        
        return total_loss, ce_loss, uncertainty_reg

def convert_numpy_types(obj):
    """Recursively convert numpy types to JSON serializable Python types"""
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

def calculate_class_weights(labels):
    """Calculate class weights for handling imbalanced dataset"""
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

def train_xgboost_component(model, train_loader, val_loader):
    """Train the XGBoost component separately"""
    print("Training XGBoost component...")
    
    # Collect all features and labels
    train_features_list = []
    train_labels_list = []
    val_features_list = []
    val_labels_list = []
    
    # Extract training features
    for batch in tqdm(train_loader, desc="Extracting train features"):
        if 'features' in batch:
            train_features_list.append(batch['features'].numpy())
            train_labels_list.append(batch['label'].numpy())
    
    # Extract validation features
    for batch in tqdm(val_loader, desc="Extracting val features"):
        if 'features' in batch:
            val_features_list.append(batch['features'].numpy())
            val_labels_list.append(batch['label'].numpy())
    
    # Combine features
    train_features = np.vstack(train_features_list)
    train_labels = np.hstack(train_labels_list)
    val_features = np.vstack(val_features_list) if val_features_list else None
    val_labels = np.hstack(val_labels_list) if val_labels_list else None
    
    print(f"Training XGBoost on {len(train_features)} samples")
    if val_features is not None:
        print(f"Validation set: {len(val_features)} samples")
    
    # Train XGBoost
    model.train_xgboost(train_features, train_labels, val_features, val_labels)
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, mode='both', epoch=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_uncertainty_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move data to device
            targets = batch['label'].to(device)
            
            # Prepare inputs based on mode
            inputs = {}
            if mode in ['both', 'embedding_only'] and 'embedding' in batch:
                inputs['embedding'] = batch['embedding'].to(device)
            if mode in ['both', 'features_only'] and 'features' in batch:
                inputs['features'] = batch['features'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if mode == 'both':
                logits, uncertainty, attention_weights = model(mode=mode, **inputs)
            else:
                logits, uncertainty = model(mode=mode, **inputs)
            
            # Calculate loss
            loss, ce_loss, uncertainty_loss = criterion(logits, uncertainty, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_uncertainty_loss += uncertainty_loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'CE': f'{ce_loss.item():.4f}',
                    'Unc': f'{uncertainty_loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
                
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
    
    return {
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'uncertainty_loss': total_uncertainty_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }

def validate_epoch(model, dataloader, criterion, device, mode='both', epoch=0):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_uncertainty_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch+1}")):
            try:
                # Move data to device
                targets = batch['label'].to(device)
                
                # Prepare inputs based on mode
                inputs = {}
                if mode in ['both', 'embedding_only'] and 'embedding' in batch:
                    inputs['embedding'] = batch['embedding'].to(device)
                if mode in ['both', 'features_only'] and 'features' in batch:
                    inputs['features'] = batch['features'].to(device)
                
                # Forward pass
                if mode == 'both':
                    logits, uncertainty, attention_weights = model(mode=mode, **inputs)
                    all_attention_weights.extend(attention_weights.cpu().numpy())
                else:
                    logits, uncertainty = model(mode=mode, **inputs)
                
                # Calculate loss
                loss, ce_loss, uncertainty_loss = criterion(logits, uncertainty, targets)
                
                # Statistics
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_uncertainty_loss += uncertainty_loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store predictions for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy().flatten())
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    result = {
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'uncertainty_loss': total_uncertainty_loss / len(dataloader),
        'accuracy': 100. * correct / total,
        'predictions': all_predictions,
        'targets': all_targets,
        'uncertainties': all_uncertainties
    }
    
    if all_attention_weights:
        result['attention_weights'] = all_attention_weights
    
    return result

def create_data_loaders(csv_file, image_dir, embeddings_dir, mode, batch_size, test_size=0.4):
    """Create train and validation data loaders"""
    print("Creating data loaders...")
    
    # Load full dataset to split
    full_df = pd.read_csv(csv_file)
    
    # Check required columns
    if 'image_name' not in full_df.columns or 'label' not in full_df.columns:
        raise ValueError("CSV file must contain 'image_name' and 'label' columns")
    
    print(f"Loaded {len(full_df)} samples with {full_df['label'].nunique()} unique classes")
    print(f"Label distribution:\n{full_df['label'].value_counts()}")
    
    # Stratified split using original labels
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
    train_df.to_csv('train_split_xgb.csv', index=False)
    test_df.to_csv('test_split_xgb.csv', index=False)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = BoneMarrowDataset('train_split_xgb.csv', image_dir, embeddings_dir, 
                                    None, mode)
    test_dataset = BoneMarrowDataset('test_split_xgb.csv', image_dir, embeddings_dir, 
                                   None, mode)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=torch.cuda.is_available())
    
    return train_loader, test_loader, train_dataset.label_encoder

def evaluate_model(predictions, targets, uncertainties, label_encoder, attention_weights=None, save_path=None):
    """Comprehensive model evaluation with uncertainty and attention analysis"""
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(targets, predictions, 
                                 target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Uncertainty statistics
    mean_uncertainty = np.mean(uncertainties)
    correct_mask = np.array(predictions) == np.array(targets)
    correct_uncertainty = np.mean(np.array(uncertainties)[correct_mask])
    incorrect_uncertainty = np.mean(np.array(uncertainties)[~correct_mask])
    
    # Attention statistics (if available)
    attention_stats = {}
    if attention_weights is not None:
        attention_weights = np.array(attention_weights)
        attention_stats = {
            'mean_xception_weight': np.mean(attention_weights[:, 0]),
            'mean_xgboost_weight': np.mean(attention_weights[:, 1]),
            'std_xception_weight': np.std(attention_weights[:, 0]),
            'std_xgboost_weight': np.std(attention_weights[:, 1])
        }
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    if save_path:
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Confusion matrix
        if SNS_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
        else:
            im = axes[0,0].imshow(cm, interpolation='nearest', cmap='Blues')
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[0,0].text(j, i, str(cm[i, j]), ha='center', va='center')
        
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # Uncertainty distribution
        axes[0,1].hist(np.array(uncertainties)[correct_mask], alpha=0.7, label='Correct', bins=30)
        axes[0,1].hist(np.array(uncertainties)[~correct_mask], alpha=0.7, label='Incorrect', bins=30)
        axes[0,1].set_title('Uncertainty Distribution')
        axes[0,1].set_xlabel('Uncertainty')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend()
        
        # Per-class F1 scores
        f1_scores = [report[cls]['f1-score'] for cls in class_names]
        axes[1,0].bar(range(len(class_names)), f1_scores, alpha=0.7)
        axes[1,0].set_title('Per-Class F1 Scores')
        axes[1,0].set_xlabel('Classes')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].set_xticks(range(len(class_names)))
        axes[1,0].set_xticklabels(class_names, rotation=45)
        
        # Attention weights (if available)
        if attention_weights is not None:
            attention_weights = np.array(attention_weights)
            axes[1,1].boxplot([attention_weights[:, 0], attention_weights[:, 1]], 
                             labels=['Xception', 'XGBoost'])
            axes[1,1].set_title('Attention Weight Distribution')
            axes[1,1].set_ylabel('Attention Weight')
        else:
            axes[1,1].text(0.5, 0.5, 'No Attention Weights\nAvailable', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Attention Analysis')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    result = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm,
        'mean_uncertainty': mean_uncertainty,
        'correct_uncertainty': correct_uncertainty,
        'incorrect_uncertainty': incorrect_uncertainty
    }
    
    result.update(attention_stats)
    return result

def plot_feature_importance(model, feature_names, save_path):
    """Plot XGBoost feature importance"""
    if not model.xgboost_wrapper.is_fitted:
        print("WARNING: XGBoost model not fitted, skipping feature importance plot")
        return
    
    importance = model.xgboost_wrapper.get_feature_importance()
    
    # Get top 20 features
    top_indices = np.argsort(importance)[-20:]
    top_importance = importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_importance)), top_importance)
    plt.yticks(range(len(top_importance)), top_names)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function for hybrid Xception + XGBoost model"""
    parser = argparse.ArgumentParser(description='Hybrid Xception + XGBoost Bone Marrow Classification')
    parser.add_argument('--csv_file', required=True, help='Features CSV file')
    parser.add_argument('--image_dir', required=True, help='Images directory')
    parser.add_argument('--embeddings_dir', required=True, help='Embeddings directory')
    parser.add_argument('--save_dir', default='results_xgb_hybrid', help='Save directory')
    parser.add_argument('--mode', default='both', choices=['both', 'embedding_only', 'features_only'])
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # XGBoost specific parameters
    parser.add_argument('--xgb_max_depth', type=int, default=6, help='XGBoost max depth')
    parser.add_argument('--xgb_learning_rate', type=float, default=0.1, help='XGBoost learning rate')
    parser.add_argument('--xgb_n_estimators', type=int, default=200, help='XGBoost n_estimators')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optional TensorBoard setup
    writer = None
    if TENSORBOARD_AVAILABLE:
        try:
            writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard'))
            print("TensorBoard logging enabled")
        except Exception as e:
            print(f"WARNING: TensorBoard setup failed: {e}")
            writer = None
    
    # Create data loaders
    train_loader, test_loader, label_encoder = create_data_loaders(
        args.csv_file, args.image_dir, args.embeddings_dir, 
        args.mode, args.batch_size
    )
    
    num_classes = len(label_encoder.classes_)
    print(f"Training on {num_classes} classes")
    
    # XGBoost parameters
    xgb_params = {
        'max_depth': args.xgb_max_depth,
        'learning_rate': args.xgb_learning_rate,
        'n_estimators': args.xgb_n_estimators,
    }
    
    # Create model
    model = HybridXceptionXGBoost(num_classes=num_classes, xgb_params=xgb_params).to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Neural model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train XGBoost component first (if using features)
    if args.mode in ['both', 'features_only']:
        model = train_xgboost_component(model, train_loader, test_loader)
        
        # Save XGBoost model
        model.xgboost_wrapper.save(os.path.join(args.save_dir, 'xgboost_model.pkl'))
        print("XGBoost model saved")
    
    # Calculate class weights for imbalanced dataset
    train_df = pd.read_csv('train_split_xgb.csv')
    train_label_encoder = LabelEncoder()
    train_encoded_labels = train_label_encoder.fit_transform(train_df['label'])
    class_weights_dict = calculate_class_weights(train_encoded_labels)
    class_weights = [class_weights_dict[i] for i in range(num_classes)]
    
    # Loss and optimizer (only for neural components)
    criterion = HybridLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        alpha_ce=1.0,
        alpha_uncertainty=0.1,
        alpha_consistency=0.1
    ).to(device)
    
    # Only optimize neural network parameters
    neural_params = list(model.xception_net.parameters()) + list(model.fusion.parameters())
    optimizer = optim.AdamW(neural_params, lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs")
    
    best_f1 = 0.0
    early_stop_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_ce_loss': [], 'train_uncertainty_loss': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_ce_loss': [], 'val_uncertainty_loss': []
    }
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, args.mode, epoch)
        
        # Validation
        val_metrics = validate_epoch(model, test_loader, criterion, device, args.mode, epoch)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != current_lr:
            print(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
        
        # Evaluate detailed metrics
        eval_metrics = evaluate_model(
            val_metrics['predictions'], 
            val_metrics['targets'],
            val_metrics['uncertainties'],
            label_encoder,
            val_metrics.get('attention_weights'),
            save_path=os.path.join(args.save_dir, f'evaluation_epoch_{epoch+1}.png')
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_ce_loss'].append(train_metrics['ce_loss'])
        history['train_uncertainty_loss'].append(train_metrics['uncertainty_loss'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(eval_metrics['f1_weighted'])
        history['val_ce_loss'].append(val_metrics['ce_loss'])
        history['val_uncertainty_loss'].append(val_metrics['uncertainty_loss'])
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            writer.add_scalar('F1_Score/Validation', eval_metrics['f1_weighted'], epoch)
            writer.add_scalar('Uncertainty/Mean', eval_metrics['mean_uncertainty'], epoch)
            
            # Log attention weights if available
            if 'mean_xception_weight' in eval_metrics:
                writer.add_scalar('Attention/Xception_Weight', eval_metrics['mean_xception_weight'], epoch)
                writer.add_scalar('Attention/XGBoost_Weight', eval_metrics['mean_xgboost_weight'], epoch)
        
        # Print metrics
        print(f"Train: Loss={train_metrics['loss']:.4f}, CE={train_metrics['ce_loss']:.4f}, "
              f"Unc={train_metrics['uncertainty_loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
        print(f"Val: Loss={val_metrics['loss']:.4f}, CE={val_metrics['ce_loss']:.4f}, "
              f"Unc={val_metrics['uncertainty_loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
        print(f"F1: {eval_metrics['f1_weighted']:.4f}, Uncertainty: {eval_metrics['mean_uncertainty']:.4f}")
        
        if 'mean_xception_weight' in eval_metrics:
            print(f"Attention - Xception: {eval_metrics['mean_xception_weight']:.3f}, "
                  f"XGBoost: {eval_metrics['mean_xgboost_weight']:.3f}")
        
        # Save best model
        if eval_metrics['f1_weighted'] > best_f1:
            best_f1 = eval_metrics['f1_weighted']
            early_stop_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'label_encoder': label_encoder,
                'args': args,
                'eval_metrics': eval_metrics
            }, os.path.join(args.save_dir, 'best_hybrid_model.pth'))
            
            # Save detailed metrics
            with open(os.path.join(args.save_dir, 'best_metrics.json'), 'w') as f:
                serializable_metrics = convert_numpy_types(eval_metrics)
                json.dump(serializable_metrics, f, indent=2)
            
            print(f"New best F1 score: {best_f1:.4f}")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement")
            break
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    # Save final models and history
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_hybrid_model.pth'))
    
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot feature importance for XGBoost
    if args.mode in ['both', 'features_only']:
        feature_columns = [col for col in pd.read_csv(args.csv_file).columns 
                          if col not in ['image_name', 'label']]
        plot_feature_importance(model, feature_columns, 
                              os.path.join(args.save_dir, 'xgboost_feature_importance.png'))
    
    # Create comprehensive training summary plot
    plt.figure(figsize=(24, 12))
    
    # Loss plots
    plt.subplot(3, 4, 1)
    plt.plot(history['train_loss'], label='Train Total', linewidth=2)
    plt.plot(history['val_loss'], label='Val Total', linewidth=2)
    plt.title('Total Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 2)
    plt.plot(history['train_ce_loss'], label='Train CE', linewidth=2)
    plt.plot(history['val_ce_loss'], label='Val CE', linewidth=2)
    plt.title('Classification Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 3)
    plt.plot(history['train_uncertainty_loss'], label='Train Unc', linewidth=2)
    plt.plot(history['val_uncertainty_loss'], label='Val Unc', linewidth=2)
    plt.title('Uncertainty Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 4)
    plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
    plt.title('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 5)
    plt.plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
    plt.title('F1 Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Per-class F1 scores
    plt.subplot(3, 4, 6)
    final_report = eval_metrics['classification_report']
    classes = label_encoder.classes_
    f1_scores = [final_report[cls]['f1-score'] for cls in classes]
    
    plt.bar(range(len(classes)), f1_scores, alpha=0.7)
    plt.title('Per-Class F1 Scores', fontsize=12)
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.xticks(range(len(classes)), classes, rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Learning curves comparison
    plt.subplot(3, 4, 7)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.title('Learning Curves', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uncertainty analysis
    plt.subplot(3, 4, 8)
    plt.text(0.1, 0.9, f"Mean Uncertainty: {eval_metrics['mean_uncertainty']:.4f}", fontsize=10)
    plt.text(0.1, 0.8, f"Correct Uncertainty: {eval_metrics['correct_uncertainty']:.4f}", fontsize=10)
    plt.text(0.1, 0.7, f"Incorrect Uncertainty: {eval_metrics['incorrect_uncertainty']:.4f}", fontsize=10)
    if 'mean_xception_weight' in eval_metrics:
        plt.text(0.1, 0.6, f"Avg Xception Weight: {eval_metrics['mean_xception_weight']:.3f}", fontsize=10)
        plt.text(0.1, 0.5, f"Avg XGBoost Weight: {eval_metrics['mean_xgboost_weight']:.3f}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Uncertainty & Attention Stats', fontsize=12)
    
    # Model architecture info
    plt.subplot(3, 4, 9)
    plt.text(0.1, 0.9, f"Hybrid Xception + XGBoost", fontsize=10, weight='bold')
    plt.text(0.1, 0.8, f"Classes: {num_classes}", fontsize=10)
    plt.text(0.1, 0.7, f"Mode: {args.mode}", fontsize=10)
    plt.text(0.1, 0.6, f"Neural Params: {total_params:,}", fontsize=10)
    plt.text(0.1, 0.5, f"XGB Max Depth: {args.xgb_max_depth}", fontsize=10)
    plt.text(0.1, 0.4, f"XGB N Estimators: {args.xgb_n_estimators}", fontsize=10)
    plt.text(0.1, 0.3, f"XGB Learning Rate: {args.xgb_learning_rate}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Model Architecture', fontsize=12)
    
    # Performance summary
    plt.subplot(3, 4, 10)
    plt.text(0.1, 0.9, f"Best F1: {best_f1:.4f}", fontsize=12, weight='bold', color='green')
    plt.text(0.1, 0.8, f"Final Accuracy: {eval_metrics['accuracy']:.4f}", fontsize=11)
    plt.text(0.1, 0.7, f"Macro F1: {eval_metrics['f1_macro']:.4f}", fontsize=11)
    plt.text(0.1, 0.6, f"Weighted F1: {eval_metrics['f1_weighted']:.4f}", fontsize=11)
    plt.text(0.1, 0.5, f"Epochs Trained: {epoch+1}/{args.num_epochs}", fontsize=10)
    plt.text(0.1, 0.4, f"Batch Size: {args.batch_size}", fontsize=10)
    plt.text(0.1, 0.3, f"Learning Rate: {args.learning_rate}", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Performance Summary', fontsize=12)
    
    # Training efficiency
    plt.subplot(3, 4, 11)
    epochs_range = range(1, len(history['train_loss']) + 1)
    train_acc_curve = history['train_acc']
    val_acc_curve = history['val_acc']
    
    plt.plot(epochs_range, train_acc_curve, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc_curve, 'r-', label='Validation Accuracy', linewidth=2)
    plt.fill_between(epochs_range, train_acc_curve, val_acc_curve, alpha=0.3)
    plt.title('Training Efficiency', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final metrics comparison
    plt.subplot(3, 4, 12)
    metrics_names = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    metrics_values = [eval_metrics['accuracy'], eval_metrics['f1_macro'], eval_metrics['f1_weighted']]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    plt.title('Final Metrics Comparison', fontsize=12)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'comprehensive_training_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a detailed model comparison plot
    if args.mode == 'both':
        plt.figure(figsize=(12, 8))
        
        # Attention weights distribution over time (if available)
        if 'attention_weights' in val_metrics:
            attention_weights = np.array(val_metrics['attention_weights'])
            
            plt.subplot(2, 2, 1)
            plt.hist(attention_weights[:, 0], alpha=0.7, label='Xception Weights', bins=30)
            plt.hist(attention_weights[:, 1], alpha=0.7, label='XGBoost Weights', bins=30)
            plt.title('Attention Weight Distribution')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.scatter(attention_weights[:, 0], attention_weights[:, 1], alpha=0.6)
            plt.xlabel('Xception Weight')
            plt.ylabel('XGBoost Weight')
            plt.title('Attention Weight Correlation')
            plt.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Perfect Anti-correlation')
            plt.legend()
            
            # Attention weights vs accuracy
            plt.subplot(2, 2, 3)
            correct_predictions = np.array(val_metrics['predictions']) == np.array(val_metrics['targets'])
            
            correct_xception_weights = attention_weights[correct_predictions, 0]
            incorrect_xception_weights = attention_weights[~correct_predictions, 0]
            
            plt.boxplot([correct_xception_weights, incorrect_xception_weights], 
                       labels=['Correct', 'Incorrect'])
            plt.title('Xception Attention vs Prediction Accuracy')
            plt.ylabel('Xception Weight')
            
            plt.subplot(2, 2, 4)
            correct_xgboost_weights = attention_weights[correct_predictions, 1]
            incorrect_xgboost_weights = attention_weights[~correct_predictions, 1]
            
            plt.boxplot([correct_xgboost_weights, incorrect_xgboost_weights], 
                       labels=['Correct', 'Incorrect'])
            plt.title('XGBoost Attention vs Prediction Accuracy')
            plt.ylabel('XGBoost Weight')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'attention_analysis.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nTraining completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Final Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Mean Uncertainty: {eval_metrics['mean_uncertainty']:.4f}")
    
    if 'mean_xception_weight' in eval_metrics:
        print(f"Average Attention - Xception: {eval_metrics['mean_xception_weight']:.3f}, "
              f"XGBoost: {eval_metrics['mean_xgboost_weight']:.3f}")
    
    print(f"Results saved to: {args.save_dir}")
    
    if TENSORBOARD_AVAILABLE:
        print(f"View TensorBoard: tensorboard --logdir {args.save_dir}/tensorboard")
    
    # Print final summary
    print("\n" + "="*60)
    print("HYBRID XCEPTION + XGBOOST TRAINING COMPLETE")
    print("="*60)
    print(f"Best Performance:")
    print(f"   • F1 Score (Weighted): {eval_metrics['f1_weighted']:.4f}")
    print(f"   • F1 Score (Macro): {eval_metrics['f1_macro']:.4f}")
    print(f"   • Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Model Complexity:")
    print(f"   • Neural Network Parameters: {total_params:,}")
    print(f"   • XGBoost Trees: {args.xgb_n_estimators}")
    print(f"Uncertainty Insights:")
    print(f"   • Avg Uncertainty (Correct): {eval_metrics['correct_uncertainty']:.4f}")
    print(f"   • Avg Uncertainty (Incorrect): {eval_metrics['incorrect_uncertainty']:.4f}")
    
    if args.mode == 'both' and 'mean_xception_weight' in eval_metrics:
        print(f"Fusion Balance:")
        print(f"   • Xception Contribution: {eval_metrics['mean_xception_weight']:.1%}")
        print(f"   • XGBoost Contribution: {eval_metrics['mean_xgboost_weight']:.1%}")
    
    print("="*60)

if __name__ == "__main__":
    main() 
    