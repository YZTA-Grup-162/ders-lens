
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)
class DAiSEEAttentionModelFixed(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Feature extractor (CNN backbone)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Shared features
        self.shared_features = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Task-specific heads
        self.engagement_head = nn.Linear(128, 1)
        self.boredom_head = nn.Linear(128, 1)
        self.confusion_head = nn.Linear(128, 1)
        self.frustration_head = nn.Linear(128, 1)
        self.happiness_head = nn.Linear(128, 1)
        self.concentration_head = nn.Linear(128, 1)
        self.stress_head = nn.Linear(128, 1)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features.unsqueeze(1))
        attended_features = (features.unsqueeze(1) * attention_weights).sum(dim=1)
        
        # Shared features
        shared = self.shared_features(attended_features)
        
        # Predictions
        engagement = self.engagement_head(shared)
        boredom = self.boredom_head(shared)
        confusion = self.confusion_head(shared)
        frustration = self.frustration_head(shared)
        happiness = self.happiness_head(shared)
        concentration = self.concentration_head(shared)
        stress = self.stress_head(shared)
        
        # Return attention score (engagement) as the main output
        return engagement
class FER2013EmotionModelFixed(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = self._create_mobilevit_backbone()
        self.emotion_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(640, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=640,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(640, 640),
            nn.ReLU(inplace=True)
        )
    def _create_mobilevit_backbone(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            self._make_mobilevit_block(32, 64, 2),
            self._make_mobilevit_block(64, 128, 3),
            self._make_mobilevit_block(128, 256, 4),
            self._make_mobilevit_block(256, 320, 5),
            self._make_mobilevit_block(320, 640, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    def _make_mobilevit_block(self, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else out_channels, 
                             out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        return nn.Sequential(*layers)
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 2:
            features = features.unsqueeze(1)
            attended_features, _ = self.attention(features, features, features)
            features = attended_features.squeeze(1)
        features = self.feature_fusion(features)
        emotion_output = self.emotion_head(features)
        return emotion_output
class MendeleyAttentionModelFixed(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Input size: [batch, 28] (flattened features)
        self.network = nn.Sequential(
            nn.Linear(28, 512),  # First layer to match input features
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(28, 14),  # Process input features
            nn.ReLU(inplace=True),
            nn.Linear(14, 28),  # Back to original feature size
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # If input is 4D tensor (batch, channels, height, width), flatten it
        if x.dim() == 4:
            x = x.view(x.size(0), -1)  # Flatten spatial dimensions
            
        # Ensure input has 28 features
        if x.size(1) != 28:
            raise ValueError(f"Expected input with 28 features, got {x.size(1)}")
            
        # Apply attention
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        # Process through network
        features = self.network(attended_features)
        output = self.classifier(features)
        
        return output
def load_model_with_fallback(model_path, model_class, num_classes=2):
    """Load a PyTorch model with fallback mechanisms for different model formats.
    
    Args:
        model_path: Path to the model file
        model_class: The model class to instantiate
        num_classes: Number of output classes
        
    Returns:
        Loaded model or None if loading failed
    """
    try:
        logger.info(f"🔍 Attempting to load model from: {model_path}")
        
        # First check if file exists
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None
            
        # Try to load the model
        try:
            # Try loading as a PyTorch model first
            model = model_class(num_classes=num_classes)
            
            # Try different ways to load the state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Try strict loading first
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info(f"✅ Model loaded successfully (strict): {model_path}")
                return model
                
            except RuntimeError as e:
                logger.warning(f"Strict loading failed: {e}")
                
                # Try non-strict loading
                try:
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"✅ Model loaded with non-strict matching: {model_path}")
                    return model
                except Exception as e2:
                    logger.error(f"Non-strict loading failed: {e2}")
                    
                    # Try to create a new model with the correct architecture
                    try:
                        # If we have a custom loading mechanism in the model class, use it
                        if hasattr(model_class, 'from_pretrained'):
                            model = model_class.from_pretrained(model_path)
                            logger.info("✅ Model loaded using custom from_pretrained method")
                            return model
                    except Exception as e3:
                        logger.error(f"Custom loading also failed: {e3}")
                    
                    return None
                    
        except Exception as e:
            logger.error(f"Error loading model (model architecture): {e}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in load_model_with_fallback (model architecture): {e}")
        return None
def create_simple_fallback_model(input_features=10, num_classes=2):
    return nn.Sequential(
        nn.Linear(input_features, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, num_classes),
        nn.Softmax(dim=1)
    )
def get_model_info(model):
    if model is None:
        return "Model is None"
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': type(model).__name__
    }