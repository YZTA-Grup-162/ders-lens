
import math
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(context)
class TemporalCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            padding = (kernel_size - 1) // 2
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)
class MobileViTBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 1000):
        super().__init__()
        self.backbone = timm.create_model('mobilevitv2_050', pretrained=pretrained)
        self.feature_dim = self.backbone.num_features
        self.backbone.head = nn.Identity()
        self._freeze_early_layers()
    def _freeze_early_layers(self, freeze_layers: int = 3):
        layer_count = 0
        for name, param in self.backbone.named_parameters():
            if layer_count < freeze_layers:
                param.requires_grad = False
            layer_count += 1
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained)
        self.feature_dim = self.backbone.num_features
        self.backbone.classifier = nn.Identity()
        self._freeze_early_layers()
    def _freeze_early_layers(self, freeze_layers: int = 50):
        params = list(self.backbone.parameters())
        for i in range(min(freeze_layers, len(params))):
            params[i].requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
class FeatureFusionModule(nn.Module):
    def __init__(self, visual_dim: int, handcrafted_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.handcrafted_projection = nn.Linear(handcrafted_dim, hidden_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=1)
           )
    def forward(self, visual_features: torch.Tensor, handcrafted_features: torch.Tensor) -> torch.Tensor:
        visual_proj = self.visual_projection(visual_features)
        handcrafted_proj = self.handcrafted_projection(handcrafted_features)
        combined = torch.cat([visual_proj, handcrafted_proj], dim=1)
        attention_weights = self.attention_weights(combined)
        weighted_visual = visual_proj * attention_weights[:, 0:1]
        weighted_handcrafted = handcrafted_proj * attention_weights[:, 1:2]
        fused = torch.cat([weighted_visual, weighted_handcrafted], dim=1)
        return self.fusion_layer(fused)
class AttentionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
class EngagementHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 4)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
class EmotionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 4)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
class DersLensModel(nn.Module):
    """Complete DersLens model with temporal fusion
    Architecture:
    1. Vision backbone (MobileViT XS or EfficientNet-B3)
    2. Temporal module (Causal CNN or Bi-LSTM)
    3. Feature fusion (Visual + Handcrafted)
    4. Multi-task heads (Attention, Engagement, Emotion)"""
    def __init__(
        self,
        backbone_type: str = "mobilevit",
        sequence_length: int = 32,
        handcrafted_dim: int = 20,
        hidden_dim: int = 256,
        temporal_type: str = "cnn",
        pretrained: bool = True
    ):
        super().__init__()
        if backbone_type == "mobilevit":
            self.backbone = MobileViTBackbone(pretrained=pretrained)
            visual_dim = self.backbone.feature_dim
        elif backbone_type == "efficientnet":
            self.backbone = EfficientNetBackbone(pretrained=pretrained)
            visual_dim = self.backbone.feature_dim
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        if temporal_type == "cnn":
            self.temporal_module = TemporalCNN(visual_dim, hidden_dim)
        elif temporal_type == "lstm":
            self.temporal_module = nn.LSTM(
                visual_dim, hidden_dim, batch_first=True, bidirectional=True
            )
            hidden_dim *= 2
        elif temporal_type == "attention":
            self.temporal_module = SelfAttention(visual_dim, num_heads=8)
            hidden_dim = visual_dim
        else:
            raise ValueError(f"Unknown temporal type: {temporal_type}")
        self.temporal_type = temporal_type
        self.fusion_module = FeatureFusionModule(hidden_dim, handcrafted_dim)
        fusion_output_dim = 256
        self.attention_head = AttentionHead(fusion_output_dim)
        self.engagement_head = EngagementHead(fusion_output_dim)
        self.emotion_head = EmotionHead(fusion_output_dim)
        self._initialize_weights()
    def _initialize_weights(self):
        for module in [self.fusion_module, self.attention_head, self.engagement_head, self.emotion_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    def forward(
        self, 
        frames: torch.Tensor,
        handcrafted_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, C, H, W = frames.shape
        frames = frames.view(batch_size * seq_len, C, H, W)
        visual_features = self.backbone(frames)
        visual_features = visual_features.view(batch_size, seq_len, -1)
        if self.temporal_type == "cnn":
            temporal_features = self.temporal_module(visual_features)
        elif self.temporal_type == "lstm":
            temporal_features, _ = self.temporal_module(visual_features)
        elif self.temporal_type == "attention":
            temporal_features = self.temporal_module(visual_features, mask)
        if self.temporal_type == "attention":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(temporal_features)
                temporal_features = temporal_features * mask_expanded
                temporal_features = temporal_features.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                temporal_features = temporal_features.mean(dim=1)
        else:
            temporal_features = temporal_features[:, -1, :]
        handcrafted_last = handcrafted_features[:, -1, :]
        fused_features = self.fusion_module(temporal_features, handcrafted_last)
        attention_score = self.attention_head(fused_features)
        engagement_logits = self.engagement_head(fused_features)
        emotion_logits = self.emotion_head(fused_features)
        return {
            "attention_score": attention_score.squeeze(-1),
            "engagement_logits": engagement_logits,
            "emotion_logits": emotion_logits,
            "features": fused_features
       }
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        attention_weight: float = 1.0,
        engagement_weight: float = 1.0,
        emotion_weight: float = 1.0,
        use_focal_loss: bool = True
   ):
        super().__init__()
        self.attention_weight = attention_weight
        self.engagement_weight = engagement_weight
        self.emotion_weight = emotion_weight
        self.attention_loss = nn.BCELoss()
        if use_focal_loss:
            self.engagement_loss = FocalLoss(alpha=0.25, gamma=2.0)
            self.emotion_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.engagement_loss = nn.CrossEntropyLoss()
            self.emotion_loss = nn.CrossEntropyLoss()
    def forward(
        self,   
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        attention_loss = self.attention_loss(
            predictions["attention_score"], 
            targets["attention"].float()
       )
        engagement_loss = self.engagement_loss(
            predictions["engagement_logits"], 
            targets["engagement"]
       )
        emotion_loss = self.emotion_loss(
            predictions["emotion_logits"], 
            targets["emotion"]
       )
        total_loss = (
            self.attention_weight * attention_loss +
            self.engagement_weight * engagement_loss +
            self.emotion_weight * emotion_loss
       )
        return {
            "total_loss": total_loss,
            "attention_loss": attention_loss,
            "engagement_loss": engagement_loss,
            "emotion_loss": emotion_loss
       }
def create_model(config: Dict) -> DersLensModel:
    return DersLensModel(
        backbone_type=config.get("backbone_type", "mobilevit"),
        sequence_length=config.get("sequence_length", 32),
        handcrafted_dim=config.get("handcrafted_dim", 20),
        hidden_dim=config.get("hidden_dim", 256),
        temporal_type=config.get("temporal_type", "cnn"),
        pretrained=config.get("pretrained", True)
    )
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb