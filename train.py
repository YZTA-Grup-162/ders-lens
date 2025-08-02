
import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnx
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from app.ai.datasets import AttentionDataset, DatasetManager
from app.ai.inference import ONNXModelOptimizer
from app.ai.models import (DersLensModel, MultiTaskLoss, count_parameters,
                           create_model, get_model_size_mb)
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
class DersLensTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        self.writer = None
        self.output_dir = Path(config.get('output_dir', './models_trained'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    def setup_model(self):
        self.model = create_model(self.config['model'])
        self.model.to(self.device)
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        logger.info(f"Model size: {get_model_size_mb(self.model):.2f} MB")
        optimizer_config = self.config.get('optimizer', {})
        if optimizer_config.get('type', 'adamw').lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 1e-5),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 1e-5)
            )
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('type', 'cosine').lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_config.get('type').lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        loss_config = self.config.get('loss', {})
        self.criterion = MultiTaskLoss(
            attention_weight=loss_config.get('attention_weight', 1.0),
            engagement_weight=loss_config.get('engagement_weight', 1.0),
            emotion_weight=loss_config.get('emotion_weight', 1.0),
            use_focal_loss=loss_config.get('use_focal_loss', True)
        )
        early_stopping_config = self.config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 15),
            min_delta=early_stopping_config.get('min_delta', 1e-4)
        )
        self.writer = SummaryWriter(
            log_dir=self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    async def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        dataset_manager = DatasetManager()
        logger.info("Loading datasets...")
        try:
            daisee_df, daisee_preprocessor = await dataset_manager.prepare_daisee()
            logger.info(f"DAiSEE dataset loaded: {len(daisee_df)} samples")
        except Exception as e:
            logger.error(f"Failed to load DAiSEE dataset: {e}")
            daisee_df, daisee_preprocessor = None, None
        try:
            mendeley_df = await dataset_manager.prepare_mendeley()
            logger.info(f"Mendeley dataset loaded: {len(mendeley_df)} samples")
        except Exception as e:
            logger.error(f"Failed to load Mendeley dataset: {e}")
            mendeley_df = None
        if daisee_df is not None and mendeley_df is not None:
            logger.info("Using combined DAiSEE and Mendeley datasets")
            train_df, val_df = dataset_manager.create_train_val_split(
                daisee_df, test_size=0.2, stratify_column='attention_binary'
            )
        elif daisee_df is not None:
            logger.info("Using DAiSEE dataset only")
            train_df, val_df = dataset_manager.create_train_val_split(
                daisee_df, test_size=0.2, stratify_column='attention_binary'
            )
        elif mendeley_df is not None:
            logger.info("Using Mendeley dataset only")
            train_df, val_df = dataset_manager.create_train_val_split(
                mendeley_df, test_size=0.2, stratify_column='attention_binary'
            )
        else:
            raise RuntimeError("No datasets available for training")
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_visual_data = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(len(train_df))]
        val_visual_data = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(len(val_df))]
        train_handcrafted = np.random.randn(len(train_df), 20).astype(np.float32)
        val_handcrafted = np.random.randn(len(val_df), 20).astype(np.float32)
        train_labels = {
            'attention': train_df['attention_binary'].values,
            'engagement': np.random.randint(0, 4, len(train_df)),
            'emotion': np.random.randint(0, 4, len(train_df))
        }
        val_labels = {
            'attention': val_df['attention_binary'].values,
            'engagement': np.random.randint(0, 4, len(val_df)),
            'emotion': np.random.randint(0, 4, len(val_df))
        }
        train_dataset = AttentionDataset(
            train_visual_data, train_handcrafted, train_labels,
            sequence_length=self.config['model']['sequence_length'],
            transform=train_transform
        )
        val_dataset = AttentionDataset(
            val_visual_data, val_handcrafted, val_labels,
            sequence_length=self.config['model']['sequence_length'],
            transform=val_transform
        )
        train_loader, val_loader = dataset_manager.get_data_loaders(
            train_dataset, val_dataset,
            batch_size=self.config.get('batch_size', 8),
            num_workers=self.config.get('num_workers', 4)
        )
        return train_loader, val_loader
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        attention_loss_sum = 0.0
        engagement_loss_sum = 0.0
        emotion_loss_sum = 0.0
        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(self.device)
            handcrafted = batch['handcrafted_features'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            self.optimizer.zero_grad()
            predictions = self.model(frames, handcrafted)
            losses = self.criterion(predictions, labels)
            loss = losses['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            attention_loss_sum += losses['attention_loss'].item()
            engagement_loss_sum += losses['engagement_loss'].item()
            emotion_loss_sum += losses['emotion_loss'].item()
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}"
                )
        avg_loss = total_loss / num_batches
        avg_attention_loss = attention_loss_sum / num_batches
        avg_engagement_loss = engagement_loss_sum / num_batches
        avg_emotion_loss = emotion_loss_sum / num_batches
        return {
            'total_loss': avg_loss,
            'attention_loss': avg_attention_loss,
            'engagement_loss': avg_engagement_loss,
            'emotion_loss': avg_emotion_loss
        }
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        attention_preds = []
        engagement_preds = []
        emotion_preds = []
        attention_targets = []
        engagement_targets = []
        emotion_targets = []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(self.device)
                handcrafted = batch['handcrafted_features'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                predictions = self.model(frames, handcrafted)
                losses = self.criterion(predictions, labels)
                total_loss += losses['total_loss'].item()
                attention_preds.extend((predictions['attention_score'] > 0.5).cpu().numpy())
                engagement_preds.extend(torch.argmax(predictions['engagement_logits'], dim=1).cpu().numpy())
                emotion_preds.extend(torch.argmax(predictions['emotion_logits'], dim=1).cpu().numpy())
                attention_targets.extend(labels['attention'].cpu().numpy())
                engagement_targets.extend(labels['engagement'].cpu().numpy())
                emotion_targets.extend(labels['emotion'].cpu().numpy())
        avg_loss = total_loss / len(val_loader)
        attention_f1 = f1_score(attention_targets, attention_preds, average='binary')
        engagement_f1 = f1_score(engagement_targets, engagement_preds, average='macro')
        emotion_f1 = f1_score(emotion_targets, emotion_preds, average='macro')
        return {
            'total_loss': avg_loss,
            'attention_f1': attention_f1,
            'engagement_f1': engagement_f1,
            'emotion_f1': emotion_f1
        }
    async def train(self):
        logger.info("Starting training...")
        self.setup_model()
        train_loader, val_loader = await self.prepare_data()
        num_epochs = self.config.get('num_epochs', 100)
        best_val_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader, epoch)
            if self.scheduler:
                self.scheduler.step()
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"Val Attention F1: {val_metrics['attention_f1']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            self.writer.add_scalar('Loss/Train', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('F1/Attention', val_metrics['attention_f1'], epoch)
            self.writer.add_scalar('F1/Engagement', val_metrics['engagement_f1'], epoch)
            self.writer.add_scalar('F1/Emotion', val_metrics['emotion_f1'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            if self.early_stopping(val_metrics['total_loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        logger.info("Training completed!")
        self.writer.close()
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    def export_to_onnx(self, model_path: Optional[str] = None) -> str:
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        batch_size = 1
        seq_len = self.config['model']['sequence_length']
        dummy_frames = torch.randn(batch_size, seq_len, 3, 224, 224).to(self.device)
        dummy_handcrafted = torch.randn(batch_size, seq_len, 20).to(self.device)
        onnx_path = self.output_dir / "attention_pulse_model.onnx"
        torch.onnx.export(
            self.model,
            (dummy_frames, dummy_handcrafted),
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['frames', 'handcrafted_features'],
            output_names=['attention_score', 'engagement_logits', 'emotion_logits'],
            dynamic_axes={
                'frames': {0: 'batch_size'},
                'handcrafted_features': {0: 'batch_size'},
                'attention_score': {0: 'batch_size'},
                'engagement_logits': {0: 'batch_size'},
                'emotion_logits': {0: 'batch_size'}
            }
        )
        optimizer = ONNXModelOptimizer()
        optimized_path = str(onnx_path).replace('.onnx', '_optimized.onnx')
        optimizer.optimize_model(str(onnx_path), optimized_path)
        quantized_path = str(onnx_path).replace('.onnx', '_quantized.onnx')
        try:
            optimizer.quantize_model(optimized_path, quantized_path)
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
        logger.info(f"ONNX model exported to {onnx_path}")
        logger.info(f"Optimized ONNX model saved to {optimized_path}")
        return str(onnx_path)
def create_training_config() -> Dict[str, Any]:
    return {
        'model': {
            'backbone_type': 'mobilevit',
            'sequence_length': 32,
            'handcrafted_dim': 20,
            'hidden_dim': 256,
            'temporal_type': 'cnn',
            'pretrained': True
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6
        },
        'loss': {
            'attention_weight': 1.0,
            'engagement_weight': 1.0,
            'emotion_weight': 1.0,
            'use_focal_loss': True
        },
        'early_stopping': {
            'patience': 15,
            'min_delta': 1e-4
        },
        'num_epochs': 100,
        'batch_size': 8,
        'num_workers': 4,
        'output_dir': './models_trained',
        'log_dir': './logs'
    }
async def main():
    parser = argparse.ArgumentParser(description='Train DersLens Model')
    parser.add_argument('--config', type=str, help='Path to training config file')
    parser.add_argument('--export-only', action='store_true', help='Only export existing model to ONNX')
    parser.add_argument('--model-path', type=str, help='Path to trained model for export')
    args = parser.parse_args()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_training_config()
        config_path = Path('./training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Default config saved to {config_path}")
    trainer = DersLensTrainer(config)
    if args.export_only:
        model_path = args.model_path or str(trainer.output_dir / "best_model.pth")
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return
        trainer.setup_model()
        onnx_path = trainer.export_to_onnx(model_path)
        logger.info(f"Model exported to {onnx_path}")
    else:
        await trainer.train()
        best_model_path = trainer.output_dir / "best_model.pth"
        if best_model_path.exists():
            onnx_path = trainer.export_to_onnx(str(best_model_path))
            logger.info(f"Best model exported to {onnx_path}")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())