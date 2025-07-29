"""
Enhanced Stable Gaze Training Module for DersLens
Integrates the excellent stable trainer with performance optimizations
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the base training classes
try:
    # Try importing from the project root
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from mpiigaze_training import GazeNet, MPIIGazeTrainerAdvanced
    from thermal_management import ThermalManager
except ImportError:
    # Fallback - simplified version if imports fail
    class MPIIGazeTrainerAdvanced:
        def __init__(self, data_dir, model_dir, use_advanced_model=False):
            self.data_dir = Path(data_dir)
            self.model_dir = Path(model_dir)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    class ThermalManager:
        def __init__(self):
            pass
        def is_safe_to_train(self):
            return True
        def stop_monitoring(self):
            pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DersLensGazeTrainer(MPIIGazeTrainerAdvanced):
    """
    Enhanced stable gaze trainer optimized for DersLens production use
    Focuses on achieving excellent results (<3° MAE) with thermal safety
    """
    
    def __init__(self, data_dir: str, model_dir: str = "models_mpiigaze_derslens"):
        """Initialize the DersLens gaze trainer"""
        try:
            super().__init__(data_dir, model_dir, use_advanced_model=False)
        except:
            # Fallback initialization if parent class unavailable
            self.data_dir = Path(data_dir)
            self.model_dir = Path(model_dir)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Ensure model directory exists
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Enhanced settings for excellent performance on RTX 4070
        self.batch_size = 256 if self.device.type == 'cuda' else 64
        self.learning_rate = 0.001
        self.grad_clip_norm = 0.2
        self.num_epochs = 80
        
        # Initialize thermal manager if available
        try:
            self.thermal_manager = ThermalManager()
        except:
            self.thermal_manager = None
            
        logger.info("DersLens Enhanced Gaze Trainer initialized")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"⚡ Learning rate: {self.learning_rate}")
    
    def create_enhanced_model(self) -> nn.Module:
        """Create the enhanced stable model architecture"""
        
        class EnhancedStableGazeNet(nn.Module):
            """Enhanced stable architecture for excellent gaze estimation"""
            
            def __init__(self):
                super(EnhancedStableGazeNet, self).__init__()
                
                # Enhanced feature extraction
                self.features = nn.Sequential(
                    # First block - enhanced for better feature extraction
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 32x32
                    
                    # Second block
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 16x16
                    
                    # Third block
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 8x8
                    
                    # Global average pooling for better generalization
                    nn.AdaptiveAvgPool2d((2, 2))
                )
                
                # Enhanced regression head
                self.regressor = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(128 * 2 * 2, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 2)
                )
                
                # Improved initialization
                self._init_weights()
            
            def _init_weights(self):
                """Improved weight initialization for excellent performance"""
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.regressor(x)
                return x
        
        return EnhancedStableGazeNet()
    
    def classify_performance(self, mae_degrees: float) -> Tuple[str, str]:
        """Classify performance based on MAE in degrees"""
        if mae_degrees < 3.0:
            return "EXCELLENT", "excellent"
        elif mae_degrees < 5.0:
            return "VERY GOOD", "very_good"
        elif mae_degrees < 8.0:
            return "GOOD", "good"
        else:
            return "TRAINING", "training"
    
    def create_optimized_data_loaders(self) -> bool:
        """Create GPU-optimized data loaders"""
        try:
            # Try to use parent method first
            if hasattr(super(), 'create_data_loaders'):
                super().create_data_loaders()
            
            # Override with GPU optimizations if CUDA available
            if self.device.type == 'cuda' and hasattr(self, 'train_loader'):
                import platform
                num_workers = 4 if platform.system() == "Windows" else 8
                pin_memory = True
                
                # Recreate with better settings
                self.train_loader = DataLoader(
                    self.train_loader.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=num_workers > 0
                )
                
                self.val_loader = DataLoader(
                    self.val_loader.dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=num_workers > 0
                )
            
            return True
        except Exception as e:
            logger.warning(f"Could not create optimized data loaders: {e}")
            return False
    
    def train_for_derslens(self) -> Dict[str, Any]:
        """
        Train the gaze model specifically for DersLens integration
        Returns comprehensive training results
        """
        logger.info("Starting DersLens gaze training...")
        
        # Thermal safety check
        if self.thermal_manager and not self.thermal_manager.is_safe_to_train():
            logger.error("System temperature unsafe for training!")
            return {"success": False, "error": "thermal_unsafe"}
        
        # Create data loaders
        if not self.create_optimized_data_loaders():
            logger.error("Failed to create data loaders")
            return {"success": False, "error": "data_loader_failed"}
        
        # Create model
        model = self.create_enhanced_model().to(self.device)
        criterion = nn.MSELoss()
        
        # GPU-optimized optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.001,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # OneCycle scheduler for excellent performance
        if hasattr(self, 'train_loader'):
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * 2,
                steps_per_epoch=len(self.train_loader),
                epochs=self.num_epochs,
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=10,
                final_div_factor=100
            )
        else:
            scheduler = None
        
        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🔥 Model parameters: {model_params:,}")
        
        # Training tracking
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        train_metrics = {
            'losses': [],
            'maes': [],
            'mae_degrees': []
        }
        val_metrics = {
            'losses': [],
            'maes': [],
            'mae_degrees': []
        }
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            model.train()
            total_train_loss = 0
            total_train_mae = 0
            num_train_batches = 0
            
            if hasattr(self, 'train_loader'):
                for batch_idx, (images, targets) in enumerate(self.train_loader):
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    
                    # NaN safety check
                    if torch.isnan(loss):
                        logger.warning(f"⚠️ NaN loss at batch {batch_idx}, skipping")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                    if torch.isnan(grad_norm):
                        logger.warning(f"⚠️ NaN gradients at batch {batch_idx}, skipping")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    
                    total_train_loss += loss.item()
                    mae = torch.mean(torch.abs(outputs - targets))
                    total_train_mae += mae.item()
                    num_train_batches += 1
                    
                    # Progress logging
                    if batch_idx % 50 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        mae_deg = mae.item() * 180 / np.pi
                        logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, "
                                  f"Loss: {loss.item():.4f}, MAE: {mae_deg:.2f}°, LR: {current_lr:.2e}")
            
            # Calculate training averages
            avg_train_loss = total_train_loss / max(num_train_batches, 1)
            avg_train_mae = total_train_mae / max(num_train_batches, 1)
            train_mae_deg = avg_train_mae * 180 / np.pi
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            total_val_mae = 0
            num_val_batches = 0
            
            with torch.no_grad():
                if hasattr(self, 'val_loader'):
                    for images, targets in self.val_loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                        
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                        
                        if not torch.isnan(loss):
                            total_val_loss += loss.item()
                            mae = torch.mean(torch.abs(outputs - targets))
                            total_val_mae += mae.item()
                            num_val_batches += 1
            
            # Calculate validation averages
            avg_val_loss = total_val_loss / max(num_val_batches, 1)
            avg_val_mae = total_val_mae / max(num_val_batches, 1)
            val_mae_deg = avg_val_mae * 180 / np.pi
            
            # Store metrics
            train_metrics['losses'].append(avg_train_loss)
            train_metrics['maes'].append(avg_train_mae)
            train_metrics['mae_degrees'].append(train_mae_deg)
            
            val_metrics['losses'].append(avg_val_loss)
            val_metrics['maes'].append(avg_val_mae)
            val_metrics['mae_degrees'].append(val_mae_deg)
            
            # Performance classification
            performance_text, performance_class = self.classify_performance(val_mae_deg)
            
            # Logging
            logger.info(f"Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae_deg:.2f}°")
            logger.info(f"📈 Val   - Loss: {avg_val_loss:.4f}, MAE: {val_mae_deg:.2f}° [{performance_text}]")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save model
                model_path = self.model_dir / 'derslens_gaze_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_mae': avg_val_mae,
                    'val_mae_degrees': val_mae_deg,
                    'performance_class': performance_class,
                    'model_architecture': 'EnhancedStableGazeNet',
                    'training_config': {
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate,
                        'optimizer': 'AdamW',
                        'scheduler': 'OneCycleLR'
                    }
                }, model_path)
                
                logger.info(f"✅ New best model! Val MAE: {val_mae_deg:.2f}° [{performance_text}]")
                
                # Celebrate excellent performance
                if val_mae_deg < 3.0:
                    logger.info("🎉🎉EXCELLENT PERFORMANCE ACHIEVED! (<3° MAE) 🎉🎉🎉")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 25:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if (epoch + 1) % 20 == 0:
                checkpoint_path = self.model_dir / f'checkpoint_derslens_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }, checkpoint_path)
                logger.info(f"💾 Checkpoint saved at epoch {epoch+1}")
        
        # Final results
        best_val_mae_deg = val_metrics['mae_degrees'][best_epoch]
        final_performance_text, final_performance_class = self.classify_performance(best_val_mae_deg)
        
        logger.info(f"\n✅ DersLens gaze training completed!")
        logger.info(f"🏆 Best epoch: {best_epoch+1}")
        logger.info(f"Best validation MAE: {best_val_mae_deg:.2f}° [{final_performance_text}]")
        
        # Cleanup thermal manager
        if self.thermal_manager:
            self.thermal_manager.stop_monitoring()
        
        return {
            "success": True,
            "best_epoch": best_epoch + 1,
            "best_val_mae": best_val_mae_deg,
            "performance_class": final_performance_class,
            "performance_text": final_performance_text,
            "model_path": str(self.model_dir / 'derslens_gaze_best.pth'),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "model_params": model_params,
            "device": str(self.device)
        }
    
    def load_best_model(self) -> Optional[nn.Module]:
        """Load the best trained model for inference"""
        model_path = self.model_dir / 'derslens_gaze_best.pth'
        
        if not model_path.exists():
            logger.error(f"Best model not found at {model_path}")
            return None
        
        try:
            model = self.create_enhanced_model()
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info(f"✅ Loaded best model from epoch {checkpoint['epoch']+1}")
            logger.info(f"Model VAL MAE: {checkpoint['val_mae_degrees']:.2f}°")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

def train_derslens_gaze_model(data_dir: str, model_dir: str = "models_mpiigaze_derslens") -> Dict[str, Any]:
    """
    Convenience function to train gaze model for DersLens
    
    Args:
        data_dir: Path to MPIIGaze dataset
        model_dir: Directory to save models
        
    Returns:
        Dictionary with training results
    """
    try:
        trainer = DersLensGazeTrainer(data_dir, model_dir)
        results = trainer.train_for_derslens()
        return results
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Test the trainer
    data_dir = "backend/datasets/MPIIGaze"
    
    if Path(data_dir).exists():
        logger.info("Starting DersLens gaze model training...")
        results = train_derslens_gaze_model(data_dir)
        
        if results["success"]:
            logger.info("Training completed successfully!")
            logger.info(f"Best MAE: {results['best_val_mae']:.2f}° ({results['performance_text']})")
        else:
            logger.error(f"Training failed: {results.get('error', 'Unknown error')}")
    else:
        logger.error(f"Dataset not found at {data_dir}")
