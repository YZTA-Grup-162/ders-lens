
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import the training classes
from mpiigaze_training import GazeNet, MPIIGazeTrainerAdvanced

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleStableTrainer(MPIIGazeTrainerAdvanced):
    """Ultra-stable trainer focused on achieving excellent results"""
    
    def __init__(self, data_dir, model_dir="models_mpiigaze_stable"):
        super().__init__(data_dir, model_dir, use_advanced_model=False)
        
        # Optimized settings for excellent performance on RTX 4070
        if self.device.type == 'cuda':
            # Dynamically set batch size based on available GPU memory
            gpu_props = torch.cuda.get_device_properties(self.device)
            total_mem_gb = gpu_props.total_memory / (1024 ** 3)
            # Use a conservative estimate: 256 for >=12GB, 128 for >=8GB, 64 for <8GB
            if total_mem_gb >= 12:
                self.batch_size = 256
            elif total_mem_gb >= 8:
                self.batch_size = 128
            else:
                self.batch_size = 64
        else:
            self.batch_size = 64  # Default for CPU
        self.learning_rate = 0.001  # Higher learning rate for faster convergence
        self.grad_clip_norm = 0.2  # Less conservative for better performance
        self.num_epochs = 80  # Sufficient for excellent results
        
        logger.info("🛡️ Using ultra-stable training configuration")
    
    def create_simple_model(self):
        """Create an enhanced model for excellent performance while maintaining stability"""
        class EnhancedStableGazeNet(nn.Module):
            def __init__(self):
                super(EnhancedStableGazeNet, self).__init__()
                
                # Enhanced architecture for better accuracy
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
                
                # Improved initialization for better performance
                self._init_weights()
            
            def _init_weights(self):
                """Improved initialization for excellent performance"""
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
    
    def train_stable(self):
        """Ultra-stable training loop"""
        logger.info("🛡️ Starting ultra-stable training...")
        
        # Check thermal safety
        if not self.thermal_manager.is_safe_to_train():
            logger.error("System temperature unsafe for training!")
            return False
        
        # GPU-optimized data loaders
        self.create_data_loaders()
        
        # Override the parent's conservative data loader settings for GPU efficiency
        if self.device.type == 'cuda':
            import os
            import platform
            max_workers = 8
            cpu_count = os.cpu_count() or 1
            num_workers = min(max_workers, cpu_count)
            pin_memory = True
            
            # Recreate data loaders with better settings
            from torch.utils.data import DataLoader
            
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
        
        # Create simple stable model
        model = self.create_simple_model().to(self.device)
        
        # Simple MSE loss only
        criterion = nn.MSELoss()
        
        # GPU-optimized optimizer
        optimizer = optim.AdamW(  # AdamW is more GPU efficient
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.001,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # OneCycle scheduler for excellent performance
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
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Eğitim
            model.train()
            total_train_loss = 0
            total_train_mae = 0
            num_train_batches = 0
            
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)  # GPU verimliliği için non-blocking
                targets = targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # NaN kontrolü
                if torch.isnan(loss):
                    logger.warning(f"Batch {batch_idx}'da NaN loss, atlanıyor")
                    continue
                
                loss.backward()
                
                # Gradient kontrolü
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                if torch.isnan(grad_norm):
                    logger.warning(f"Batch {batch_idx}'da NaN gradients, atlanıyor")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                scheduler.step()  # OneCycle scheduler steps after each batch
                
                total_train_loss += loss.item()
                mae = torch.mean(torch.abs(outputs - targets))
                total_train_mae += mae.item()
                num_train_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, MAE: {mae.item():.4f}")
                    
                # Her epoch'un %25'inde ilerleme göster
                if batch_idx > 0 and batch_idx % (len(self.train_loader) // 4) == 0:
                    progress = (batch_idx / len(self.train_loader)) * 100
                    logger.info(f"Epoch ilerlemesi: {progress:.0f}% | Mevcut MAE: {mae.item() * 180 / np.pi:.2f}°")
            
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
            avg_train_mae = total_train_mae / num_train_batches if num_train_batches > 0 else float('inf')
            
            # Validation
            model.eval()
            total_val_loss = 0
            total_val_mae = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for images, targets in self.val_loader:
                    images = images.to(self.device, non_blocking=True)  # Non-blocking for GPU
                    targets = targets.to(self.device, non_blocking=True)
                    
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    
                    if not torch.isnan(loss):
                        total_val_loss += loss.item()
                        mae = torch.mean(torch.abs(outputs - targets))
                        total_val_mae += mae.item()
                        num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            avg_val_mae = total_val_mae / num_val_batches if num_val_batches > 0 else float('inf')
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_maes.append(avg_train_mae)
            val_maes.append(avg_val_mae)
            
            # Dereceye çevir
            train_mae_deg = avg_train_mae * 180 / np.pi
            val_mae_deg = avg_val_mae * 180 / np.pi
            
            logger.info(f"Eğitim - Loss: {avg_train_loss:.4f}, MAE: {train_mae_deg:.2f}°")
            logger.info(f"Doğrulama - Loss: {avg_val_loss:.4f}, MAE: {val_mae_deg:.2f}°")
            
            # Performans sınıflandırması
            if val_mae_deg < 3.0:
                performance = "İSTENİLEN"
            elif val_mae_deg < 5.0:
                performance = "ÇOK İYİ"
            elif val_mae_deg < 8.0:
                performance = "İYİ"
            else:
                performance = "EĞİTİMDE"
            
            logger.info(f"Performans: {performance}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_mae': avg_val_mae,
                }, self.model_dir / 'simple_stable_best.pth')
                
                best_mae_deg = avg_val_mae * 180 / np.pi
                logger.info(f"Yeni en iyi model! MAE: {best_mae_deg:.2f}°")
                
                # Celebrate excellent performance
                if best_mae_deg < 3.0:
                    logger.info("🎉🎉🎉 EXCELLENT PERFORMANCE ACHIEVED! (<3° MAE) 🎉🎉🎉")
            else:
                patience_counter += 1
            
            # Early stopping with less patience for faster training
            if patience_counter >= 25:  # Reduced from 50
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Checkpoint kaydet
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_maes': train_maes,
                    'val_maes': val_maes,
                }, self.model_dir / f'checkpoint_stable_{epoch+1}.pth')
        
        # Son sonuçlar
        best_mae_deg = val_maes[best_epoch] * 180 / np.pi
        logger.info(f"\nKararlı eğitim tamamlandı!")
        logger.info(f"En iyi epoch: {best_epoch+1}")
        logger.info(f"En iyi doğrulama MAE: {best_mae_deg:.2f}°")
        
        if best_mae_deg < 3.0:
            logger.info("PERFORMANS: İSTENİLEN (<3° MAE)")
        elif best_mae_deg < 5.0:
            logger.info("PERFORMANS: ÇOK İYİ (<5° MAE)")
        elif best_mae_deg < 8.0:
            logger.info("PERFORMANS: İYİ")

        return True

def main():
    """Main training function"""
    data_dir = Path("backend/datasets/MPIIGaze")
    
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        return False
    
    try:
        trainer = SimpleStableTrainer(data_dir)
        success = trainer.train_stable()
        return success
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False
    finally:
        if 'trainer' in locals() and hasattr(trainer, 'thermal_manager'):
            trainer.thermal_manager.stop_monitoring()

if __name__ == "__main__":
    logger.info("🛡️ Starting simple stable MPIIGaze training...")
    success = main()
    sys.exit(0 if success else 1)
