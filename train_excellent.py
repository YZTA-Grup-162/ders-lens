"""
High-Performance MPIIGaze Training for Excellent Results
Optimized for RTX 4070 to achieve <3° MAE (EXCELLENT performance grade)
"""
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import the training classes
from mpiigaze_training import MPIIGazeTrainerAdvanced

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcellentPerformanceTrainer(MPIIGazeTrainerAdvanced):
    """High-performance trainer targeting EXCELLENT results (<3° MAE)"""
    
    def __init__(self, data_dir, model_dir="models_mpiigaze_excellent"):
        super().__init__(data_dir, model_dir, use_advanced_model=False)
        
        # RTX 4070 optimized settings for excellent performance
        self.batch_size = 256 if self.device.type == 'cuda' else 64  # Large batch for RTX 4070
        self.learning_rate = 0.001  # Higher LR for faster convergence
        self.grad_clip_norm = 0.2  # Reasonable clipping
        self.num_epochs = 80  # Optimize edilmiş ayarlarla yeterli olmalı
        
        logger.info(f"Hedef: <3° MAE")
    
    def create_enhanced_model(self):
        """Create an enhanced model for excellent performance"""
        class EnhancedGazeNet(nn.Module):
            def __init__(self):
                super(EnhancedGazeNet, self).__init__()
                
                # Enhanced architecture for better accuracy
                self.features = nn.Sequential(
                    # First block - more channels for better feature extraction
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 32x32
                    nn.Dropout2d(0.1),
                    
                    # Second block - increased capacity
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 16x16
                    nn.Dropout2d(0.15),
                    
                    # Third block - deeper features
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 8x8
                    nn.Dropout2d(0.2),
                    
                    # Final feature extraction
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((2, 2))  # 2x2
                )
                
                # Enhanced regression head
                self.regressor = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(256 * 2 * 2, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(64, 2)  # [theta, phi]
                )
                
                # Better initialization for excellent performance
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight, gain=0.1)  # Conservative but effective
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.regressor(x)
                return x
        
        return EnhancedGazeNet()
    
    def train_for_excellence(self):
        
        # Termal güvenlik kontrolü
        if not self.thermal_manager.is_safe_to_train():
            logger.error("Sistem sıcaklığı eğitim için güvenli değil!")
            return False
        
        # RTX 4070 için optimize edilmiş yüksek performans veri yükleyicileri
        self.create_data_loaders()
        
        if self.device.type == 'cuda':
            import platform
            num_workers = 6 if platform.system() == "Windows" else 12  # RTX 4070 için daha fazla worker
            pin_memory = True
            
            # Yüksek performans ayarlarıyla veri yükleyicileri yeniden oluştur
            from torch.utils.data import DataLoader
            
            self.train_loader = DataLoader(
                self.train_loader.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=4  # RTX 4070 için daha fazla prefetch
            )
            
            self.val_loader = DataLoader(
                self.val_loader.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=2
            )
        
        # Geliştirilmiş model oluştur
        model = self.create_enhanced_model().to(self.device)
        
        # Daha iyi doğruluk için geliştirilmiş kayıp fonksiyonu
        criterion = nn.MSELoss()
        
        # Yüksek performans optimizer konfigürasyonu
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.005,  # Biraz daha fazla düzenlileştirme
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate * 3,  # Temel LR'nin 3 katında tepe
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,  # Eğitimin %10'unda ısınma
            anneal_strategy='cos'
        )
        
        logger.info(f"Geliştirilmiş model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Batch boyutu: {self.batch_size}")
        logger.info(f"Öğrenme oranı: {self.learning_rate}")
        logger.info(f"Hedef: <3° MAE")

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n🎯 Epoch {epoch+1}/{self.num_epochs}")
            
            # Training with enhanced monitoring
            model.train()
            total_train_loss = 0
            total_train_mae = 0
            num_train_batches = 0
            
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Check for NaN
                if torch.isnan(loss):
                    logger.warning(f"⚠️ NaN loss at batch {batch_idx}, skipping")
                    continue
                
                loss.backward()
                
                # Gradient clipping with monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                if torch.isnan(grad_norm):
                    logger.warning(f"⚠️ NaN gradients at batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                scheduler.step()  # OneCycle needs step every batch
                
                total_train_loss += loss.item()
                mae = torch.mean(torch.abs(outputs - targets))
                total_train_mae += mae.item()
                num_train_batches += 1
                
                # Enhanced progress monitoring
                if batch_idx % 25 == 0:  # More frequent updates for big batches
                    current_mae_deg = mae.item() * 180 / np.pi
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, "
                              f"MAE: {current_mae_deg:.2f}°, LR: {current_lr:.2e}")
            
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
            avg_train_mae = total_train_mae / num_train_batches if num_train_batches > 0 else float('inf')
            
            # Enhanced validation
            model.eval()
            total_val_loss = 0
            total_val_mae = 0
            num_val_batches = 0
            
            with torch.no_grad():
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
            
            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            avg_val_mae = total_val_mae / num_val_batches if num_val_batches > 0 else float('inf')
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_maes.append(avg_train_mae)
            val_maes.append(avg_val_mae)
            
            # Convert to degrees
            train_mae_deg = avg_train_mae * 180 / np.pi
            val_mae_deg = avg_val_mae * 180 / np.pi
            
            # Enhanced performance classification
            if val_mae_deg < 2.0:
                grade = "🏆 EXCEPTIONAL"
            elif val_mae_deg < 3.0:
                grade = "🥇 EXCELLENT"
            elif val_mae_deg < 4.0:
                grade = "🥈 VERY GOOD"
            else:
                grade = "🥉 GOOD"
            
            logger.info(f"📊 Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae_deg:.2f}°")
            logger.info(f"📈 Val   - Loss: {avg_val_loss:.4f}, MAE: {val_mae_deg:.2f}° [{grade}]")
            
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
                    'val_mae_degrees': val_mae_deg,
                    'grade': grade
                }, self.model_dir / 'excellent_model_best.pth')
                
                logger.info(f"✅ New best model! Val MAE: {val_mae_deg:.2f}° [{grade}]")
                
                # Celebrate excellent results
                if val_mae_deg < 3.0:
                    logger.info("🎉 EXCELLENT PERFORMANCE ACHIEVED! (<3° MAE)")
                elif val_mae_deg < 2.0:
                    logger.info("🏆 EXCEPTIONAL PERFORMANCE! State-of-the-art accuracy!")
            else:
                patience_counter += 1
            
            # Early stopping for excellent models
            if patience_counter >= 15:  # Less patience since we want fast excellent results
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_maes': train_maes,
                    'val_maes': val_maes,
                }, self.model_dir / f'checkpoint_excellent_{epoch+1}.pth')
                logger.info(f"💾 Checkpoint saved at epoch {epoch+1}")
        
        # Final results analysis
        best_mae_deg = val_maes[best_epoch] * 180 / np.pi
        
        logger.info(f"\n🏆 HIGH-PERFORMANCE TRAINING COMPLETED!")
        logger.info(f"🎯 Best epoch: {best_epoch+1}")
        logger.info(f"📊 Best validation MAE: {best_mae_deg:.2f}°")
        
        # Performance classification
        if best_mae_deg < 2.0:
            logger.info("🏆 EXCEPTIONAL PERFORMANCE! State-of-the-art accuracy achieved!")
            logger.info("🚀 Ready for high-precision gaze tracking applications!")
        elif best_mae_deg < 3.0:
            logger.info("🥇 EXCELLENT PERFORMANCE! Professional-grade accuracy achieved!")
            logger.info("✅ Perfect for production gaze detection systems!")
        elif best_mae_deg < 4.0:
            logger.info("🥈 VERY GOOD PERFORMANCE! Solid accuracy for most applications!")
        else:
            logger.info("🥉 GOOD PERFORMANCE! Consider running advanced training for better results.")
        
        return True

def main():
    """Main training function for excellent performance"""
    data_dir = Path("backend/datasets/MPIIGaze")
    
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        return False
    
    try:
        trainer = ExcellentPerformanceTrainer(data_dir)
        success = trainer.train_for_excellence()
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
    logger.info("🎯 Starting HIGH-PERFORMANCE training for EXCELLENT results...")
    success = main()
    sys.exit(0 if success else 1)
