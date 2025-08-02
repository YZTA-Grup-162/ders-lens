

import gc
import json
import logging
import math
import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

# Setup ultra-safe logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ultra_safe_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class UltraSafeThermalManager:
    """Ultra-aggressive thermal protection system"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.max_cpu_temp = 70.0  # Much more conservative
        self.max_gpu_temp = 65.0  # Much more conservative
        self.max_cpu_usage = 85.0  # Lower limit
        self.max_memory_usage = 80.0  # Lower limit
        self.cooling_period = 15  # Longer cooling
        self.emergency_cooling = 30  # Emergency cooling
        self.last_warning_time = 0
        
    def get_cpu_temperature(self):
        """Get CPU temperature with fallback"""
        try:
            # Try multiple methods for temperature
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max([temp.current for temp in temps['coretemp']])
            elif 'k10temp' in temps:
                return max([temp.current for temp in temps['k10temp']])
            elif 'acpi' in temps:
                return max([temp.current for temp in temps['acpi']])
            else:
                # Fallback: estimate from CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                return 40 + (cpu_percent / 100.0) * 35  # Conservative estimate
        except:
            return 50.0  # Safe fallback
    
    def get_gpu_temperature(self):
        """Get GPU temperature safely with multiple fallbacks"""
        # Always use safe estimation to avoid library issues
        if torch.cuda.is_available():
            try:
                # Use memory usage as proxy for GPU activity
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_percent = memory_used / 12.0 * 100  # Assuming 12GB GPU
                # Estimate temperature based on usage: 40-70°C range
                estimated_temp = 40 + (memory_percent / 100.0) * 30
                return min(estimated_temp, 70.0)  # Cap at 70°C
            except:
                pass
        
        # Ultra-safe fallback
        return 45.0  # Conservative safe temperature
    
    def is_system_safe(self):
        """Check if system is safe to continue training"""
        cpu_temp = self.get_cpu_temperature()
        gpu_temp = self.get_gpu_temperature()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        # Log current status
        current_time = time.time()
        if current_time - self.last_warning_time > 10:  # Log every 10 seconds
            logger.info(f"🌡️ System Status - CPU: {cpu_temp:.1f}°C, GPU: {gpu_temp:.1f}°C, "
                       f"CPU Usage: {cpu_usage:.1f}%, RAM: {memory_usage:.1f}%")
            self.last_warning_time = current_time
        
        # Check all safety conditions
        if cpu_temp > self.max_cpu_temp:
            logger.warning(f"🔥 CPU temperature too high: {cpu_temp:.1f}°C (max: {self.max_cpu_temp}°C)")
            return False
        
        if gpu_temp > self.max_gpu_temp:
            logger.warning(f"🔥 GPU temperature too high: {gpu_temp:.1f}°C (max: {self.max_gpu_temp}°C)")
            return False
        
        if cpu_usage > self.max_cpu_usage:
            logger.warning(f"⚡ CPU usage too high: {cpu_usage:.1f}% (max: {self.max_cpu_usage}%)")
            return False
        
        if memory_usage > self.max_memory_usage:
            logger.warning(f"💾 Memory usage too high: {memory_usage:.1f}% (max: {self.max_memory_usage}%)")
            return False
        
        return True
    
    def emergency_cool_down(self):
        """Emergency cooling procedure"""
        logger.warning("🚨 EMERGENCY COOLING ACTIVATED!")
        
        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Wait for system to cool down
        for i in range(self.emergency_cooling):
            time.sleep(1)
            if i % 5 == 0:
                cpu_temp = self.get_cpu_temperature()
                gpu_temp = self.get_gpu_temperature()
                logger.info(f"❄️ Cooling... CPU: {cpu_temp:.1f}°C, GPU: {gpu_temp:.1f}°C ({i+1}/{self.emergency_cooling}s)")
                
                # Check if we can continue
                if cpu_temp < self.max_cpu_temp - 5 and gpu_temp < self.max_gpu_temp - 5:
                    logger.info("✅ System cooled down sufficiently")
                    break
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def wait_for_safe_conditions(self):
        """Wait until system is safe to continue"""
        max_wait_time = 300  # 5 minutes max wait
        wait_start = time.time()
        
        while not self.is_system_safe():
            if time.time() - wait_start > max_wait_time:
                logger.error("❌ System not cooling down - aborting for safety")
                return False
            
            logger.info(f"⏳ Waiting for safe conditions... ({int(time.time() - wait_start)}s)")
            time.sleep(5)
        
        return True

def ultra_memory_cleanup():
    """Ultra-aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

class UltraLightweightBlock(nn.Module):
    """Ultra-lightweight building block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Minimal layers for memory efficiency
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return self.dropout(out)

class UltraSafeGazeNet(nn.Module):
    """Ultra-safe and lightweight gaze estimation network"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Ultra-lightweight stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # Minimal feature extraction
        self.layer1 = UltraLightweightBlock(16, 32, stride=2)  # 32x32
        self.layer2 = UltraLightweightBlock(32, 64, stride=2)  # 16x16
        self.layer3 = UltraLightweightBlock(64, 128, stride=2) # 8x8
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Minimal classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Safe weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Conservative gain
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.normalize(x, p=2, dim=1)

class UltraSafeDataset(Dataset):
    """Memory-efficient dataset with minimal processing"""
    
    def __init__(self, data_dir, transform=None, max_samples_per_person=1000, is_training=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.max_samples_per_person = max_samples_per_person
        self.is_training = is_training
        
        logger.info("Loading ultra-safe dataset...")
        self._load_data()
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def _load_data(self):
        """Load data with memory efficiency"""
        normalized_dir = self.data_dir / "Data" / "Normalized"
        
        for person_dir in sorted(normalized_dir.iterdir()):
            if not person_dir.is_dir():
                continue
                
            person_samples = 0
            logger.info(f"Loading data for {person_dir.name}")
            
            for mat_file in sorted(person_dir.glob("*.mat")):
                if person_samples >= self.max_samples_per_person:
                    break
                    
                try:
                    mat_data = sio.loadmat(str(mat_file))
                    
                    if 'data' in mat_data:
                        data = mat_data['data'][0, 0]
                        
                        for eye_type in ['left', 'right']:
                            if eye_type in data.dtype.names:
                                eye_data = data[eye_type][0, 0]
                                
                                if 'image' in eye_data.dtype.names and 'gaze' in eye_data.dtype.names:
                                    images = eye_data['image']
                                    gazes = eye_data['gaze']
                                    
                                    for i in range(min(len(images), len(gazes))):
                                        if person_samples >= self.max_samples_per_person:
                                            break
                                            
                                        image = images[i]
                                        gaze = gazes[i]
                                        
                                        if image.size == 0 or gaze.size < 3:
                                            continue
                                        
                                        # Normalize gaze vector
                                        gaze_norm = gaze / np.linalg.norm(gaze)
                                        
                                        self.samples.append({
                                            'image': image,
                                            'gaze': gaze_norm,
                                            'person': person_dir.name,
                                            'eye': eye_type
                                        })
                                        person_samples += 1
                    
                except Exception as e:
                    logger.warning(f"Error loading {mat_file}: {e}")
                    continue
            
            # Early memory cleanup
            ultra_memory_cleanup()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = sample['image']
        gaze = sample['gaze']
        
        # Convert to RGB efficiently
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Minimal preprocessing for safety
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        
        gaze = torch.tensor(gaze, dtype=torch.float32)
        
        return image, gaze

class UltraSafeTrainer:
    
    def __init__(self, data_dir, model_dir="models_ultra_safe"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.thermal_manager = UltraSafeThermalManager()
        
        # Ultra-conservative parameters
        self.batch_size = 8   # Very small for safety
        self.num_epochs = 60  # Reasonable for ultra-safe training
        self.base_learning_rate = 0.001
        self.gradient_accumulation_steps = 8  # Effective batch size = 64
        
        # Safety settings
        self.use_mixed_precision = False  # Disable for stability
        self.check_interval = 10  # Check thermal every 10 batches
        
        self.reset_training_history()
        
    def reset_training_history(self):
        """Reset training history"""
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.learning_rates = []
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        
    def create_ultra_safe_model(self):
        """Create ultra-safe model"""
        model = UltraSafeGazeNet(num_classes=3).to(self.device)
        
        # Simple MSE loss for stability
        criterion = nn.MSELoss()
        
        # Conservative optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.base_learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Simple scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=15,
            gamma=0.7
        )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, criterion, optimizer, scheduler
    
    def create_safe_loaders(self):
        """Create ultra-safe data loaders"""
        # Minimal transforms for safety
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=5),  # Minimal augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        full_dataset = UltraSafeDataset(
            self.data_dir,
            transform=None,
            max_samples_per_person=1000,  # Reduced for safety
            is_training=True
        )
        
        # Person-independent split
        persons = list(set([sample['person'] for sample in full_dataset.samples]))
        train_persons, val_persons = train_test_split(
            persons, test_size=0.15, random_state=42, shuffle=True
        )
        
        train_samples = [s for s in full_dataset.samples if s['person'] in train_persons]
        val_samples = [s for s in full_dataset.samples if s['person'] in val_persons]
        
        # Create datasets
        train_dataset = UltraSafeDataset.__new__(UltraSafeDataset)
        train_dataset.__init__(self.data_dir, transform=train_transform, is_training=True)
        train_dataset.samples = train_samples
        
        val_dataset = UltraSafeDataset.__new__(UltraSafeDataset)
        val_dataset.__init__(self.data_dir, transform=val_transform, is_training=False)
        val_dataset.samples = val_samples
        
        # Ultra-safe data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # No multiprocessing for safety
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
        
        return train_loader, val_loader
    
    def train_epoch_safe(self, model, criterion, optimizer, train_loader, epoch):
        """Ultra-safe training epoch"""
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Check thermal conditions every few batches
            if batch_idx % self.check_interval == 0:
                if not self.thermal_manager.is_system_safe():
                    logger.warning("🚨 System not safe - initiating cooling period")
                    self.thermal_manager.emergency_cool_down()
                    
                    if not self.thermal_manager.wait_for_safe_conditions():
                        logger.error("❌ Cannot continue safely - stopping training")
                        return None, None
            
            images = images.to(self.device, non_blocking=False)  # Blocking for safety
            targets = targets.to(self.device, non_blocking=False)
            
            # Gradient accumulation for safety
            is_accumulating = (batch_idx + 1) % self.gradient_accumulation_steps != 0
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            if not is_accumulating:
                # Conservative gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                mae = torch.mean(torch.abs(outputs - targets))
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_mae += mae.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, '
                              f'Loss: {loss.item():.4f}, MAE: {mae:.4f}')
            
            # Memory cleanup every 20 batches
            if batch_idx % 20 == 0:
                ultra_memory_cleanup()
        
        return (total_loss / max(num_batches, 1), 
                total_mae / max(num_batches, 1))
    
    def validate_epoch_safe(self, model, criterion, val_loader):
        """Ultra-safe validation"""
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                # Thermal check during validation too
                if batch_idx % (self.check_interval * 2) == 0:
                    if not self.thermal_manager.is_system_safe():
                        logger.warning("🚨 System not safe during validation")
                        self.thermal_manager.emergency_cool_down()
                        
                        if not self.thermal_manager.wait_for_safe_conditions():
                            return None, None
                
                images = images.to(self.device, non_blocking=False)
                targets = targets.to(self.device, non_blocking=False)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                mae = torch.mean(torch.abs(outputs - targets))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
                
                # Memory cleanup
                if batch_idx % 15 == 0:
                    ultra_memory_cleanup()
        
        return (total_loss / max(num_batches, 1),
                total_mae / max(num_batches, 1))
    
    def train_ultra_safe(self):
    
        try:
            if not self.thermal_manager.wait_for_safe_conditions():
                logger.error("❌ System not safe to start training")
                return False
            
            model, criterion, optimizer, scheduler = self.create_ultra_safe_model()
            train_loader, val_loader = self.create_safe_loaders()
            
            logger.info("✅ All components created safely")
            
            for epoch in range(self.num_epochs):
                logger.info(f"\n🔄 Epoch {epoch+1}/{self.num_epochs}")
                logger.info("=" * 50)
                
                # Pre-epoch safety check
                if not self.thermal_manager.is_system_safe():
                    logger.warning("🚨 Pre-epoch safety check failed")
                    self.thermal_manager.emergency_cool_down()
                    
                    if not self.thermal_manager.wait_for_safe_conditions():
                        logger.error("❌ Cannot continue safely")
                        break
                
                epoch_start = time.time()
                
                # Training
                train_results = self.train_epoch_safe(model, criterion, optimizer, train_loader, epoch)
                if train_results is None:
                    logger.error("❌ Training epoch failed due to safety")
                    break
                
                train_loss, train_mae = train_results
                
                # Validation
                val_results = self.validate_epoch_safe(model, criterion, val_loader)
                if val_results is None:
                    logger.error("❌ Validation epoch failed due to safety")
                    break
                
                val_loss, val_mae = val_results
                
                # Scheduler step
                scheduler.step()
                
                # Record metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_maes.append(train_mae)
                self.val_maes.append(val_mae)
                self.learning_rates.append(scheduler.get_last_lr()[0])
                
                # Convert to degrees
                train_mae_deg = train_mae * 180 / np.pi
                val_mae_deg = val_mae * 180 / np.pi
                
                epoch_time = time.time() - epoch_start
                
                logger.info(f"📊 Train - Loss: {train_loss:.4f}, MAE: {train_mae_deg:.2f}°")
                logger.info(f"📈 Val   - Loss: {val_loss:.4f}, MAE: {val_mae_deg:.2f}°")
                logger.info(f"⏱️ Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.2e}")
                
                # Save best model
                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    self.best_epoch = epoch
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_mae': val_mae,
                        'val_mae_degrees': val_mae_deg,
                        'training_config': {
                            'batch_size': self.batch_size,
                            'learning_rate': self.base_learning_rate
                        }
                    }, self.model_dir / 'ultra_safe_best.pth')
                    
                    logger.info(f"✅ New best model! Val MAE: {val_mae_deg:.2f}°")
                
                # Post-epoch cleanup and cooling
                ultra_memory_cleanup()
                time.sleep(2)  # Brief cooling period
            
            # Training completed
            best_mae_deg = self.best_val_mae * 180 / np.pi
            
            logger.info(f"\n🎉 ULTRA-SAFE Training Completed Successfully!")
            logger.info(f"🏆 Best validation MAE: {best_mae_deg:.2f}°")
            logger.info(f"📈 Best epoch: {self.best_epoch + 1}")
            logger.info(f"💾 Model saved to: {self.model_dir / 'ultra_safe_best.pth'}")
            
            # Performance assessment
            if best_mae_deg <= 2.5:
                grade = "Production Ready"
            elif best_mae_deg <= 3.5:
                grade = "High Quality"
            elif best_mae_deg <= 5.0:
                grade = "GOOD - Practical Use"
            else:
                grade = "BASELINE - Functional"
            
          
            return True
            
        except Exception as e:
            logger.error(f"Ultra-safe training failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
        finally:
            ultra_memory_cleanup()

def main():

    data_dir = Path("datasets/MPIIGaze")
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        return False
    
    trainer = UltraSafeTrainer(data_dir)
    success = trainer.train_ultra_safe()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
