"""
Training Recovery Script for MPIIGaze
Recovers from NaN training issues by loading the last good checkpoint.
"""
import json
import logging
import sys
from pathlib import Path

import torch

from mpiigaze_training import MPIIGazeTrainerAdvanced

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_last_good_checkpoint(model_dir):
    """Find the most recent checkpoint before NaN issues"""
    model_dir = Path(model_dir)
    
    # Look for checkpoint files
    checkpoint_files = list(model_dir.glob("checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        logger.error("No checkpoint files found")
        return None
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    # Check each checkpoint for NaN values
    for checkpoint_path in reversed(checkpoint_files):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if training metrics contain NaN
            if 'val_maes' in checkpoint:
                maes = checkpoint['val_maes']
                if any(torch.isnan(torch.tensor(mae)) for mae in maes[-5:]):  # Check last 5 epochs
                    logger.warning(f"⚠️ Found NaN in {checkpoint_path}, skipping")
                    continue
            
            logger.info(f"✅ Found good checkpoint: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading {checkpoint_path}: {e}")
            continue
    
    logger.error("No good checkpoints found")
    return None

def recover_training():
    """Recover training from last good checkpoint"""
    data_dir = Path("backend/datasets/MPIIGaze")
    model_dir = Path("models_mpiigaze_advanced")
    
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        return False
    
    # Find last good checkpoint
    checkpoint_path = find_last_good_checkpoint(model_dir)
    if not checkpoint_path:
        logger.error("Cannot recover - no good checkpoints found")
        return False
    
    try:
        # Initialize trainer
        trainer = MPIIGazeTrainerAdvanced(data_dir, model_dir=str(model_dir), use_advanced_model=True)
        
        # Create data loaders
        trainer.create_data_loaders()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        start_epoch = checkpoint['epoch'] + 1
        
        logger.info(f"🔄 Recovering from epoch {start_epoch}")
        
        # Initialize model
        from mpiigaze_training import GazeNetAdvanced
        model = GazeNetAdvanced(num_classes=2, dropout_rate=0.3).to(trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=trainer.learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        if 'train_losses' in checkpoint:
            trainer.train_losses = checkpoint['train_losses']
            trainer.val_losses = checkpoint['val_losses']
            trainer.train_maes = checkpoint['train_maes']
            trainer.val_maes = checkpoint['val_maes']
        
        # Find best epoch
        if trainer.val_losses:
            trainer.best_val_loss = min(trainer.val_losses)
            trainer.best_epoch = trainer.val_losses.index(trainer.best_val_loss)
        
        logger.info(f"📊 Recovered state: {len(trainer.train_losses)} epochs completed")
        logger.info(f"🏆 Best validation loss so far: {trainer.best_val_loss:.4f}")
        
        # Continue training with more conservative settings
        success = trainer.train_model_advanced()
        
        if success:
            results = trainer.save_results_advanced()
            logger.info("✅ Recovery training completed successfully!")
            return True
        else:
            logger.error("Recovery training failed")
            return False
            
    except Exception as e:
        logger.error(f"Recovery error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🔄 Starting training recovery...")
    success = recover_training()
    sys.exit(0 if success else 1)
