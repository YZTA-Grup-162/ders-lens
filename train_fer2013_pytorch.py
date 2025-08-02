

import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
NUM_CLASSES = 7
# Support both naming conventions
EMOTIONS_LOWERCASE = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] 
EMOTIONS_CAPITALIZED = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = 48
BATCH_SIZE = 64  # Increased for faster convergence with more stable gradients
EPOCHS = 100
LEARNING_RATE = 1e-3  # Increased for faster initial learning

class FER2013Dataset(Dataset):
    """Custom Dataset for FER2013 data"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Detect emotion naming convention
        self.emotions = self._detect_emotion_names(data_dir)
        print(f"Detected emotion names: {self.emotions}")
        
        # Load samples from directory structure
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_name)
                        self.samples.append((img_path, emotion_idx))
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _detect_emotion_names(self, data_dir):
        """Detect which emotion naming convention is used"""
        if not os.path.exists(data_dir):
            return EMOTIONS_CAPITALIZED  # Default fallback
            
        existing_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Check for capitalized names first (augmented dataset)
        if any(emotion in existing_dirs for emotion in EMOTIONS_CAPITALIZED):
            return EMOTIONS_CAPITALIZED
        # Check for lowercase names (standard dataset)
        elif any(emotion in existing_dirs for emotion in EMOTIONS_LOWERCASE):
            return EMOTIONS_LOWERCASE
        else:
            print(f"Warning: No recognized emotion directories found in {data_dir}")
            print(f"Available directories: {existing_dirs}")
            return EMOTIONS_CAPITALIZED  # Default fallback
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FER2013Net(nn.Module):
    
    def __init__(self, num_classes=7, use_resnet=False):
        super(FER2013Net, self).__init__()
        
        if use_resnet:
            # Keep ResNet option but simplified
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            self.use_resnet = True
        else:
            # Simple CNN architecture like successful Kaggle approach
            self.use_resnet = False
            
            # Convolutional layers with larger initial features
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Increased from 32 to 64
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout2d(0.25)
            
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Increased from 64 to 128
            self.bn2 = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.dropout2 = nn.Dropout2d(0.25)
            
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Increased from 128 to 256
            self.bn3 = nn.BatchNorm2d(256)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.dropout3 = nn.Dropout2d(0.25)
            
            # Calculate flattened size: 256 * (48//8) * (48//8) = 256 * 6 * 6
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256 * 6 * 6, 512)  # Updated for 256 channels
            self.bn4 = nn.BatchNorm1d(512)
            self.dropout4 = nn.Dropout(0.5)
            
            self.fc2 = nn.Linear(512, 256)
            self.bn5 = nn.BatchNorm1d(256)
            self.dropout5 = nn.Dropout(0.3)  # Reduced dropout in final layer
            
            self.fc3 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        if self.use_resnet:
            return self.backbone(x)
        else:
            # Simple CNN forward pass
            x = self.dropout1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
            x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
            x = self.dropout3(self.pool3(torch.relu(self.bn3(self.conv3(x)))))
            
            x = self.flatten(x)
            x = self.dropout4(torch.relu(self.bn4(self.fc1(x))))
            x = self.dropout5(torch.relu(self.bn5(self.fc2(x))))
            x = self.fc3(x)
            
            return x

def get_transforms():
    """Get enhanced data transforms based on successful Kaggle approach"""
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced from 0.5
        transforms.RandomRotation(degrees=15),   # Reduced from 30 - less aggressive
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),    # Reduced from (0.2, 0.2)
            scale=(0.9, 1.1),        # Reduced from (0.8, 1.2)
            shear=(-10, 10, -10, 10) # Reduced from (-20, 20, -20, 20)
        ),
        transforms.ColorJitter(
            brightness=(0.9, 1.1),   # Reduced from (0.8, 1.2)
            contrast=0.2,            # Reduced from 0.3
            saturation=0.2,          # Reduced from 0.3
            hue=0.05                 # Reduced from 0.1
        ),
        transforms.ToTensor(),
        # Use [0,1] normalization instead of ImageNet stats - KEY CHANGE!
        # This matches the successful Kaggle approach
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # No normalization for [0,1] range
    ])
    
    return train_transform, val_transform

def calculate_class_weights(dataset):
    """Calculate class weights for balanced training"""
    labels = [sample[1] for sample in dataset.samples]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequency
    weights = total_samples / (NUM_CLASSES * class_counts)
    return torch.FloatTensor(weights).to(device)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(dataloader), 100. * correct / total, all_preds, all_targets

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('models_fer2013/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()  # Auto-close to continue training

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix with proper class handling"""
    # Ensure all classes are represented
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models_fer2013/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()  # Auto-close to continue training
    
    # Print classification report with proper error handling
    print("\nClassification Report:")
    try:
        print(classification_report(y_true, y_pred, 
                                  target_names=class_names, 
                                  labels=list(range(len(class_names))), 
                                  digits=4, 
                                  zero_division=0))
    except Exception as e:
        print(f"⚠️  Classification report error: {e}")
        # Manual calculation as fallback
        unique_classes = sorted(set(y_true))
        print(f"Classes present in validation: {[class_names[i] for i in unique_classes]}")
        for i, class_name in enumerate(class_names):
            if i in unique_classes:
                class_correct = sum(1 for t, p in zip(y_true, y_pred) if t == i and p == i)
                class_total = sum(1 for t in y_true if t == i)
                if class_total > 0:
                    print(f"{class_name}: {class_correct}/{class_total} = {100*class_correct/class_total:.2f}%")

def save_model(model, save_path, epoch, accuracy, emotions):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'num_classes': NUM_CLASSES,
        'emotions': emotions,
        'img_size': IMG_SIZE
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path} with accuracy: {accuracy:.2f}%")

def main():
    """Main training function"""
    
    # Data paths - check both augmented and standard datasets
    augmented_train_dir = "datasets/augmented-fe22013/train"
    standard_train_dir = "datasets/fer2013/train"
    standard_val_dir = "datasets/fer2013/validation"
    
    # Choose the best available dataset
    if os.path.exists(augmented_train_dir):
        print("🎯 Using Augmented FER2013 dataset (balanced classes)")
        train_data_dir = augmented_train_dir
        val_data_dir = None  # We'll create validation split from training data
        use_augmented = True
    elif os.path.exists(standard_train_dir):
        print("📊 Using Standard FER2013 dataset")
        train_data_dir = standard_train_dir
        val_data_dir = standard_val_dir if os.path.exists(standard_val_dir) else None
        use_augmented = False
    else:
        print("❌ No FER2013 dataset found!")
        print("Please organize your FER2013 data in one of these structures:")
        print("datasets/augmented-fe22013/train/")
        print("├── Anger/")
        print("├── Disgust/")
        print("├── Fear/")
        print("├── Happy/")
        print("├── Neutral/")
        print("├── Sad/")
        print("└── Surprise/")
        print("\nOR\n")
        print("datasets/fer2013/")
        print("├── train/")
        print("│   ├── angry/")
        print("│   ├── disgust/")
        print("│   ├── fear/")
        print("│   ├── happy/")
        print("│   ├── neutral/")
        print("│   ├── sad/")
        print("│   └── surprise/")
        print("└── validation/")
        print("    ├── angry/")
        print("    ├── disgust/")
        print("    ├── fear/")
        print("    ├── happy/")
        print("    ├── neutral/")
        print("    ├── sad/")
        print("    └── surprise/")
        return
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    print("📁 Loading datasets...")
    full_train_dataset = FER2013Dataset(train_data_dir, transform=train_transform)
    emotions = full_train_dataset.emotions  # Get the detected emotion names
    
    if use_augmented or val_data_dir is None:
        # Split training data into train/validation for augmented dataset
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset_temp = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )
        
        # Create validation dataset with validation transforms
        val_dataset = FER2013Dataset(train_data_dir, transform=val_transform)
        val_indices = val_dataset_temp.indices
        val_dataset.samples = [val_dataset.samples[i] for i in val_indices]
        
        print(f"📊 Created train/val split: {len(train_dataset)}/{len(val_dataset)}")
    else:
        # Use existing validation directory
        train_dataset = full_train_dataset
        val_dataset = FER2013Dataset(val_data_dir, transform=val_transform)
        print(f"📊 Using existing validation split: {len(train_dataset)}/{len(val_dataset)}")
    
    # Create data loaders (num_workers=0 for Windows compatibility)
    import platform
    num_workers = 0 if platform.system() == "Windows" else 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"🎭 Emotion classes: {emotions}")
    
    # Create model
    print("🏗️ Creating model...")
    # Use simple CNN by default (like successful Kaggle approach)
    model = FER2013Net(num_classes=NUM_CLASSES, use_resnet=False).to(device)
    
    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(full_train_dataset)
    print(f"📊 Class weights: {class_weights}")
    
    # Enhanced loss and optimizer - using standard Adam like successful Kaggle approach
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)  # More aggressive LR reduction
    
    # Training history and early stopping
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    patience = 8  # Reduced from 12 for faster stopping if not improving
    patience_counter = 0
    
    print("🚀 Starting enhanced training...")
    print(f"🎯 Target: Validation accuracy >85% with <5% overfitting gap")
    print(f"🔧 Key improvements: Faster LR, less aggressive augmentation, larger model")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate with validation loss
        scheduler.step(val_loss)
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Calculate overfitting gap
        overfitting_gap = train_acc - val_acc
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Overfitting Gap: {overfitting_gap:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Health check warnings
        if overfitting_gap > 8:
            print("⚠️  WARNING: High overfitting gap detected!")
        elif overfitting_gap < 3:
            print("✅ Healthy training gap maintained")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            save_model(model, 'models_fer2013/fer2013_pytorch_best.pth', epoch, val_acc, emotions)
            
            print(f"🏆 New best validation accuracy: {val_acc:.2f}%")
            print("\nClassification Report:")
            # Fix: Ensure all classes are represented in the classification report
            try:
                print(classification_report(val_targets, val_preds, 
                                          target_names=emotions, 
                                          labels=list(range(NUM_CLASSES)), 
                                          digits=4, 
                                          zero_division=0))
            except Exception as e:
                print(f"⚠️  Classification report error: {e}")
                # Fallback: simple accuracy report
                unique_classes = sorted(set(val_targets))
                print(f"Detected classes in validation: {[emotions[i] for i in unique_classes]}")
                for i, emotion in enumerate(emotions):
                    if i in unique_classes:
                        class_acc = sum(1 for t, p in zip(val_targets, val_preds) if t == i and p == i)
                        class_total = sum(1 for t in val_targets if t == i)
                        if class_total > 0:
                            print(f"{emotion}: {class_acc}/{class_total} = {100*class_acc/class_total:.2f}%")
            
            # Save confusion matrix for best model
            if epoch > 0:
                plot_confusion_matrix(val_targets, val_preds, emotions)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement")
            print(f"Best validation accuracy: {best_acc:.2f}%")
            break
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    print(f"\n🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")
    
    # Convert to ONNX for deployment
    print("🔄 Converting to ONNX...")
    convert_to_onnx(model, 'models_fer2013/fer2013_pytorch.onnx')

def convert_to_onnx(model, save_path):
    """Convert PyTorch model to ONNX format"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model saved to {save_path}")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models_fer2013', exist_ok=True)
    
    main()
