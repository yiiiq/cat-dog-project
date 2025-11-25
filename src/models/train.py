"""
Training script for Cat vs Dog classifier

AI Usage: This code was developed with assistance from GitHub Copilot (AI pair programmer).
Approximately 50% AI-generated with significant human design, debugging, and optimization.
See AI_USAGE.md for detailed attribution.
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from google.cloud import storage
from PIL import Image
import io
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class SimpleCNN(nn.Module):
    """Simple CNN for binary classification (cat vs dog)"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers (128x128 input -> 16x16 after 3 pooling)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(self.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(self.relu(self.conv2(x)))  # 64 -> 32
        x = self.pool(self.relu(self.conv3(x)))  # 32 -> 16
        
        # Flatten
        x = x.view(-1, 128 * 16 * 16)
        
        # FC layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        
        return x


class GCSImageDataset(Dataset):
    """Dataset that loads images from Google Cloud Storage"""
    def __init__(self, bucket_name, prefix, transform=None):
        self.bucket_name = bucket_name
        self.transform = transform
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Load cats (label 0)
        cat_prefix = f"{prefix}/cats/"
        cat_blobs = list(self.bucket.list_blobs(prefix=cat_prefix))
        for blob in cat_blobs:
            if blob.name.endswith('.jpg') or blob.name.endswith('.jpeg'):
                self.image_paths.append(blob.name)
                self.labels.append(0)
        
        # Load dogs (label 1)
        dog_prefix = f"{prefix}/dogs/"
        dog_blobs = list(self.bucket.list_blobs(prefix=dog_prefix))
        for blob in dog_blobs:
            if blob.name.endswith('.jpg') or blob.name.endswith('.jpeg'):
                self.image_paths.append(blob.name)
                self.labels.append(1)
        
        print(f"Loaded {len(self.labels)} images ({self.labels.count(0)} cats, {self.labels.count(1)} dogs)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Download image from GCS
        blob = self.bucket.blob(self.image_paths[idx])
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_transforms():
    """Define image transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Collect predictions
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return val_loss, val_acc, precision, recall, f1


def main():
    # Load configuration
    config = load_config()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load full dataset
    print("Loading dataset from GCS...")
    full_dataset = GCSImageDataset(
        bucket_name=config['data']['bucket'],
        prefix=config['data']['processed_path'],
        transform=train_transform
    )
    
    # Split dataset
    val_split = config['train']['val_split']
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['train']['seed'])
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    model = SimpleCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model': 'SimpleCNN',
            'epochs': config['train']['epochs'],
            'batch_size': config['train']['batch_size'],
            'learning_rate': config['train']['lr'],
            'val_split': config['train']['val_split'],
            'seed': config['train']['seed'],
            'device': str(device)
        })
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(config['train']['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }, step=epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'models/best_model.pth')
                mlflow.log_artifact('models/best_model.pth')
                print(f"Saved new best model with val_acc: {val_acc:.4f}")
        
        # Log final metrics
        mlflow.log_metric('best_val_acc', best_val_acc)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
