"""
Validate a trained model on the validation set
"""
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.train import SimpleCNN, GCSImageDataset, load_config


def validate_model(model_path='models/best_model.pth', config_path='configs/config.yaml'):
    """Validate a trained model"""
    
    # Load config
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # Load dataset with validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nLoading dataset from GCS...")
    full_dataset = GCSImageDataset(
        bucket_name=config['data']['bucket'],
        prefix=config['data']['processed_path'],
        transform=val_transform
    )
    
    # Split dataset (same way as training)
    val_split = config['train']['val_split']
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    _, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['train']['seed'])
    )
    
    print(f"Validation set size: {val_size}")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False
    )
    
    # Validation
    print("\nRunning validation...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            
            # Get predictions
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Cat    Dog")
    print(f"Actual Cat  {conf_matrix[0][0]:4d}  {conf_matrix[0][1]:4d}")
    print(f"       Dog  {conf_matrix[1][0]:4d}  {conf_matrix[1][1]:4d}")
    
    # Additional statistics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Correctly classified examples
    correct_mask = np.array(all_preds) == all_labels
    correct_probs = all_probs[correct_mask]
    incorrect_probs = all_probs[~correct_mask]
    
    print("\nPrediction Confidence:")
    print(f"Average confidence (correct):   {np.mean(np.abs(correct_probs - 0.5) + 0.5):.4f}")
    if len(incorrect_probs) > 0:
        print(f"Average confidence (incorrect): {np.mean(np.abs(incorrect_probs - 0.5) + 0.5):.4f}")
    
    # Per-class accuracy
    cat_mask = all_labels == 0
    dog_mask = all_labels == 1
    cat_accuracy = accuracy_score(all_labels[cat_mask], np.array(all_preds)[cat_mask])
    dog_accuracy = accuracy_score(all_labels[dog_mask], np.array(all_preds)[dog_mask])
    
    print("\nPer-Class Accuracy:")
    print(f"Cat accuracy: {cat_accuracy:.4f} ({cat_accuracy*100:.2f}%)")
    print(f"Dog accuracy: {dog_accuracy:.4f} ({dog_accuracy*100:.2f}%)")
    print("="*50)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate trained model')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Path to model file')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    validate_model(args.model, args.config)
