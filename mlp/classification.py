import os
import glob
import random
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from library_functions import Load_Data_With_Labels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f" Seed set to {seed}")

set_seed(42)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.avgpool_out = None

    def forward(self, x):
        x = self.features(x)
        self.avgpool_out = self.avgpool(x)
        x = self.avgpool_out
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for batch in loader:
        imgs, labels = batch
        
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples

def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch
            
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def plot_history(history, output_path="training_history.png"):
    plt.figure(figsize=(12, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend(); plt.grid(True)
    
    plt.savefig(output_path)
    plt.close()

def extract_features(model, loader, device):
    model.eval()
    feats_list, lbls_list = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            _ = model(imgs)
            feats = model.avgpool_out.view(imgs.size(0), -1).cpu().numpy()
            feats_list.append(feats)
            lbls_list.append(labels.numpy())
    return np.concatenate(feats_list), np.concatenate(lbls_list)


def plot_pca(features, labels, classes, output_prefix="pca"):
    if len(features) == 0: return
    
    pca2 = PCA(n_components=2)
    f2d = pca2.fit_transform(features)
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        idx = labels == i
        plt.scatter(f2d[idx, 0], f2d[idx, 1], label=cls, alpha=0.6)
    plt.legend(); plt.title("PCA 2D"); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_prefix}_2d.png"); plt.close()
    
    pca3 = PCA(n_components=3)
    f3d = pca3.fit_transform(features)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, cls in enumerate(classes):
        idx = labels == i
        ax.scatter(f3d[idx, 0], f3d[idx, 1], f3d[idx, 2], label=cls, alpha=0.6)
    ax.legend(); ax.set_title("PCA 3D")
    plt.savefig(f"{output_prefix}_3d.png"); plt.close()


def load_and_preprocess_data(data_dir, labels_csv):
    """
    Load data using Load_Data_With_Labels and process it
    """
    print(f"\n{'='*70}")
    print("LOADING AND PREPROCESSING DATA")
    print(f"{'='*70}")
    
    # Load data with labels using new function (ensures alignment)
    print("\n1. Loading data with labels using Load_Data_With_Labels...")
    X, y_labels, matched_filenames = Load_Data_With_Labels(data_dir, labels_csv)
    print(f"   [OK] Loaded shape: {X.shape}")
    print(f"   [OK] Number of files: {len(matched_filenames)}")
    
    # CRITICAL VERIFICATION
    if len(X) != len(y_labels) or len(X) != len(matched_filenames):
        print(f"   CRITICAL ERROR: Alignment mismatch during load!")
        raise ValueError("Data and labels not aligned after Load_Data_With_Labels")
    print(f"   VERIFIED: X, y_labels, and filenames are aligned")
    
    # Extract class labels from loaded data
    print("\n2. Processing class labels...")
    
    # Load the CSV to get shape information
    labels_df = pd.read_csv(labels_csv)
    
    # Create mapping from filename to shape class
    filename_to_shape = {}
    for idx, row in labels_df.iterrows():
        try:
            filename = str(row['filename']).strip()
            label = str(row['shape']).strip()
            
            # Check if there's a type column (for Step_R and Step_T distinction)
            if 'type' in labels_df.columns and label == 'Step':
                type_val = str(row['type']).strip()
                if type_val and type_val.lower() != 'nan':
                    label = f'Step_{type_val}'  # e.g., "Step_R" or "Step_T"
            
            if label and label.lower() != 'nan':
                filename_to_shape[filename] = label
        except (ValueError, KeyError):
            continue
    
    # Extract class labels from matched filenames
    y = []
    for filename in matched_filenames:
        if filename in filename_to_shape:
            y.append(filename_to_shape[filename])
        else:
            print(f"   [WARN] {filename} not found in shape mapping")
            y.append('Unknown')
    
    # Create class mapping
    unique_classes = sorted(list(set(y)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    y_encoded = np.array([class_to_idx[label] for label in y])
    
    print(f"   [OK] Total samples: {len(X)} files")
    print(f"   [OK] Classes: {unique_classes}")
    
    # Upscale data to handle STD imbalance
    print("\n3. Upscaling data...")
    x_both_data = X
    a, b, c = x_both_data.shape
    
    # Group by class for STD calculation
    classes = np.array(y)
    unique_shape_classes = unique_classes
    class_data = {cls: [] for cls in unique_shape_classes}
    
    for idx, cls in enumerate(classes):
        class_data[cls].append(x_both_data[idx])
    for cls in unique_shape_classes:
        class_data[cls] = np.array(class_data[cls])
    
    stds = {cls: np.std(class_data[cls]) for cls in unique_shape_classes}
    print(f"   STD từng shape class:")
    for cls in unique_shape_classes:
        print(f"     {cls}: {stds[cls]:.6f}")
    
    # Find class with smallest STD and scale it up
    small_cls = min(stds, key=stds.get)
    small_std = stds[small_cls]
    target_std = np.mean([stds[c] for c in unique_shape_classes if c != small_cls])
    
    if target_std > 0 and small_std > 0:
        scale_factor = target_std / small_std
    else:
        scale_factor = 1.0
    
    print(f"   Scale factor: {scale_factor:.6f}")
    
    X_upscaled = np.copy(x_both_data)
    for idx, cls in enumerate(classes):
        if cls == small_cls:
            X_upscaled[idx] *= scale_factor
    
    print(f"   [OK] Upscaled shape: {X_upscaled.shape}")
    
    return X_upscaled, y_encoded, unique_classes, class_to_idx, a, b, c


def split_and_normalize_data(X, y, test_size=0.2, val_size=0.1):
    """
    Split data into train/val/test and normalize using StandardScaler
    """
    print(f"\n{'='*70}")
    print("SPLITTING AND NORMALIZING DATA (70/10/20)")
    print(f"{'='*70}")
    
    n_total = len(X)
    a, b, c = X.shape
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Second split: split 80% into 87.5% train (70% of total), 12.5% val (10% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
    )
    
    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    
    print(f"\nTotal samples:    {n_total}")
    print(f"Train samples:    {n_train} ({100*n_train/n_total:.1f}%)")
    print(f"Val samples:      {n_val}   ({100*n_val/n_total:.1f}%)")
    print(f"Test samples:     {n_test}  ({100*n_test/n_total:.1f}%)")
    
    # Normalize using StandardScaler fit on train set only
    print(f"\nFitting StandardScaler on training data...")
    train_flat = X_train.reshape(n_train * b, c)
    X_scaler = StandardScaler()
    X_train_flat_norm = X_scaler.fit_transform(train_flat)
    
    # Transform val and test using training scaler
    val_flat = X_val.reshape(n_val * b, c)
    X_val_flat_norm = X_scaler.transform(val_flat)
    
    test_flat = X_test.reshape(n_test * b, c)
    X_test_flat_norm = X_scaler.transform(test_flat)
    
    # Reshape back
    X_train_norm = X_train_flat_norm.reshape(n_train, b, c)
    X_val_norm = X_val_flat_norm.reshape(n_val, b, c)
    X_test_norm = X_test_flat_norm.reshape(n_test, b, c)
    
    print(f"[OK] Scaler Mean: {X_scaler.mean_[0]:.6f}")
    print(f"[OK] Scaler Std: {X_scaler.scale_[0]:.6f}")
    
    return (X_train_norm, X_val_norm, X_test_norm), (y_train, y_val, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data_Dipole_Model', help='Path to data directory')
    parser.add_argument('--labels_csv', type=str, default='dataset_labels.csv', help='Path to labels CSV')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Path to save model checkpoints')
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"[CHECKPOINT] Directory: {args.checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Load and preprocess data
    X, y, classes, class_to_idx, a, b, c = load_and_preprocess_data(args.data_dir, args.labels_csv)
    
    # Split and normalize data
    (X_train_norm, X_val_norm, X_test_norm), (y_train, y_val, y_test) = split_and_normalize_data(X, y)
    
    print(f"\nCreating datasets...")
    # Reshape data for Conv2D: (N, H, W) -> (N, 1, H, W)
    X_train_input = torch.from_numpy(X_train_norm.astype(np.float32)).unsqueeze(1)
    X_val_input = torch.from_numpy(X_val_norm.astype(np.float32)).unsqueeze(1)
    X_test_input = torch.from_numpy(X_test_norm.astype(np.float32)).unsqueeze(1)
    
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64))
    y_test_tensor = torch.from_numpy(y_test.astype(np.int64))
    
    train_ds = torch.utils.data.TensorDataset(X_train_input, y_train_tensor)
    val_ds = torch.utils.data.TensorDataset(X_val_input, y_val_tensor)
    test_ds = torch.utils.data.TensorDataset(X_test_input, y_test_tensor)
    
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(val_ds)}")
    print(f"   Test samples:  {len(test_ds)}")
    print(f"   Classes: {classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    patience_cnt = 0
    
    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        
        print(f"[Epoch {epoch+1:03d}/{args.num_epochs}] "
              f"T_Loss: {train_loss:.4f} T_Acc: {train_acc:.4f} | "
              f"V_Loss: {val_loss:.4f} V_Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            print(f"   Loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            patience_cnt = 0
            model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"   Model saved to: {model_path}")
        else:
            patience_cnt += 1
            print(f"   No improvement. Patience: {patience_cnt}/{args.early_stop}")
            if patience_cnt >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1} (Val Loss did not improve)")
                break

    print("\nEvaluating on TEST set...")
    test_loss, test_acc = evaluate_one_epoch(model, test_loader, criterion, device)
    print(f"   Final Test Acc: {test_acc:.4f} (Loss: {test_loss:.4f})")
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))
    
    plot_history(history, output_path=os.path.join(args.checkpoint_dir, "training_history.png"))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, "confusion_matrix.png")); plt.close()

    print("Generating PCA...")
    feats, lbls = extract_features(model, test_loader, device)
    plot_pca(feats, lbls, classes, output_prefix=os.path.join(args.checkpoint_dir, "pca"))
    print("Done!")

if __name__ == "__main__":
    main()
