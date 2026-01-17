import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix

ImageFile.LOAD_TRUNCATED_IMAGES = True

class WildfireConfig:
    SEED = 42
    DATA_PATH = "wildfire"
    OUTPUT_DIR = "output_results"
    MODEL_NAME = "convnext_tiny"
    
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.002
    PATIENCE = 8
    NUM_WORKERS = 4
    
    IDX_TO_CLASS = {0: "nowildfire", 1: "wildfire"}
    CLASS_TO_IDX = {"nowildfire": 0, "wildfire": 1}
    
    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class WildfireDataset(Dataset):
    def __init__(self, root_dir, stage, transform=None):
        self.root_dir = root_dir
        self.stage = stage
        self.transform = transform
        self.config = WildfireConfig
        
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for class_name, class_idx in self.config.CLASS_TO_IDX.items():
            folder = os.path.join(self.root_dir, self.stage, class_name)
            if not os.path.exists(folder):
                continue
                
            images = sorted(os.listdir(folder))
            for img_name in images:
                self.image_paths.append(os.path.join(folder, img_name))
                self.labels.append(class_idx)
        
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
            
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        
        metrics = MetricCollection({
            'acc': Accuracy(task="binary"),
            'f1': F1Score(task="binary"),
            'auc': AUROC(task="binary")
        }).to(device)
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        self.train_metrics.reset()
        
        pbar = tqdm(loader, leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
            preds = torch.sigmoid(outputs)
            self.train_metrics.update(preds, labels.long())
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = total_loss / len(loader.dataset)
        epoch_metrics = self.train_metrics.compute()
        
        return epoch_loss, epoch_metrics

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        self.val_metrics.reset()
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                
                preds = torch.sigmoid(outputs)
                self.val_metrics.update(preds, labels.long())
        
        epoch_loss = total_loss / len(loader.dataset)
        epoch_metrics = self.val_metrics.compute()
        
        return epoch_loss, epoch_metrics

    def fit(self, train_loader, val_loader, epochs, patience, save_path):
        best_val_f1 = 0.0
        early_stop_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_mets = self.train_epoch(train_loader)
            val_loss, val_mets = self.validate_epoch(val_loader)
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_mets['train_acc'].item())
            self.history['val_acc'].append(val_mets['val_acc'].item())
            self.history['train_f1'].append(train_mets['train_f1'].item())
            self.history['val_f1'].append(val_mets['val_f1'].item())

            print(f"Train Loss: {train_loss:.4f} | Acc: {train_mets['train_acc']:.4f} | F1: {train_mets['train_f1']:.4f}")
            print(f"Valid Loss: {val_loss:.4f} | Acc: {val_mets['val_acc']:.4f} | F1: {val_mets['val_f1']:.4f}")
            
            current_f1 = val_mets['val_f1'].item()
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_model.pth"))
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                break
                
        return self.history

    def evaluate(self, test_loader, load_path=None):
        if load_path:
            self.model.load_state_dict(torch.load(load_path))
            
        self.model.eval()
        
        test_metrics = MetricCollection({
            'Accuracy': Accuracy(task="binary"),
            'Precision': Precision(task="binary"),
            'Recall': Recall(task="binary"),
            'F1': F1Score(task="binary"),
            'AUROC': AUROC(task="binary"),
            'ConfMat': ConfusionMatrix(task="binary", num_classes=2)
        }).to(self.device)
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).squeeze()
                probs = torch.sigmoid(outputs)
                
                test_metrics.update(probs, labels.long())
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        results = test_metrics.compute()
        return results, np.array(all_preds), np.array(all_labels)

def plot_learning_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Treino')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validação')
    plt.title('Curva de Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Treino')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validação')
    plt.title('Curva de Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def plot_confusion_matrix(cm_tensor, class_names, save_dir):
    cm = cm_tensor.cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    cfg = WildfireConfig()
    set_seed(cfg.SEED)
    ensure_dir(cfg.OUTPUT_DIR)
    device = cfg.get_device()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = WildfireDataset(cfg.DATA_PATH, "train", transform=train_transform)
    val_ds = WildfireDataset(cfg.DATA_PATH, "valid", transform=val_test_transform)
    test_ds = WildfireDataset(cfg.DATA_PATH, "test", transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    
    model.classifier[2] = nn.Linear(in_features=768, out_features=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    trainer = Trainer(model, optimizer, criterion, scheduler, device)
    
    history = trainer.fit(
        train_loader, 
        val_loader, 
        epochs=cfg.EPOCHS, 
        patience=cfg.PATIENCE, 
        save_path=cfg.OUTPUT_DIR
    )

    plot_learning_curves(history, cfg.OUTPUT_DIR)
    
    pd.DataFrame(history).to_csv(os.path.join(cfg.OUTPUT_DIR, "training_history.csv"), index=False)

    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
    test_results, _, _ = trainer.evaluate(test_loader, load_path=best_model_path)
    
    cm = test_results.pop('ConfMat')
    plot_confusion_matrix(cm, list(cfg.CLASS_TO_IDX.keys()), cfg.OUTPUT_DIR)
    
    print("\n--- Test Results ---")
    for k, v in test_results.items():
        print(f"{k}: {v:.4f}")