import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from dataset import prepare_dataloaders
from model import EHRTransformer
from tqdm import tqdm

# Flash Attention Optimization
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True) 

def set_seed(seed=42):
    """Ensures 100% reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.register_buffer('pos_weight', pos_weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.sigmoid(inputs)
        pt_true = pt * targets + (1 - pt) * (1 - targets)
        modulating_factor = (1.0 - pt_true) ** self.gamma
        focal_loss = modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss

class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def compute_metrics(all_labels, all_preds):
    """Isolates CPU-bound sklearn metrics from the training loop."""
    metrics = {'aucs': [], 'aps': [], 'f1s': [], 'briers': []}
    
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            metrics['aucs'].append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
            metrics['aps'].append(average_precision_score(all_labels[:, i], all_preds[:, i]))
            
            binary_preds = (all_preds[:, i] > 0.5).astype(int)
            metrics['f1s'].append(f1_score(all_labels[:, i], binary_preds, zero_division=0))
            metrics['briers'].append(brier_score_loss(all_labels[:, i], all_preds[:, i]))
            
    return {
        'val_auc': np.mean(metrics['aucs']) if metrics['aucs'] else 0.0,
        'val_map': np.mean(metrics['aps']) if metrics['aps'] else 0.0,
        'val_f1': np.mean(metrics['f1s']) if metrics['f1s'] else 0.0,
        'val_brier': np.mean(metrics['briers']) if metrics['briers'] else 0.0
    }

def train_model():
    set_seed(42)
    
    wandb.init(project="m31-patient-timelines", config={
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 35,
        "d_model": 128,
        "max_seq_len": 512,
        "mixed_precision": True
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, val_loader = prepare_dataloaders(batch_size=config.batch_size)
    
    # Calculate pos_weights dynamically
    print("Calculating class weights from training set...")
    all_train_labels = torch.cat([batch['labels'] for batch in train_loader], dim=0)
    pos_counts = all_train_labels.sum(dim=0)
    total_counts = all_train_labels.size(0)
    pos_weights = (total_counts - pos_counts) / torch.clamp(pos_counts, min=1.0)
    pos_weights = pos_weights.to(device)
    
    model = EHRTransformer(
        num_classes=all_train_labels.shape[1], 
        d_model=config.d_model, 
        max_seq_len=config.max_seq_len
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = FocalLoss(pos_weight=pos_weights, gamma=2.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=4, min_delta=0.001)
    
    # Initialize Gradient Scaler for Mixed Precision
    scaler = GradScaler()

    best_val_auc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = torch.nan_to_num(batch['labels'].to(device), nan=0.0)
            
            optimizer.zero_grad(set_to_none=True) # More efficient than zero_grad()
            
            # Autocast enables Automatic Mixed Precision (AMP)
            with autocast():
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = torch.nan_to_num(batch['labels'].to(device), nan=0.0)
                
                with autocast():
                    logits = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Compute Metrics
        metrics = compute_metrics(np.vstack(all_labels), np.vstack(all_preds))
        scheduler.step(metrics['val_auc'])

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val AUROC: {metrics['val_auc']:.4f} | Val mAP: {metrics['val_map']:.4f} | Val F1: {metrics['val_f1']:.4f} | Val Brier: {metrics['val_brier']:.4f}\n")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_macro_auc": metrics['val_auc'],
            "val_mean_ap": metrics['val_map'],
            "val_macro_f1": metrics['val_f1'],
            "val_brier_score": metrics['val_brier'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        if metrics['val_auc'] > best_val_auc:
            best_val_auc = metrics['val_auc']
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            print("--> Saved new best model!\n")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Halting training to prevent overfitting.")
            break

    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    train_model()