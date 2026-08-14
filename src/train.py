import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from dataset import prepare_dataloaders
from model import EHRTransformer
from tqdm import tqdm

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True) 

def train_model():
    wandb.init(project="m31-patient-timelines", config={
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 15,
        "d_model": 128,
        "max_seq_len": 512
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, val_loader = prepare_dataloaders(batch_size=config.batch_size)
    
    sample_batch = next(iter(train_loader))
    num_classes = sample_batch['labels'].shape[1]
    
    print("Calculating class weights from training set...")
    all_train_labels = torch.cat([batch['labels'] for batch in train_loader], dim=0)
    pos_counts = all_train_labels.sum(dim=0)
    total_counts = all_train_labels.size(0)
    pos_weights = (total_counts - pos_counts) / torch.clamp(pos_counts, min=1.0)
    pos_weights = pos_weights.to(device)
    
    model = EHRTransformer(num_classes=num_classes, d_model=config.d_model, max_seq_len=config.max_seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_auc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = torch.nan_to_num(batch['labels'].to(device), nan=0.0)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = torch.nan_to_num(batch['labels'].to(device), nan=0.0)
                
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        valid_aucs, valid_aps, valid_f1s, valid_briers = [], [], [], []
        
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                valid_aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
                valid_aps.append(average_precision_score(all_labels[:, i], all_preds[:, i]))
                
                binary_preds = (all_preds[:, i] > 0.5).astype(int)
                valid_f1s.append(f1_score(all_labels[:, i], binary_preds, zero_division=0))
                valid_briers.append(brier_score_loss(all_labels[:, i], all_preds[:, i]))
                
        val_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        val_map = np.mean(valid_aps) if valid_aps else 0.0
        val_f1 = np.mean(valid_f1s) if valid_f1s else 0.0
        val_brier = np.mean(valid_briers) if valid_briers else 0.0

        scheduler.step(val_auc)

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val AUROC: {val_auc:.4f} | Val mAP: {val_map:.4f} | Val F1: {val_f1:.4f} | Val Brier: {val_brier:.4f}\n")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_macro_auc": val_auc,
            "val_mean_ap": val_map,
            "val_macro_f1": val_f1,
            "val_brier_score": val_brier,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            print("--> Saved new best model!\n")

    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    train_model()