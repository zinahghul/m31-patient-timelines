import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

class EHRTimelineDataset(Dataset):
    def __init__(self, sequences, labels_df, max_seq_len=512, pad_token_id=0):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Ensure patient_id is treated as a string to match the JSON keys
        self.patient_ids = labels_df['patient_id'].astype(str).tolist()
        self.labels = labels_df.drop(columns=['patient_id']).values
        
        # Retrieve sequences for valid patients (default to empty list if missing)
        self.sequences = [sequences.get(pid, []) for pid in self.patient_ids]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # FIX: Give patients with no history a [CLS] token to prevent NaN division
        if len(seq) == 0:
            seq = [2] 

        # Truncate sequences that exceed the maximum length.
        if len(seq) > self.max_seq_len:
            # Keep the [CLS] token at index 0, then take the most recent events
            seq = [seq[0]] + seq[-(self.max_seq_len - 1):]

        # Pad sequences that are shorter than the maximum length
        pad_len = self.max_seq_len - len(seq)
        if pad_len > 0:
            seq = seq + [self.pad_token_id] * pad_len
            mask = [1] * (self.max_seq_len - pad_len) + [0] * pad_len
        else:
            mask = [1] * self.max_seq_len

        return {
            'patient_id': self.patient_ids[idx],
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def prepare_dataloaders(
    seq_path='data/train_val_sequences.json',
    labels_path='data/train_val_targets.csv',
    splits_path='data/patient_splits.csv',
    batch_size=32,
    max_seq_len=512
):
    print("Loading data for PyTorch...")
    with open(seq_path, 'r') as f:
        sequences = json.load(f)
        
    labels_df = pd.read_csv(labels_path)
    splits_df = pd.read_csv(splits_path)
    
    # Merge splits to separate train and val cohorts
    merged = pd.merge(labels_df, splits_df, left_on='patient_id', right_on='Id', how='inner')
    
    train_df = merged[merged['split'] == 'train'].drop(columns=['Id', 'split']).reset_index(drop=True)
    val_df = merged[merged['split'] == 'val'].drop(columns=['Id', 'split']).reset_index(drop=True)
    
    print(f"Train patients: {len(train_df)} | Validation patients: {len(val_df)}")
    
    train_dataset = EHRTimelineDataset(sequences, train_df, max_seq_len)
    val_dataset = EHRTimelineDataset(sequences, val_df, max_seq_len)
    
    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = prepare_dataloaders()
    
    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print("Dataset setup successful!")