import json
import torch
import pandas as pd
from model import EHRTransformer
from dataset import EHRTimelineDataset
from torch.utils.data import DataLoader


def generate_predictions(
    test_seq_path='data/test_sequences.json',
    test_anchors_path='data/test_anchors.csv',
    targets_path='data/target_conditions.csv',
    checkpoint_path='checkpoints/best_model.pt',
    output_path='predictions.csv',
    max_seq_len=512,
    d_model=128,
    batch_size=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading test sequences and target codes...")
    with open(test_seq_path, 'r') as f:
        sequences = json.load(f)

    targets = pd.read_csv(targets_path)
    code_col = [c for c in targets.columns if 'code' in c.lower()][0]
    target_codes = targets[code_col].astype(str).tolist()

    anchors = pd.read_csv(test_anchors_path)
    id_col = 'Id' if 'Id' in anchors.columns else anchors.columns[0]
    patient_ids = anchors[id_col].astype(str).tolist()

    # Dummy all-zero labels just so we can reuse EHRTimelineDataset's
    # padding/truncation logic - they are never used for anything.
    dummy_labels = pd.DataFrame({'patient_id': patient_ids})
    for code in target_codes:
        dummy_labels[code] = 0

    dataset = EHRTimelineDataset(sequences, dummy_labels, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Loading trained model...")
    model = EHRTransformer(
        d_model=d_model, max_seq_len=max_seq_len, num_classes=len(target_codes)
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_ids, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_ids.extend(batch['patient_id'])
            all_probs.append(probs)

    import numpy as np
    all_probs = np.vstack(all_probs)

    out = pd.DataFrame(all_probs, columns=target_codes)
    out.insert(0, 'patient_id', all_ids)

    # Every test patient we were given must be scored.
    missing = set(patient_ids) - set(out['patient_id'])
    assert not missing, f"Missing predictions for {len(missing)} test patients: {list(missing)[:5]}"

    out.to_csv(output_path, index=False)
    print(f"Saved predictions for {len(out)} test patients to {output_path}")


if __name__ == "__main__":
    generate_predictions()
