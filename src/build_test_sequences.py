import pandas as pd
import os
import json
from build_sequences import get_time_token  # reuse the same time-bucketing logic


def build_test_sequences(
    input_dir='data/test',
    vocab_path='data/vocab.json',
    output_path='data/test_sequences.json'
):
    """
    Mirrors build_sequences.py but for the test split.

    Important: test/ tables are already truncated at each patient's anchor
    (per the README), so unlike train_val we do NOT run filter_timeline.py
    on them - doing so again would be a no-op at best, and wrong at worst if
    it tried to recompute anchors from data that's already been cut off.
    We also reuse vocab.json exactly as built from train/val: any token seen
    only in test falls back to [UNK], which is correct - the vocab must not
    be rebuilt/extended from test data, or that would leak test information
    into the model's input representation.
    """
    print("Loading vocabulary (built from train/val only)...")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    print("Aggregating clinical events for test patients...")
    events_list = []

    table_prefix_map = {
        'conditions.csv': ('CODE', 'COND_'),
        'encounters.csv': ('ENCOUNTERCLASS', 'ENC_'),
        'allergies.csv': ('CODE', 'ALG_'),
        'careplans.csv': ('CODE', 'CARE_'),
    }

    for table_name, (field, prefix) in table_prefix_map.items():
        path = os.path.join(input_dir, table_name)
        if not os.path.exists(path):
            print(f"Skipping {table_name} - file not found.")
            continue
        df = pd.read_csv(path, low_memory=False)
        df = df[['PATIENT', 'START', field]].dropna()
        df['TOKEN'] = prefix + df[field].astype(str)
        events_list.append(df[['PATIENT', 'START', 'TOKEN']])

    if not events_list:
        raise FileNotFoundError(f"No recognized tables found under {input_dir}")

    all_events = pd.concat(events_list, ignore_index=True)
    all_events['START'] = pd.to_datetime(all_events['START'], utc=True, format='mixed')
    all_events.sort_values(by=['PATIENT', 'START'], inplace=True)

    print("Generating token ID sequences...")
    patient_sequences = {}

    for patient_id, group in all_events.groupby('PATIENT'):
        seq = [vocab['[CLS]']]
        last_date = None
        for _, row in group.iterrows():
            current_date = row['START']
            token_str = row['TOKEN']
            if last_date is not None:
                delta_days = max(0, (current_date - last_date).days)
                time_token = get_time_token(delta_days)
                seq.append(vocab.get(time_token, vocab['[UNK]']))
            seq.append(vocab.get(token_str, vocab['[UNK]']))
            last_date = current_date
        seq.append(vocab['[SEP]'])
        patient_sequences[patient_id] = seq

    # Make sure every test patient gets an entry, even those with zero
    # pre-anchor events - dataset.py's EHRTimelineDataset already falls back
    # to a bare [CLS] token for missing patients, but writing them explicitly
    # here means predictions.csv can't silently drop anyone.
    test_anchors = pd.read_csv('data/test_anchors.csv')
    id_col = 'Id' if 'Id' in test_anchors.columns else test_anchors.columns[0]
    for pid in test_anchors[id_col].astype(str):
        if pid not in patient_sequences:
            patient_sequences[pid] = [vocab['[CLS]'], vocab['[SEP]']]

    print(f"Saving sequences for {len(patient_sequences)} test patients...")
    with open(output_path, 'w') as f:
        json.dump(patient_sequences, f)

    print(f"Done! Test sequences saved to: {output_path}")


if __name__ == "__main__":
    build_test_sequences()
