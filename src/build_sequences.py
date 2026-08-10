import pandas as pd
import os
import json
from datetime import timedelta

def get_time_token(delta_days):
    if delta_days == 0: return 'TIME_SAME_DAY'
    elif delta_days <= 7: return 'TIME_0_7_DAYS'
    elif delta_days <= 28: return 'TIME_1_4_WEEKS'
    elif delta_days <= 180: return 'TIME_1_6_MONTHS'
    elif delta_days <= 365: return 'TIME_6_12_MONTHS'
    elif delta_days <= 1825: return 'TIME_1_5_YEARS'
    else: return 'TIME_5+_YEARS'

def build_patient_sequences(
    input_dir='data/train_val_filtered',
    vocab_path='data/vocab.json',
    output_path='data/train_val_sequences.json'
):
    print("Loading vocabulary...")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        
    print("Aggregating clinical events...")
    events_list = []
    
    # 1. Load Conditions
    if os.path.exists(os.path.join(input_dir, 'conditions.csv')):
        df = pd.read_csv(os.path.join(input_dir, 'conditions.csv'), low_memory=False)
        df = df[['PATIENT', 'START', 'CODE']].dropna()
        df['TOKEN'] = 'COND_' + df['CODE'].astype(str)
        events_list.append(df[['PATIENT', 'START', 'TOKEN']])
        
    # 2. Load Encounters
    if os.path.exists(os.path.join(input_dir, 'encounters.csv')):
        df = pd.read_csv(os.path.join(input_dir, 'encounters.csv'), low_memory=False)
        df = df[['PATIENT', 'START', 'ENCOUNTERCLASS']].dropna()
        df['TOKEN'] = 'ENC_' + df['ENCOUNTERCLASS'].astype(str)
        events_list.append(df[['PATIENT', 'START', 'TOKEN']])

    # 3. Load Allergies
    if os.path.exists(os.path.join(input_dir, 'allergies.csv')):
        df = pd.read_csv(os.path.join(input_dir, 'allergies.csv'), low_memory=False)
        df = df[['PATIENT', 'START', 'CODE']].dropna()
        df['TOKEN'] = 'ALG_' + df['CODE'].astype(str)
        events_list.append(df[['PATIENT', 'START', 'TOKEN']])

    # 4. Load Careplans
    if os.path.exists(os.path.join(input_dir, 'careplans.csv')):
        df = pd.read_csv(os.path.join(input_dir, 'careplans.csv'), low_memory=False)
        df = df[['PATIENT', 'START', 'CODE']].dropna()
        df['TOKEN'] = 'CARE_' + df['CODE'].astype(str)
        events_list.append(df[['PATIENT', 'START', 'TOKEN']])

    # Combine and sort all events chronologically
    print("Sorting timelines...")
    all_events = pd.concat(events_list, ignore_index=True)
    all_events['START'] = pd.to_datetime(all_events['START'], utc=True)
    all_events.sort_values(by=['PATIENT', 'START'], inplace=True)
    
    print("Generating token ID sequences...")
    patient_sequences = {}
    
    # Group by patient to build individual timelines
    for patient_id, group in all_events.groupby('PATIENT'):
        seq = [vocab['[CLS]']] # Start sequence token
        
        last_date = None
        for _, row in group.iterrows():
            current_date = row['START']
            token_str = row['TOKEN']
            
            # Calculate time difference and append time token
            if last_date is not None:
                delta_days = (current_date - last_date).days
                # Safeguard against negative deltas due to sorting quirks
                delta_days = max(0, delta_days) 
                time_token = get_time_token(delta_days)
                seq.append(vocab.get(time_token, vocab['[UNK]']))
            
            # Append the actual event token
            seq.append(vocab.get(token_str, vocab['[UNK]']))
            last_date = current_date
            
        seq.append(vocab['[SEP]']) # End sequence token
        patient_sequences[patient_id] = seq

    print(f"Saving sequences for {len(patient_sequences)} patients...")
    with open(output_path, 'w') as f:
        json.dump(patient_sequences, f)
        
    print(f"Done! Sequences saved to: {output_path}")

if __name__ == "__main__":
    build_patient_sequences()