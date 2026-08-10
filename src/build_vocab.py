import pandas as pd
import os
import json

def build_vocabulary(
    input_dir='data/train_val_filtered',
    output_vocab_path='data/vocab.json'
):
    print("Scanning filtered tables to build vocabulary...")
    
    # Initialize vocabulary with special tokens
    vocab = {
        '[PAD]': 0,
        '[UNK]': 1,
        '[CLS]': 2,
        '[SEP]': 3
    }
    
    # Define time-delta tokens to encode the gaps between events
    time_tokens = [
        'TIME_SAME_DAY', 'TIME_0_7_DAYS', 'TIME_1_4_WEEKS', 
        'TIME_1_6_MONTHS', 'TIME_6_12_MONTHS', 'TIME_1_5_YEARS', 'TIME_5+_YEARS'
    ]
    for tt in time_tokens:
        vocab[tt] = len(vocab)
        
    # Process Conditions
    cond_path = os.path.join(input_dir, 'conditions.csv')
    if os.path.exists(cond_path):
        cond_df = pd.read_csv(cond_path, low_memory=False)
        unique_conds = cond_df['CODE'].dropna().unique()
        for code in unique_conds:
            token = f"COND_{code}"
            if token not in vocab:
                vocab[token] = len(vocab)
                
    # Process Encounters
    enc_path = os.path.join(input_dir, 'encounters.csv')
    if os.path.exists(enc_path):
        enc_df = pd.read_csv(enc_path, low_memory=False)
        unique_encs = enc_df['ENCOUNTERCLASS'].dropna().unique()
        for cls in unique_encs:
            token = f"ENC_{cls}"
            if token not in vocab:
                vocab[token] = len(vocab)

    # Process Allergies
    alg_path = os.path.join(input_dir, 'allergies.csv')
    if os.path.exists(alg_path):
        alg_df = pd.read_csv(alg_path, low_memory=False)
        unique_algs = alg_df['CODE'].dropna().unique()
        for code in unique_algs:
            token = f"ALG_{code}"
            if token not in vocab:
                vocab[token] = len(vocab)
                
    # Process Careplans
    care_path = os.path.join(input_dir, 'careplans.csv')
    if os.path.exists(care_path):
        care_df = pd.read_csv(care_path, low_memory=False)
        unique_cares = care_df['CODE'].dropna().unique()
        for code in unique_cares:
            token = f"CARE_{code}"
            if token not in vocab:
                vocab[token] = len(vocab)

    # Save the mapping to a JSON file
    with open(output_vocab_path, 'w') as f:
        json.dump(vocab, f, indent=4)
        
    print(f"Vocabulary built successfully! Total unique tokens: {len(vocab)}")
    print(f"Saved to: {output_vocab_path}")

if __name__ == "__main__":
    build_vocabulary()