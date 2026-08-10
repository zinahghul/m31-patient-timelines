import pandas as pd
import os

def process_train_val_labels(
    encounters_file='data/train_val/encounters.csv',
    conditions_file='data/train_val/conditions.csv',
    splits_file='data/patient_splits.csv',
    targets_file='data/target_conditions.csv',
    output_dir='data'
):
    # 1. Load datasets
    print("Loading structured CSV tables...")
    encounters = pd.read_csv(encounters_file)
    conditions = pd.read_csv(conditions_file)
    splits = pd.read_csv(splits_file)
    targets = pd.read_csv(targets_file)
    
    # 2. Identify Train and Validation Patients
    train_val_ids = splits[splits['split'].isin(['train', 'val'])]['Id'].unique()
    
    # 3. Compute Anchors
    print("Computing 5-year anchors for train/val patients...")
    enc_tv = encounters[encounters['PATIENT'].isin(train_val_ids)].copy()
    
    enc_tv['START'] = pd.to_datetime(enc_tv['START'], utc=True)
    enc_tv['STOP'] = pd.to_datetime(enc_tv['STOP'], utc=True)
    enc_tv['END_DATE'] = enc_tv['STOP'].fillna(enc_tv['START'])
    
    anchors_df = enc_tv.groupby('PATIENT')['END_DATE'].max().reset_index()
    anchors_df['ANCHOR_DATE'] = anchors_df['END_DATE'] - pd.DateOffset(years=5)
    anchors_df.rename(columns={'PATIENT': 'Id'}, inplace=True)
    
    # 4. Generate Target Labels
    print("Identifying newly diagnosed target conditions...")
    cond_tv = conditions[conditions['PATIENT'].isin(train_val_ids)].copy()
    
    code_col = [col for col in targets.columns if 'code' in col.lower()][0]
    target_codes = targets[code_col].unique()
    cond_targets = cond_tv[cond_tv['CODE'].isin(target_codes)].copy()
    cond_targets['START'] = pd.to_datetime(cond_targets['START'], utc=True)
    
    first_dx = cond_targets.groupby(['PATIENT', 'CODE'])['START'].min().reset_index()
    first_dx.rename(columns={'PATIENT': 'Id', 'START': 'FIRST_DX_DATE'}, inplace=True)
    
    merged = pd.merge(anchors_df, first_dx, on='Id', how='left')
    
    merged['is_positive'] = (
        (merged['FIRST_DX_DATE'] > merged['ANCHOR_DATE']) & 
        (merged['FIRST_DX_DATE'] <= merged['END_DATE'])
    )
    
    positives = merged[merged['is_positive'] == True]
    
    labels_df = pd.DataFrame({'patient_id': anchors_df['Id']})
    for code in target_codes:
        labels_df[str(code)] = 0
        
    labels_df.set_index('patient_id', inplace=True)
    
    for _, row in positives.iterrows():
        pid = row['Id']
        code = str(row['CODE'])
        if pid in labels_df.index:
            labels_df.at[pid, code] = 1
            
    labels_df.reset_index(inplace=True)
    
    # 5. Save the processed files
    anchors_path = os.path.join(output_dir, 'processed_anchors.csv')
    labels_path = os.path.join(output_dir, 'train_val_targets.csv')
    
    anchors_df[['Id', 'ANCHOR_DATE']].to_csv(anchors_path, index=False)
    labels_df.to_csv(labels_path, index=False)
    
    print(f"Done! Files saved to:\n- {anchors_path}\n- {labels_path}")

if __name__ == "__main__":
    process_train_val_labels()