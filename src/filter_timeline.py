import pandas as pd
import os

def filter_events(
    anchors_file='data/processed_anchors.csv',
    input_dir='data/train_val',
    output_dir='data/train_val_filtered'
):
    print("Loading anchors...")
    anchors = pd.read_csv(anchors_file)
    anchors['ANCHOR_DATE'] = pd.to_datetime(anchors['ANCHOR_DATE'], utc=True)
    
    # Rename Id to PATIENT to easily match the clinical tables
    anchors.rename(columns={'Id': 'PATIENT'}, inplace=True)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # The core tables we need to process
    tables = ['allergies.csv', 'careplans.csv', 'conditions.csv', 'devices.csv', 'encounters.csv']
    
    for table_name in tables:
        file_path = os.path.join(input_dir, table_name)
        if not os.path.exists(file_path):
            print(f"Skipping {table_name} - file not found.")
            continue
            
        print(f"Filtering {table_name}...")
        df = pd.read_csv(file_path, low_memory=False)
        
        # Merge the table with our anchor dates
        merged = pd.merge(df, anchors[['PATIENT', 'ANCHOR_DATE']], on='PATIENT', how='inner')
        
        # Convert START dates for accurate comparison
        merged['START'] = pd.to_datetime(merged['START'], utc=True)
        
        # STRICT FILTER: The event MUST start strictly before the anchor date
        filtered = merged[merged['START'] < merged['ANCHOR_DATE']].copy()
        
        # Remove the helper column so the table format remains pristine
        filtered.drop(columns=['ANCHOR_DATE'], inplace=True)
        
        # Save the safe, truncated timeline
        out_path = os.path.join(output_dir, table_name)
        filtered.to_csv(out_path, index=False)
        
        print(f"  -> Saved {len(filtered)} pre-anchor events (dropped {len(df) - len(filtered)} future events).")
        
    print(f"\nSuccess! Filtered tables are ready in the '{output_dir}' folder.")

if __name__ == "__main__":
    filter_events()