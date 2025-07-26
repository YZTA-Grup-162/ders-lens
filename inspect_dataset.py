import pandas as pd
import os

# Path to the dataset
dataset_path = 'backend/datasets/mendeley_attention/attention_detection_dataset_v2.csv'

# Check if file exists
if not os.path.exists(dataset_path):
    print(f"Error: File not found at {dataset_path}")
    exit(1)

try:
    # Read the first few rows to understand the structure
    df = pd.read_csv(dataset_path, nrows=10)
    
    # Save the output to a file
    with open('dataset_inspection.txt', 'w', encoding='utf-8') as f:
        f.write("=== Dataset Columns ===\n")
        f.write("\n".join(df.columns) + "\n\n")
        
        f.write("=== First 5 Rows ===\n")
        f.write(df.head().to_string() + "\n\n")
        
        f.write("=== Basic Statistics ===\n")
        f.write(df.describe().to_string() + "\n")
    
    print("Dataset inspection complete. Check 'dataset_inspection.txt' for details.")
    
except Exception as e:
    print(f"Error analyzing dataset: {str(e)}")
