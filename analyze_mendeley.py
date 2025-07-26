import pandas as pd
import os

def analyze_dataset(filepath):
    try:
        # Read the dataset
        df = pd.read_csv(filepath)
        
        # Create analysis results
        analysis = {
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'head': df.head().to_dict('records'),
            'describe': df.describe().to_dict()
        }
        
        # Save analysis to file
        with open('mendeley_analysis.txt', 'w') as f:
            f.write("=== Dataset Analysis ===\n\n")
            f.write(f"File: {filepath}\n")
            f.write(f"Shape: {analysis['shape']} (rows, columns)\n\n")
            
            f.write("=== Columns ===\n")
            for col in analysis['columns']:
                f.write(f"- {col}: {analysis['dtypes'][col]}\n")
            
            f.write("\n=== Missing Values ===\n")
            for col, count in analysis['missing_values'].items():
                if count > 0:
                    f.write(f"- {col}: {count} missing values\n")
            
            f.write("\n=== First 5 Rows ===\n")
            for row in analysis['head']:
                f.write(str(row) + "\n")
            
            f.write("\n=== Numeric Statistics ===\n")
            for col, stats in analysis['describe'].items():
                if pd.api.types.is_numeric_dtype(df[col]):
                    f.write(f"\n--- {col} ---\n")
                    for stat, value in stats.items():
                        f.write(f"{stat}: {value}\n")
        
        print("Analysis complete. Check 'mendeley_analysis.txt' for details.")
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    dataset_path = r"backend\datasets\mendeley_attention\attention_detection_dataset_v2.csv"
    analyze_dataset(dataset_path)
