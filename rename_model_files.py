import os
import shutil
#from pathlib import Path

def rename_model_files():
    """Rename model files to match expected filenames."""
    # Define the model file mappings (current_path, new_name)
    model_mappings = [
        ("models_daisee/daisee_emotional_model_best.pth", "daisee_model.pth"),
        
    ]
    
    for current_path, new_name in model_mappings:
        current_path = Path(current_path)
        new_path = current_path.parent / new_name
        
        # Skip if source doesn't exist
        if not current_path.exists():
            print(f"⚠️  Source file not found: {current_path}")
            continue
            
        # Skip if destination already exists
        if new_path.exists():
            print(f"ℹ️  Destination already exists: {new_path}")
            continue
            
        try:
            # Create a copy of the file with the new name
            shutil.copy2(current_path, new_path)
            #print(f"Created copy: {current_path} -> {new_path}")
        except Exception as e:
            print(f"Failed to create copy {new_path}: {e}")

if __name__ == "__main__":
    print("Renaming model files to match expected filenames...")
    rename_model_files()
