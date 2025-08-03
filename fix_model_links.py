import os
import shutil
#from pathlib import Path

def create_model_links():
    # List of model mappings (source and destination)
    model_mappings = [
        ("models_daisee/daisee_emotional_model_best.pth", "models_daisee/daisee_model.pth"),
    ]
    
    # Loop through each model mapping to create symbolic links
    for src, dst in model_mappings:
        src_path = Path(src)    # Convert source string to a Path object
        dst_path = Path(dst)    # Convert destination string to a Path object

        # Check if the source file exists
        if not src_path.exists():
            print(f"Source file not found: {src_path}")
            continue
            
        # Skip if destination already exists
        if dst_path.exists():
            print(f"Destination already exists: {dst_path}")
            continue
            
        try:
            # Create a symbolic link
            dst_path.symlink_to(src_path.resolve())
            print(f"Created symlink: {dst_path} -> {src_path}")
        except OSError as e:
            # Handle errors if symlink creation fails
            print(f"Failed to create symlink {dst_path}: {e}")

if __name__ == "__main__":
    print("Creating model symbolic links...")
    create_model_links()
