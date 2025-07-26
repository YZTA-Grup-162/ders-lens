import os
import shutil
#from pathlib import Path

def create_model_links():
    model_mappings = [
        ("models_daisee/daisee_emotional_model_best.pth", "models_daisee/daisee_model.pth"),
    ]
    
    for src, dst in model_mappings:
        src_path = Path(src)
        dst_path = Path(dst)
        
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
            print(f"Failed to create symlink {dst_path}: {e}")

if __name__ == "__main__":
    print("Creating model symbolic links...")
    create_model_links()
