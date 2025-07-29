import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

print("=" * 60)
print("ONNX to PyTorch Model Conversion Tool")
print("=" * 60)

def convert_onnx_to_torch(onnx_path, output_path):
    """Convert an ONNX model to PyTorch format."""
    print(f"\nConverting {onnx_path} to PyTorch format...")
    
    # Verify ONNX model first
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
    
    # Get input/output names and shapes
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    
    # Create a wrapper class for the ONNX model
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, onnx_path):
            super().__init__()
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
        def forward(self, x):
            # Convert PyTorch tensor to numpy array
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
                
            # Run inference
            outputs = self.session.run(None, {input_name: x})
            
            # Convert back to PyTorch tensor
            if len(outputs) == 1:
                return torch.from_numpy(outputs[0])
            return [torch.from_numpy(out) for out in outputs]
    
    # Create and save the wrapper model
    print("Creating PyTorch wrapper...")
    model = ONNXWrapper(onnx_path)
    
    # Test the model with a dummy input
    print("Testing model with dummy input...")
    dummy_input = torch.randn(1, 1, 48, 48)  
    try:
        output = model(dummy_input)
        print(f"✅ Model test successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"⚠️  Model test failed: {e}")
    
    # Save the model
    print(f"Saving PyTorch model to {output_path}...")
    torch.save(model.state_dict(), output_path)
    print("✅ Conversion complete!")
    return output_path

if __name__ == "__main__":
    # Paths
    onnx_path = os.path.join("models_fer2013", "fer2013_model.onnx")
    torch_output_path = os.path.join("models_fer2013", "fer2013_model.pth")
    
    # Convert the model
    convert_onnx_to_torch(onnx_path, torch_output_path)
