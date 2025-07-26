import onnx
import os

def check_onnx_model(model_path):
    print(f"Checking ONNX model at: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ Error: Model file does not exist")
        return False
        
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Model file exists. Size: {file_size:.2f} MB")
    
    try:
        print("\nAttempting to load ONNX model...")
        model = onnx.load(model_path)
        print("✅ ONNX model loaded successfully")
        
        print("\nChecking model validity...")
        onnx.checker.check_model(model)
        print("✅ ONNX model is valid")
        
        print("\nModel Info:")
        print(f"  IR Version: {model.ir_version}")
        print(f"  Producer: {model.producer_name} {model.producer_version}")
        print(f"  Domain: {model.domain}")
        print(f"  Version: {model.model_version}")
        print(f"  Doc String: {model.doc_string}")
        
        print("\nGraph Info:")
        print(f"  Name: {model.graph.name}")
        print(f"  Inputs: {[i.name for i in model.graph.input]}")
        print(f"  Outputs: {[o.name for o in model.graph.output]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking ONNX model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    model_path = os.path.join("models_fer2013", "fer2013_model.onnx")
    print("=" * 60)
    print("ONNX Model Validation Tool")
    print("=" * 60)
    check_onnx_model(model_path)
