import os
import sys
import logging
import torch
import onnx
import onnxruntime as ort
#from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_onnx_model(model_path):
    """Validate the ONNX model and return True if valid."""
    try:
        logger.info(f"Validating ONNX model at: {model_path}")
        
        # Check file exists and size
        if not os.path.exists(model_path):
            logger.error("Model file does not exist")
            return False
            
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model file size: {file_size:.2f} MB")
        
        # Try to load the ONNX model
        try:
            logger.info("Loading ONNX model...")
            onnx_model = onnx.load(model_path)
            logger.info("ONNX model loaded successfully")
            
            # Check model
            logger.info("Checking model...")
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model is valid")
            
            # Print model metadata
            logger.info(f"Model IR version: {onnx_model.ir_version}")
            logger.info(f"Producer name: {onnx_model.producer_name}")
            logger.info(f"Producer version: {onnx_model.producer_version}")
            
            # Try to create inference session
            logger.info("Creating ONNX Runtime session...")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            # Print input/output details
            for i, input in enumerate(session.get_inputs()):
                logger.info(f"Input {i}: {input.name}, shape: {input.shape}, type: {input.type}")
            
            for i, output in enumerate(session.get_outputs()):
                logger.info(f"Output {i}: {output.name}, shape: {output.shape}, type: {output.type}")
            
            #logger.info("ONNX Runtime session created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error validating ONNX model: {str(e)}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error during validation: {str(e)}", exc_info=True)
        return False

def convert_onnx_to_torch(onnx_path, output_path):
    """Attempt to convert ONNX model to PyTorch format."""
    try:
        logger.info(f"Attempting to convert {onnx_path} to PyTorch format...")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Create dummy input based on model's expected input shape
        # This is a common input shape for FER models: (batch_size, channels, height, width)
        dummy_input = torch.randn(1, 1, 48, 48)
        
        # Direct conversion from ONNX to PyTorch is not supported by PyTorch.
        # You need to manually recreate the model architecture in PyTorch and load weights, or use a third-party library like onnx2pytorch.
        logger.error("Direct conversion from ONNX to PyTorch is not supported. Please use onnx2pytorch or manually recreate the model.")
        return False
        
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Paths
    onnx_path = os.path.join("models_fer2013", "fer2013_model.onnx")
    torch_output_path = os.path.join("models_fer2013", "fer2013_model.pth")
    
    logger.info("=" * 80)
    logger.info("FER2013 Model Validation and Conversion Tool")
    logger.info("=" * 80)
    
    # Step 1: Validate ONNX model
    logger.info("\n[STEP 1/2] Validating ONNX model...")
    is_valid = validate_onnx_model(onnx_path)
    
    if not is_valid:
        logger.error("ONNX model validation failed. Cannot proceed with conversion.")
        sys.exit(1)
    
    # Step 2: Convert to PyTorch
    logger.info("\n[STEP 2/2] Attempting to convert to PyTorch format...")
    if convert_onnx_to_torch(onnx_path, torch_output_path):
        logger.info("\nConversion completed successfully!")
        logger.info(f"PyTorch model saved to: {torch_output_path}")
    else:
        logger.error("\nConversion failed. See error messages above for details.")
        sys.exit(1)
