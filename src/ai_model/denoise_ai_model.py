# src/ai_model/denoise_ai_model.py
import torch
import numpy as np

from ai_model.model import RDUNet  # Ensure your model class is imported

def load_pytorch_model(model_path):
    # Load the PyTorch model from .pth file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model with the correct parameters
    model = RDUNet(channels=3, base_filters=128)  # Ensure 'base_filters' matches training setup
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    model.to(device)  # Move model to the appropriate device
    model.eval()  # Set the model to evaluation mode
    return model


def apply_ai_denoise(model, image_input):
    # Ensure input is in the format (C, H, W) after any batch dimension removal
    if image_input.ndim == 4:  # Case when we have an extra batch dimension
        image_input = image_input.squeeze(0)
    
    # Convert to (C, H, W) if input is in (H, W, C) format
    if image_input.shape[0] != 3:  # Assuming the first dimension should be the channel
        image_input = image_input.transpose(2, 0, 1)  # Convert HWC to CHW
    
    # Convert to a float tensor for processing and ensure it’s on the same device as the model
    image_tensor = torch.tensor(image_input, dtype=torch.float32, device=model.device)
    
    # Normalize to model input range (assuming 0-1 for 12-bit data)
    image_tensor /= 4095.0  # Scale for 12-bit range
    
    with torch.no_grad():
        image_output = model(image_tensor.unsqueeze(0))  # Add batch dimension for model processing
    
    return image_output


