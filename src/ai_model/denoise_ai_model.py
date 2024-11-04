import torch
import numpy as np
from ai_model.model import RDUNet  # Ensure your model class is imported

def load_pytorch_model(model_path):
    """
    Loads the PyTorch model with specified weights from a .pth file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model with the same parameters used in training
    model = RDUNet(channels=3, base_filters=128)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    # Move the model to the correct device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model


def apply_ai_denoise(model, image_input):
    """
    Processes an image through the denoising model.
    Arguments:
    - model: the pre-trained RDUNet model.
    - image_input: numpy array of shape (H, W, C), representing a 12-bit RGB image.

    Returns:
    - A denoised image scaled back to 12-bit format.
    """
    # Ensure the input image is in the correct format (H, W, C)
    if image_input.ndim == 3 and image_input.shape[2] == 3:
        # Convert from HWC (Height, Width, Channels) to CHW format
        image_input = np.transpose(image_input, (2, 0, 1))  # Shape becomes (C, H, W)

    # Convert to torch tensor, normalize and move to the model's device
    image_tensor = torch.from_numpy(image_input).float().to(model.device) / 4095.0  # Scale 12-bit to [0, 1]

    # Add a batch dimension for model compatibility (required shape: (1, C, H, W))
    image_tensor = image_tensor.unsqueeze(0)

    # Model inference (without gradient calculations)
    with torch.no_grad():
        image_output = model(image_tensor)  # Shape (1, C, H, W)

    # Remove batch dimension and convert back to numpy format
    denoised_image = image_output.squeeze(0).cpu().numpy()  # Shape (C, H, W)
    
    # Convert back to (H, W, C) format
    denoised_image = np.transpose(denoised_image, (1, 2, 0))

    # Rescale output back to 12-bit integer range
    denoised_image = np.clip(denoised_image * 4095, 0, 4095).astype(np.uint16)

    return denoised_image
