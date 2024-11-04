import numpy as np
import torch
import cv2
from ai_model.denoise_ai_model import load_pytorch_model, apply_ai_denoise

def load_raw_image_tile(file_path, width=1920, height=1280, bit_depth=12, tile_size=256):
    # Load raw Bayer data
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint16).reshape((height, width))

    # Convert Bayer pattern to RGB, adjust pattern if needed
    rgb_image_12bit = np.zeros((height, width, 3), dtype=np.uint16)
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = raw_data[y:y+tile_size, x:x+tile_size]
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BAYER_GR2RGB)  # Adjust pattern if necessary
            rgb_image_12bit[y:y+tile_size, x:x+tile_size] = tile_rgb

    return rgb_image_12bit


def apply_denoise_in_tiles(model, input_image, tile_size, overlap=10):
    height, width, _ = input_image.shape
    denoised_image = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width, 3), dtype=np.float16)  # Using float16 for lighter memory

    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Extract the tile with overlap, making sure not to exceed image dimensions
            tile = input_image[y:y + tile_size, x:x + tile_size]

            # Ensure the tile size is valid
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue  # Skip if the tile is not full-sized (could be at the edges)

            # Apply AI denoise model to the tile
            tile_denoised = apply_ai_denoise(model, tile)  # Model output is assumed to be a tensor
            tile_denoised_np = tile_denoised.detach().cpu().numpy()  # Convert to NumPy

            # Remove batch dimension if present
            if tile_denoised_np.shape[0] == 1:
                tile_denoised_np = tile_denoised_np.squeeze(0)  # Now shape (3, 256, 256)

            # Define the blending mask (weight map) for smooth blending in overlap
            tile_h, tile_w = tile_denoised_np.shape[1:3]  # Get height and width from the denoised tile shape
            mask_y = np.linspace(0, 1, tile_h, dtype=np.float16).reshape(-1, 1)
            mask_x = np.linspace(0, 1, tile_w, dtype=np.float16).reshape(1, -1)
            weight_mask = np.minimum(mask_y, mask_y[::-1]) * np.minimum(mask_x, mask_x[:, ::-1])
            weight_mask = np.stack([weight_mask] * 3, axis=0)  # Expand mask to (3, tile_h, tile_w)

            # Place the denoised tile into the output image with weighted blending
            denoised_image[y:y + tile_h, x:x + tile_w] += tile_denoised_np.transpose(1, 2, 0) * weight_mask.transpose(1, 2, 0)
            weight_map[y:y + tile_h, x:x + tile_w] += weight_mask.transpose(1, 2, 0)

    # Normalize to ensure the brightness remains unchanged in the overlapping areas
    denoised_image /= (weight_map + 1e-8)

    # Convert to 8-bit RGB format if needed
    denoised_image = np.clip(denoised_image * 255, 0, 255).astype(np.uint8)
    return denoised_image



# Define model loading and denoise function
model_path = 'D:/assignment2/src/ai_model/model_color/model_color.pth'
model = load_pytorch_model(model_path)

# Load raw Bayer image and convert to 12-bit RGB
width, height = 1920, 1280
tile_size = 256
input_image_12bit = load_raw_image_tile(r'D:\assignment2\data\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw', width, height, tile_size=tile_size)

# Apply denoise with overlap blending
denoised_image_12bit = apply_denoise_in_tiles(model, input_image_12bit, tile_size)

# Save and display denoised image
cv2.imwrite('denoised_image_12bit.png', denoised_image_12bit)

# Display in a brightened format for visualization
display_image = cv2.convertScaleAbs(denoised_image_12bit, alpha=(255.0 / 4095.0))
cv2.imshow('Denoised Image (Brightened)', display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
