import cv2
import numpy as np
from filters.median_filter import apply_median_filter
from filters.bilateral_filter import apply_bilateral_filter
from filters.gaussian_filter import apply_gaussian_filter
from ai_model.denoise_ai_model import load_pytorch_model, apply_ai_denoise
import torch
import torch.nn.functional as F
def load_bayer_raw_image(file_path, width=1920, height=1280, bit_depth=12):
    # Load the raw image data as a 12-bit grayscale image
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint16)

    # Reshape the data based on the image dimensions and Bayer pattern
    raw_data = raw_data.reshape((height, width))

    # Normalize to 8-bit if needed
    rgb_image = (raw_data >> 4).astype(np.uint8)  # Shift to 8-bit if required

    # Convert from Bayer to RGB (using OpenCV's demosaicing)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BAYER_GR2RGB)

    return rgb_image

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
    weight_map = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Define tile boundaries and adjust for edges
            end_y = min(y + tile_size, height)
            end_x = min(x + tile_size, width)
            tile = input_image[y:end_y, x:end_x]

            # Resize tile to model's expected input size if it's smaller
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size))

            # Apply AI denoise model to the resized tile
            tile_denoised = apply_ai_denoise(model, tile)
            tile_denoised_np = tile_denoised.detach().cpu().numpy()

            # Remove batch dimension if present
            if tile_denoised_np.shape[0] == 1:
                tile_denoised_np = tile_denoised_np.squeeze(0)

            # Resize denoised tile back to original tile size if it was resized
            if end_y - y != tile_size or end_x - x != tile_size:
                tile_denoised_np = F.interpolate(
                    torch.from_numpy(tile_denoised_np).unsqueeze(0),
                    size=(end_y - y, end_x - x),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0).numpy()

            # Adjust mask to the actual tile size
            tile_h, tile_w = tile_denoised_np.shape[1:3]
            mask_y = np.linspace(0, 1, tile_h, dtype=np.float32).reshape(-1, 1)
            mask_x = np.linspace(0, 1, tile_w, dtype=np.float32).reshape(1, -1)
            weight_mask = np.minimum(mask_y, mask_y[::-1]) * np.minimum(mask_x, mask_x[:, ::-1])
            weight_mask = np.stack([weight_mask] * 3, axis=0)

            # Place the denoised tile into the output image with weighted blending
            denoised_image[y:end_y, x:end_x] += tile_denoised_np.transpose(1, 2, 0) * weight_mask.transpose(1, 2, 0)
            weight_map[y:end_y, x:end_x] += weight_mask.transpose(1, 2, 0)

    # Normalize for overlapping areas
    weight_map = np.where(weight_map == 0, 1e-8, weight_map)
    denoised_image /= weight_map

    # Convert to 8-bit RGB format
    denoised_image = np.clip(denoised_image * 255, 0, 255).astype(np.uint8)
    return denoised_image

def main():
    # Load Bayer RAW image and convert to RGB
    input_file = r"D:\assignment2\data\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
    input_image_12bit = load_raw_image_tile(input_file)

    # Load pre-trained AI model (PyTorch)
    model_path = r'D:\assignment2\src\ai_model\model_color\model_color.pth'
    model = load_pytorch_model(model_path)

    # Apply denoise with larger tile size and proportional overlap
    tile_size = 512  # Increased tile size
    overlap = 20     # Increased overlap to keep blending smooth with larger tiles
    ai_denoised_image_24bit = apply_denoise_in_tiles(model, input_image_12bit, tile_size, overlap)

    # Save and display denoised image
    cv2.imwrite('AI_denoised_24bit.png', ai_denoised_image_24bit)

    # Display in a brightened format for visualization
    cv2.imshow('Denoised Image (Brightened)', ai_denoised_image_24bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply other filters (median, bilateral, gaussian) to the original RGB image
    rgb_image = load_bayer_raw_image(input_file)  # Load RGB again for filtering
    median_filtered = apply_median_filter(rgb_image)
    bilateral_filtered = apply_bilateral_filter(rgb_image)
    gaussian_filtered = apply_gaussian_filter(rgb_image)

    # Save the results as 8-bit images (for easy viewing)
    cv2.imwrite('output/median_filtered.png', median_filtered)
    cv2.imwrite('output/bilateral_filtered.png', bilateral_filtered)
    cv2.imwrite('output/gaussian_filtered.png', gaussian_filtered)

if __name__ == "__main__":
    main()

# The main function loads a Bayer RAW image, converts it to RGB, and applies denoising using the AI model. It then saves and displays the denoised image. Additionally, it applies other filters (median, bilateral, and Gaussian) to the original RGB image and saves the results as separate images for comparison.
