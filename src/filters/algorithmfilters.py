# bilateral_filter.py
import numpy as np
import cv2

def apply_bilateral_filter(image: np.ndarray, diameter=9, sigma_color=75, sigma_space=75) -> np.ndarray:
    """
    Apply bilateral filter to image.
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

# gaussian_filter.py
def apply_gaussian_filter(image: np.ndarray, kernel_size=5, sigma=1.5) -> np.ndarray:
    """
    Apply Gaussian filter to image.
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


# median_filter.py
def apply_median_filter(image: np.ndarray, kernel_size=3) -> np.ndarray:
    """
    Apply median filter to image.
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return cv2.medianBlur(image, kernel_size)

# laplacian_filter.py
def apply_laplacian_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian filter for edge enhancement while preserving the original image content.
    
    Args:
        image (np.ndarray): Input image in BGR format
        
    Returns:
        np.ndarray: Edge-enhanced image
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Convert to float32 for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Process each channel separately
    channels = cv2.split(image_float)
    enhanced_channels = []
    
    for channel in channels:
        # Apply Gaussian blur to reduce noise (smaller kernel for better detail preservation)
        blurred = cv2.GaussianBlur(channel, (3, 3), 0.5)
        
        # Apply Laplacian with a smaller kernel size
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=1)
        
        # Enhance edges by subtracting Laplacian from original
        # (subtracting because Laplacian highlights edges with both positive and negative values)
        enhanced = channel - laplacian
        
        # Normalize to [0, 1] range
        enhanced = np.clip(enhanced, 0, 1)
        
        enhanced_channels.append(enhanced)
    
    # Merge channels back together
    enhanced_image = cv2.merge(enhanced_channels)
    
    # Convert back to uint8
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    
    # Apply subtle contrast enhancement
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.1, beta=0)
    
    return enhanced_image

def save_as_png(image: np.ndarray, filepath: str):
    """
    Save image as PNG file.
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def compute_snr(original: np.ndarray, denoised: np.ndarray) -> float:
    noise = original.astype(np.float32) - denoised.astype(np.float32)
    signal_power = np.mean(original.astype(np.float32) ** 2)
    noise_power = np.mean(noise ** 2)
    snr_value = 10 * np.log10(signal_power / noise_power)
    return snr_value

def compute_edge_strength(image: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    return np.sqrt(sobel_x**2 + sobel_y**2)
