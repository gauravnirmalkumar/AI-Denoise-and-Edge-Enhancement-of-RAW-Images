import numpy as np
import cv2

def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter using OpenCV's optimized function.
    Ensure output remains in high bit-depth format.
    """
    # Ensure image is in 16-bit for the bilateral filter
    if image.dtype == np.uint16:
        # Convert to float for processing
        image_float = image.astype(np.float32) / 65535.0
    else:
        image_float = np.clip(image * 65535, 0, 65535).astype(np.uint16)

    output = cv2.bilateralFilter(
        image_float,
        d=d,  # Diameter of each pixel neighborhood
        sigmaColor=sigma_color,  # Filter sigma in the color space
        sigmaSpace=sigma_space  # Filter sigma in the coordinate space
    )
    
    return output.astype(np.float32)  # Output remains as float in the range [0, 1]

    
    return output.astype(np.float32) / 65535 

def apply_gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.5) -> np.ndarray:
    """
    Apply Gaussian filter using OpenCV while preserving high bit-depth.
    """
    return cv2.GaussianBlur(image.astype(np.float32), (kernel_size, kernel_size), sigma)

def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filter using OpenCV while preserving high bit-depth.
    """
    kernel_size = max(3, kernel_size + (kernel_size + 1) % 2)  # Ensure kernel size is odd
    # Ensure the image is in a suitable format
    return cv2.medianBlur(image.astype(np.uint16), kernel_size)



def apply_laplacian_filter(image: np.ndarray) -> np.ndarray:
    """Apply a Laplacian filter to enhance edges using OpenCV."""
    # Apply GaussianBlur to reduce noise before Laplacian
    blurred_image = cv2.GaussianBlur(image.astype(np.float32), (3, 3), 0)
    laplacian_image = cv2.Laplacian(blurred_image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian_image)

def save_as_png(image: np.ndarray, filepath: str):
    """
    Save the high-bit image as a PNG file.
    Ensure no data is lost by saving as 16-bit or higher.
    """
    # Normalize to 0-65535 range if necessary
    if image.dtype == np.float32:
        image = np.clip(image * 65535, 0, 65535).astype(np.uint16)  # Convert to 16-bit unsigned int

    cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Save without compression

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
