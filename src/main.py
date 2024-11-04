import cv2
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import time
from datetime import datetime
from typing import Tuple, Optional
from ai_model.model import RDUNet 
from filters.algorithmfilters import apply_bilateral_filter, apply_gaussian_filter, apply_median_filter, save_as_png, apply_laplacian_filter
import os
class EnhancedTimer:
    def __init__(self, description: str):
        self.description = description
        self.times = []
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        end_time = time.perf_counter()
        self.times.append(end_time - self.start_time)
        
    @property
    def average_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0

def optimize_torch_settings():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    torch.set_num_threads(multiprocessing.cpu_count())

def load_bayer_raw_image(file_path: str, width: int = 1920, height: int = 1280, 
                        bit_depth: int = 12) -> Optional[np.ndarray]:
    try:
        with EnhancedTimer("Loading RAW image"):
            raw_data = np.memmap(file_path, dtype=np.uint16, mode='r', shape=(height, width))
            rgb_image = (raw_data >> (bit_depth - 8)).astype(np.uint8)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BAYER_GR2RGB)
            return rgb_image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None
def load_bayer_raw_imagefilter(file_path: str, width: int = 1920, height: int = 1280, 
                         bit_depth: int = 12) -> Optional[np.ndarray]:
    try:
        with EnhancedTimer("Loading RAW image"):
            raw_data = np.memmap(file_path, dtype=np.uint16, mode='r', shape=(height, width))
            # No need to convert to uint8; keep it as uint16
            return raw_data
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def process_tile(args: Tuple) -> np.ndarray:
    tile, model, device = args
    with torch.no_grad():
        tile_tensor = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        denoised_tile = model(tile_tensor)
        return (denoised_tile.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)

def apply_denoise_in_tiles(model: torch.nn.Module, input_image: np.ndarray, 
                           tile_size: int = 512, overlap: int = 16, 
                           num_workers: Optional[int] = None) -> np.ndarray:
    height, width, channels = input_image.shape
    denoised_image = np.zeros_like(input_image, dtype=np.float32)
    weight_map = np.zeros_like(input_image, dtype=np.float32)
    
    if num_workers is None:
        num_workers = min(32, multiprocessing.cpu_count())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    tiles = []
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            end_y = min(y + tile_size, height)
            end_x = min(x + tile_size, width)
            tile = input_image[y:end_y, x:end_x]
            tiles.append((tile, model, device))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_tile, tiles), total=len(tiles), desc="Processing tiles"))
        
    tile_idx = 0
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            end_y = min(y + tile_size, height)
            end_x = min(x + tile_size, width)
            denoised_image[y:end_y, x:end_x] += results[tile_idx][:end_y-y, :end_x-x]
            weight_map[y:end_y, x:end_x] += 1
            tile_idx += 1
    
    denoised_image /= np.clip(weight_map, 1e-6, None)
    return np.clip(denoised_image, 0, 255).astype(np.uint8)

def load_pytorch_model(model_path: str) -> torch.nn.Module:
    model = RDUNet()  # Initialize the model architecture
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)  # Load the state dictionary into the model
    return model

def compute_snr(original: np.ndarray, denoised: np.ndarray) -> float:
    noise = original.astype(np.float32) - denoised.astype(np.float32)
    signal_power = np.mean(original.astype(np.float32) ** 2)
    noise_power = np.mean(noise ** 2)
    snr_value = 10 * np.log10(signal_power / noise_power)
    return snr_value

def compute_edge_strength(image: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return np.sqrt(sobel_x**2 + sobel_y**2)

def display_results(original: np.ndarray, denoised: np.ndarray):
    snr = compute_snr(original, denoised)
    psnr = cv2.PSNR(original, denoised)
    print(f"PSNR: {psnr:.2f} dB, SNR: {snr:.2f} dB")

def save_metrics_to_file(metrics: dict, file_path: str):
    """Save metrics data to a text file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the output directory exists
    with open(file_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")
    print(f"Metrics saved to {file_path}")
def display_and_store_results(original: np.ndarray, denoised: np.ndarray, filter_name: str, output_folder: str, summary: list):
    """Compute and save PSNR, SNR, and edge strength results for comparison."""
    snr = compute_snr(original, denoised)
    psnr = cv2.PSNR(original, denoised)
    edge_strength_original = np.mean(compute_edge_strength(original))
    edge_strength_denoised = np.mean(compute_edge_strength(denoised))
    
    metrics = {
        'Filter': filter_name,
        'PSNR': psnr,
        'SNR': snr,
        'Edge Strength Original': edge_strength_original,
        'Edge Strength Denoised': edge_strength_denoised
    }
    
    # Append metrics to summary list
    summary.append(metrics)
    
    # Save individual filter metrics
    metrics_file = os.path.join(output_folder, f"{filter_name}_metrics.txt")
    save_metrics_to_file(metrics, metrics_file)
    print(f"Metrics saved for {filter_name}: {metrics_file}")
def save_summary_report(summary: list, report_path: str):
    """Save a summary report of all metrics for each filter and denoising technique."""
    with open(report_path, 'w') as report_file:
        report_file.write("Filter Comparison Report\n\n")
        for metrics in summary:
            report_file.write(f"Filter: {metrics['Filter']}\n")
            report_file.write(f"PSNR: {metrics['PSNR']:.2f} dB\n")
            report_file.write(f"SNR: {metrics['SNR']:.2f} dB\n")
            report_file.write(f"Edge Strength Original: {metrics['Edge Strength Original']:.2f}\n")
            report_file.write(f"Edge Strength Denoised: {metrics['Edge Strength Denoised']:.2f}\n\n")
    print(f"Summary report saved at {report_path}")

def main():
    try:
        # Optimize Torch settings based on system capabilities
        optimize_torch_settings()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define paths for input, model, and output
        input_file = r"D:\assignment2\data\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
        model_path = r'D:\assignment2\src\ai_model\model_color\model_color.pth'
        output_folder = r"D:\assignment2\src\output"
        denoised_folder = os.path.join(output_folder, "AI_Denoised_Images")
        traditional_folder = os.path.join(output_folder, "Traditional_Filter_Images")
        os.makedirs(denoised_folder, exist_ok=True)
        os.makedirs(traditional_folder, exist_ok=True)
        
        # Load the RAW input image
        input_image = load_bayer_raw_image(input_file)
        if input_image is None:
            print("Failed to load input image")
            return
        if input_image.dtype == np.uint16:
            input_image = input_image.astype(np.float32) / 65535  # Convert to float and normalize
        elif input_image.dtype == np.float32:
            input_image = np.clip(input_image, 0, 1)  # Ensure values are within [0, 1]

        # Load AI model (commented out for testing filters)
        # print("Loading AI model...")
        # model = load_pytorch_model(model_path)
        # model.eval()  # Set model to evaluation mode
        
        # Apply AI model-based denoising in tiles for memory efficiency (commented out)
        # denoised_image = apply_denoise_in_tiles(
        #     model,
        #     input_image,
        #     tile_size=256,
        #     overlap=16,
        #     num_workers=min(32, multiprocessing.cpu_count())
        # )

        # Placeholder for denoised_image (commented out)
        # denoised_image = input_image  # Uncomment this line when running AI denoising

        # Load the raw image specifically for filters
        input_image_filter = load_bayer_raw_imagefilter(input_file)
        
        # Apply traditional filters for comparison
        bilateral_image = apply_bilateral_filter(input_image_filter)
        median_image = apply_median_filter(input_image_filter)
        gaussian_image = apply_gaussian_filter(input_image_filter)
        laplacian_image = apply_laplacian_filter(input_image_filter)

        # Save traditional filter outputs
        save_as_png(bilateral_image, os.path.join(traditional_folder, f'Bilateral_{timestamp}.png'))
        save_as_png(median_image, os.path.join(traditional_folder, f'Median_{timestamp}.png'))
        save_as_png(gaussian_image, os.path.join(traditional_folder, f'Gaussian_{timestamp}.png'))
        save_as_png(laplacian_image, os.path.join(traditional_folder, f'Laplacian_{timestamp}.png'))

        summary = []
        # Compute, display, and store metrics for each result
        # display_and_store_results(input_image_filter, denoised_image, "AI Denoised", denoised_folder, summary)  # Commented out for now
        display_and_store_results(input_image_filter, bilateral_image, "Bilateral", traditional_folder, summary)
        display_and_store_results(input_image_filter, median_image, "Median", traditional_folder, summary)
        display_and_store_results(input_image_filter, gaussian_image, "Gaussian", traditional_folder, summary)
        display_and_store_results(input_image_filter, laplacian_image, "Laplacian", traditional_folder, summary)
        
        # Save summary report
        summary_report_path = os.path.join(output_folder, "filter_comparison_report.txt")
        save_summary_report(summary, summary_report_path)

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()
