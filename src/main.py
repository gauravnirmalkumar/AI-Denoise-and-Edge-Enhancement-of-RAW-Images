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
from filters.algorithmfilters import (
    apply_bilateral_filter, 
    apply_gaussian_filter, 
    apply_median_filter, 
    save_as_png, 
    apply_laplacian_filter,
    compute_snr,
    compute_edge_strength
)
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
    """Load and convert Bayer RAW image to RGB."""
    try:
        raw_data = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
        rgb_image = (raw_data >> (bit_depth - 8)).astype(np.uint8)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BAYER_GR2RGB)
        return rgb_image
    except Exception as e:
        print("Error loading image:", str(e))
        return None

def process_tile(args: Tuple) -> np.ndarray:
    """Process a single tile through the AI model."""
    tile, model, device = args
    with torch.no_grad():
        tile_tensor = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        denoised_tile = model(tile_tensor)
        return (denoised_tile.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)

def apply_denoise_in_tiles(model: torch.nn.Module, input_image: np.ndarray, 
                          tile_size: int = 512, overlap: int = 16, 
                          num_workers: Optional[int] = None) -> np.ndarray:
    """Apply AI denoising in tiles for memory efficiency."""
    height, width, channels = input_image.shape
    denoised_image = np.zeros_like(input_image, dtype=np.float32)
    weight_map = np.zeros_like(input_image, dtype=np.float32)
    
    num_workers = num_workers or min(32, multiprocessing.cpu_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    tiles = []
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            end_y = min(y + tile_size, height)
            end_x = min(x + tile_size, width)
            tiles.append((input_image[y:end_y, x:end_x], model, device))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_tile, tiles), 
                          total=len(tiles), desc="Processing tiles"))
    
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
    """Load the trained PyTorch model."""
    model = RDUNet()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def save_metrics_to_file(metrics: dict, file_path: str):
    """Save filter metrics to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for key, value in metrics.items():
            f.write("%s: %s\n" % (key, value))
    print("Metrics saved to", file_path)

def display_and_store_results(original: np.ndarray, processed: np.ndarray, 
                            filter_name: str, output_folder: str, summary: list):
    """Compute and store filter comparison metrics."""
    snr = compute_snr(original, processed)
    psnr = cv2.PSNR(original, processed)
    edge_strength_original = np.mean(compute_edge_strength(original))
    edge_strength_processed = np.mean(compute_edge_strength(processed))
    
    metrics = {
        'Filter': filter_name,
        'PSNR': psnr,
        'SNR': snr,
        'Edge Strength Original': edge_strength_original,
        'Edge Strength Processed': edge_strength_processed
    }
    
    summary.append(metrics)
    metrics_file = os.path.join(output_folder, filter_name+"_metrics.txt")
    save_metrics_to_file(metrics, metrics_file)
    print("Metrics saved for"+filter_name+":"+metrics_file)

def save_summary_report(summary: list, report_path: str):
    """Generate and save comprehensive comparison report."""
    with open(report_path, 'w') as report_file:
        report_file.write("Filter Comparison Report\n\n")
        for metrics in summary:
            report_file.write("Filter: " + metrics["Filter"] + "\n")
            report_file.write("PSNR: " + str(round(metrics["PSNR"], 2)) + " dB\n")
            report_file.write("SNR: " + str(round(metrics["SNR"], 2)) + " dB\n")
            report_file.write("Edge Strength Original: " + str(round(metrics["Edge Strength Original"], 2)) + "\n")
            report_file.write("Edge Strength Processed: " + str(round(metrics["Edge Strength Processed"], 2)) + "\n\n")
    print("Summary report saved at",report_path)

def main():
    try:
        # Initialize settings and paths
        optimize_torch_settings()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure paths
        input_file = "D:\\assignment2\\data\\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
        model_path = 'ai_model/model_color/model_color.pth'
        output_folder = "output"
        denoised_folder = os.path.join(output_folder, "AI_Denoised_Images")
        traditional_folder = os.path.join(output_folder, "Traditional_Filter_Images")
        
        # Create output directories
        os.makedirs(denoised_folder, exist_ok=True)
        os.makedirs(traditional_folder, exist_ok=True)
        
        # Load and process input image
        input_image = load_bayer_raw_image(input_file)
        if input_image is None:
            raise ValueError("Failed to load input image")
            
        # Load AI model and apply denoising
        print("Loading AI model...")
        model = load_pytorch_model(model_path)
        
        print("Applying AI denoising...")
        denoised_image = apply_denoise_in_tiles(
            model,
            input_image,
            tile_size=256,
            overlap=16
        )
        
        # Save AI-denoised result
        save_as_png(denoised_image, 
                   os.path.join(denoised_folder, "AI_Denoised_"+timestamp+".png"))
        
        # Apply traditional filters
        print("Applying traditional filters...")
        bilateral_image = apply_bilateral_filter(input_image)
        median_image = apply_median_filter(input_image)
        gaussian_image = apply_gaussian_filter(input_image)
        laplacian_image = apply_laplacian_filter(input_image)
        
        # Save traditional filter results
        save_as_png(bilateral_image, 
                   os.path.join(traditional_folder, 'Bilateral.png'))
        save_as_png(median_image, 
                   os.path.join(traditional_folder, 'Median.png'))
        save_as_png(gaussian_image, 
                   os.path.join(traditional_folder, 'Gaussian.png'))
        save_as_png(laplacian_image, 
                   os.path.join(traditional_folder, 'Laplacian.png'))
        
        # Compute and save metrics
        summary = []
        print("Computing metrics...")
        display_and_store_results(input_image, denoised_image, 
                                "AI Denoised", denoised_folder, summary)
        display_and_store_results(input_image, bilateral_image, 
                                "Bilateral", traditional_folder, summary)
        display_and_store_results(input_image, median_image, 
                                "Median", traditional_folder, summary)
        display_and_store_results(input_image, gaussian_image, 
                                "Gaussian", traditional_folder, summary)
        display_and_store_results(input_image, laplacian_image, 
                                "Laplacian", traditional_folder, summary)
        
        # Save final report
        summary_report_path = os.path.join(output_folder, "filter_comparison_report.txt")
        save_summary_report(summary, summary_report_path)
        print("Processing complete!")
        
    except Exception as e:
        print("An error occurred in main():", str(e))

if __name__ == "__main__":
    main()