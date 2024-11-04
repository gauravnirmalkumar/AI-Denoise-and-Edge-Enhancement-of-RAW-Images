import cv2
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import time
from datetime import datetime
from typing import Tuple, Optional, Dict,List
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
def compute_region_snr(image: np.ndarray, region: Tuple[int, int, int, int], 
                      gray_level: str) -> float:
    """
    Compute SNR for a specific region of the image.
    
    Args:
        image: Input image array
        region: Tuple of (x, y, width, height) defining the region
        gray_level: String identifier for the gray level being measured
    
    Returns:
        float: SNR value for the region
    """
    x, y, w, h = region
    roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale if image is RGB
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Compute mean signal level
    signal_mean = np.mean(roi)
    
    # Compute noise as standard deviation
    noise_std = np.std(roi)
    
    # Avoid division by zero
    if noise_std == 0:
        return float('inf')
    
    snr = 20 * np.log10(signal_mean / noise_std)
    return snr

def analyze_gray_tones(original_image: np.ndarray, 
                      processed_images: Dict[str, np.ndarray],
                      regions: List[Tuple[int, int, int, int]],
                      gray_levels: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Analyze SNR for multiple gray tone regions across different processing methods.
    
    Args:
        original_image: Original input image
        processed_images: Dictionary of processed images with method names as keys
        regions: List of regions (x, y, width, height) for each gray tone
        gray_levels: List of gray level identifiers
        
    Returns:
        Dictionary containing SNR values for each method and region
    """
    results = {}
    
    # Analyze original image
    results['Original'] = {}
    for region, gray_level in zip(regions, gray_levels):
        results['Original'][gray_level] = compute_region_snr(
            original_image, region, gray_level)
    
    # Analyze processed images
    for method_name, processed_image in processed_images.items():
        results[method_name] = {}
        for region, gray_level in zip(regions, gray_levels):
            results[method_name][gray_level] = compute_region_snr(
                processed_image, region, gray_level)
    
    return results

def save_gray_tone_analysis(results: Dict[str, Dict[str, float]], 
                          output_path: str):
    """
    Save gray tone analysis results to a file.
    
    Args:
        results: Dictionary containing SNR results
        output_path: Path to save the results
    """
    with open(output_path, 'w') as f:
        f.write("Gray Tone SNR Analysis Report\n")
        f.write("============================\n\n")
        
        # Get all methods and gray levels
        methods = list(results.keys())
        gray_levels = list(results[methods[0]].keys())
        
        # Write header
        f.write(f"{'Method':<15}")
        for gray_level in gray_levels:
            f.write(f"{gray_level:<15}")
        f.write("\n")
        f.write("-" * (15 + 15 * len(gray_levels)) + "\n")
        
        # Write results for each method
        for method in methods:
            f.write(f"{method:<15}")
            for gray_level in gray_levels:
                snr = results[method][gray_level]
                f.write(f"{snr:14.2f} ")
            f.write("\n")
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
def save_detailed_snr_report(results: Dict[str, Dict[str, Dict[str, float]]], 
                           timestamp: str,
                           output_path: str):
    """
    Save a detailed SNR analysis report including timestamp and method comparisons.
    
    Args:
        results: Nested dictionary containing analysis results for each method and region
        timestamp: Timestamp string for the report
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write(f"Detailed Gray Tone SNR Analysis Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        # Get all methods and gray levels
        methods = list(results.keys())
        gray_levels = list(results[methods[0]].keys())
        
        # Table header
        f.write(f"{'Method':<20}")
        for gray_level in gray_levels:
            f.write(f"{gray_level + ' SNR':<15}")
        f.write("Average SNR\n")
        f.write("-" * 65 + "\n")
        
        # Results for each method
        best_method = {"name": "", "avg_snr": float('-inf')}
        for method in methods:
            f.write(f"{method:<20}")
            snr_values = []
            for gray_level in gray_levels:
                snr = results[method][gray_level]['snr']  # Access SNR from metrics dictionary
                snr_values.append(snr)
                f.write(f"{snr:14.2f} ")
            avg_snr = sum(snr_values) / len(snr_values)
            f.write(f"{avg_snr:14.2f}\n")
            
            # Track best performing method
            if avg_snr > best_method["avg_snr"]:
                best_method["name"] = method
                best_method["avg_snr"] = avg_snr
        
        # Analysis summary
        f.write("\nAnalysis Summary\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best performing method: {best_method['name']}\n")
        f.write(f"Best average SNR: {best_method['avg_snr']:.2f} dB\n\n")
        
        # Detailed metrics for each method and region
        f.write("\nDetailed Metrics by Region\n")
        f.write("=" * 50 + "\n")
        
        for method in methods:
            f.write(f"\n{method}:\n")
            f.write("-" * 20 + "\n")
            for gray_level in gray_levels:
                metrics = results[method][gray_level]
                f.write(f"{gray_level}:\n")
                f.write(f"  SNR: {metrics['snr']:.2f} dB\n")
                f.write(f"  Mean: {metrics['mean']:.2f}\n")
                f.write(f"  Std Dev: {metrics['std']:.2f}\n")
                f.write(f"  Min: {metrics['min']:.2f}\n")
                f.write(f"  Max: {metrics['max']:.2f}\n")
        
        # Test conditions
        f.write("\nTest Conditions\n")
        f.write("-" * 20 + "\n")
        f.write("- White region coordinates: (860, 375, 100, 50)\n")
        f.write("- Gray region coordinates: (660, 575, 100, 50)\n")
        f.write("- Black region coordinates: (760, 775, 100, 50)\n")
        f.write("- Analysis window size: 100x50 pixels\n")
def compute_region_snr_advanced(image: np.ndarray, 
                              region: Tuple[int, int, int, int]) -> Dict[str, float]:
    """
    Compute advanced SNR metrics for a specific region.
    """
    x, y, w, h = region
    roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Basic SNR calculation
    signal_mean = np.mean(roi)
    noise_std = np.std(roi)
    snr = 20 * np.log10(signal_mean / noise_std) if noise_std > 0 else float('inf')
    
    # Additional metrics
    metrics = {
        'snr': snr,
        'mean': signal_mean,
        'std': noise_std,
        'min': np.min(roi),
        'max': np.max(roi)
    }
    
    return metrics

def analyze_gray_tones_advanced(image: np.ndarray, 
                              processed_images: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Perform advanced analysis of gray tone regions.
    """
    # Define regions based on provided coordinates
    regions = {
        'White': (860, 375, 100, 50),  # Centered around 910,400
        'Gray': (660, 575, 100, 50),   # Centered around 710,600
        'Black': (760, 775, 100, 50)   # Centered around 810,800
    }
    
    results = {}
    
    # Analyze original image
    results['Original'] = {}
    for tone, region in regions.items():
        results['Original'][tone] = compute_region_snr_advanced(image, region)
    
    # Analyze processed images
    for method, processed in processed_images.items():
        results[method] = {}
        for tone, region in regions.items():
            results[method][tone] = compute_region_snr_advanced(processed, region)
    
    return results
def main():
    try:
        # Initialize settings and paths
        optimize_torch_settings()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure paths
        input_file = "..\\data\\eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
        model_path = '..\\src\\ai_model\\model_color\\model_color.pth'
        output_folder = f"output_{timestamp}"
        denoised_folder = os.path.join(output_folder, "AI_Denoised_Images")
        traditional_folder = os.path.join(output_folder, "Traditional_Filter_Images")
        reports_folder = os.path.join(output_folder, "Analysis_Reports")
        
        # Create output directories
        for folder in [denoised_folder, traditional_folder, reports_folder]:
            os.makedirs(folder, exist_ok=True)
        
        # Load and process input image
        print("Loading input image...")
        input_image = load_bayer_raw_image(input_file)
        if input_image is None:
            raise ValueError("Failed to load input image")
            
        # Load and apply AI model
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
        ai_output_path = os.path.join(denoised_folder, f"AI_Denoised_{timestamp}.png")
        save_as_png(denoised_image, ai_output_path)
        print(f"AI-denoised image saved: {ai_output_path}")
        
        # Apply traditional filters
        print("Applying traditional filters...")
        processed_images = {
            "AI_Denoised": denoised_image,
            "Bilateral": apply_bilateral_filter(input_image),
            "Median": apply_median_filter(input_image),
            "Gaussian": apply_gaussian_filter(input_image),
            "Laplacian": apply_laplacian_filter(input_image)
        }
        
        # Save traditional filter results
        for method, image in processed_images.items():
            if method != "AI_Denoised":  # Already saved AI result
                output_path = os.path.join(traditional_folder, f"{method}_{timestamp}.png")
                save_as_png(image, output_path)
                print(f"{method} filter result saved: {output_path}")
        
        # Perform advanced gray tone analysis
        print("Performing gray tone analysis...")
        analysis_results = analyze_gray_tones_advanced(input_image, processed_images)
        
        # Save detailed SNR report
        snr_report_path = os.path.join(reports_folder, f"snr_analysis_{timestamp}.txt")
        save_detailed_snr_report(analysis_results, timestamp, snr_report_path)
        print(f"SNR analysis report saved: {snr_report_path}")
        
        # Generate summary metrics
        summary = []
        for method, image in processed_images.items():
            display_and_store_results(
                input_image, 
                image, 
                method, 
                reports_folder if method == "AI_Denoised" else traditional_folder,
                summary
            )
        
        # Save final summary report
        summary_report_path = os.path.join(reports_folder, f"filter_comparison_{timestamp}.txt")
        save_summary_report(summary, summary_report_path)
        print(f"Summary report saved: {summary_report_path}")
        
        print("\nProcessing complete!")
        print(f"All results saved in: {output_folder}")
        
    except Exception as e:
        print(f"An error occurred in main(): {str(e)}")
        raise

if __name__ == "__main__":
    main()