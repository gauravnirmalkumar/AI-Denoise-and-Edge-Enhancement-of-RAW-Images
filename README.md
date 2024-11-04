# Image Denoising and Enhancement Techniques

## Overview
This repository contains the implementation of various image denoising and enhancement techniques for 12-bit RAW images. The goal of this project is to apply and compare methods such as AI-based denoising, median filtering, bilateral filtering, and edge enhancement techniques using a Laplacian filter. The output is a 24-bit RGB image, suitable for further processing and analysis.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
     - [AI Denoising Example](#ai-denoising-example)
     - [Tradtional Example](#tradtional-example)
- [Evaluation Metrics](#evaluation-metrics)
     - [Spatial Signal-to-Noise Ratio Calculation](#spatial-signal-to-noise-ratio-calculation)
     - [Edge Strength Calculation](#edge-strength-calculation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- Implementation of AI-based denoising using U-net Neural Network model. From [here](https://github.com/JavierGurrola/RDUNet) 
- Application of median and bilateral filters for traditional denoising.
- Edge enhancement using Laplacian filters.
- Evaluation metrics including spatial signal-to-noise ratio and edge strength.
- Modular and organized code structure for easy understanding and contributions.

## Technologies Used
- Python
- OpenCV
- PyTorch
- NumPy
- Matplotlib (for visualization)

## Getting Started
To get a local copy of this project up and running, follow these steps:

1. **Clone the repository:**

2. **Set up a virtual environment:** (Recommended: [Anaconda](https://www.anaconda.com/))
   ```bash
   conda create --name myenv
   conda activate myenv
   ```
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
## Usage
To run the main script, use the following command:
   ```bash
   python main.py
   ```
### Input
The input for the program should be a 12-bit RAW image file located in the data/ directory.

### Output
The denoised output will be saved in the output/ directory in 24-bit RGB format.

### Pre-trained Model
You can find the pre-trained model [here](https://drive.google.com/drive/folders/1jF8YF-7SoVpc4y39_lFl25OBFVQmZAWJ) (model_color.pth). Download the model_color.pth file and place it in the model_color/ directory for use in the denoising process.

## Results

### AI Denoising Example
Below are some examples of the denoising results achieved with this project.

#### AI denoised image using [RDUNet](https://github.com/JavierGurrola/RDUNet):
![AI_denoised_24bit](https://github.com/user-attachments/assets/3a34ed64-3c89-4483-9d01-6aea6f8657ec)

### Tradtional Example
###### Bilateral filter:
![bilateral_filtered](https://github.com/user-attachments/assets/418f06bc-b308-44db-9e3d-3f2f074a3998)

###### Gaussian filter (Implemented in Assignment 1):
![gaussian_filtered](https://github.com/user-attachments/assets/2282ab21-2753-4f7b-853a-d548009f07c8)

###### Laplacian filter
![Laplacian_20241105_000209](https://github.com/user-attachments/assets/5ff2c7a8-ec65-4c37-92d5-6cec032a7c53)


###### Median filter:
![median_filtered](https://github.com/user-attachments/assets/7c123fc1-935f-4cb5-b88c-bf85a4610ec3)

## Evaluation Metrics

## Spatial Signal-to-Noise Ratio Calculation

### Overview
In this section, we compute the **Spatial Signal-to-Noise Ratio (SSNR)** for three different gray tones using various denoising and enhancement methods implemented in this project. We will also apply edge enhancement techniques, including the **Laplacian filter**, and compare the results with previous methods implemented in Assignment 2.

### Methods Implemented
1. **Laplacian Filter for Edge Enhancement**
   - The Laplacian filter is used to enhance edges in an image. It calculates the second derivative of the image, highlighting areas of rapid intensity change.
   
2. **Denoising Techniques Compared**
   - **Median Filter:** Reduces noise by replacing each pixel's value with the median value of the pixels in its neighborhood.
   - **Bilateral Filter:** Preserves edges while smoothing out noise by considering both spatial and intensity differences.
   - **Gaussian Filter:** Smooths the image by averaging pixel values, which may blur edges.
   - **Edge Enhancement (from Assignment 2):** A previous method that enhances edges, which we will compare against.

### Computation of SSNR
The formula for calculating the SSNR is given by:
 ```math
\text{SSNR} = 10 \cdot \log_{10}\left(\frac{(R^2)}{MSE}\right)
```

Where:
-  $R\$ is the maximum possible pixel value (255 for 8-bit grayscale).
- $MSE\$ is the Mean Squared Error between the original and denoised images.

# Image Filter Performance Analysis

Analysis of various image filtering techniques applied to RAW image processing, comparing quality metrics and edge preservation characteristics.

## Configuration

### Input Parameters
- **Format**: Bayer 12-bits
- **Pattern**: GRBG
- **Resolution**: 1920x1280

### Output Parameters
- **Format**: RGB channel
- **Bit Depth**: 24-bit (8 bits per channel)

### Tools Used
- PixelViewer
- Irfanview with RAW plugin
- Tensorflow (CNN-based denoising)

## Performance Metrics

| Filter Type | PSNR (dB) | SNR (dB) | Edge Strength (Original) | Edge Strength (Processed) |
|-------------|:---------:|:--------:|:----------------------:|:-----------------------:|
| Median      | 42.31     | 32.26    | 261.88                | 211.09                 |
| AI Denoised | 40.06     | 30.02    | 261.88                | 159.42                 |
| Gaussian    | 38.98     | 28.94    | 261.88                | 178.64                 |
| Bilateral   | 37.92     | 27.88    | 261.88                | 145.96                 |
| Laplacian   | 28.39     | 18.35    | 261.88                | 477.54                 |

## Key Findings

### Filter Rankings (by PSNR)
1. Median Filter: 42.31 dB
2. AI Denoised: 40.06 dB
3. Gaussian: 38.98 dB
4. Bilateral: 37.92 dB
5. Laplacian: 28.39 dB

### Edge Preservation Analysis
- **Best Edge Retention**: Median Filter (211.09)
- **Strongest Edge Enhancement**: Laplacian (477.54)
- **Most Edge Smoothing**: Bilateral (145.96)

### Notable Observations
- Median filtering provides the best balance between noise reduction and edge preservation
- AI-based denoising shows promising results with second-best PSNR performance
- Laplacian filter significantly amplifies edges but introduces noise (lowest PSNR)

## Model Information
- Framework: Tensorflow
- Architecture: CNN-based denoising
- Implementation: Pre-trained model

## License
This project is licensed under the MIT License.

## Citation (AI U-Net for Image Denoising)
```
@article{gurrola2021residual,
  title={A Residual Dense U-Net Neural Network for Image Denoising},
  author={Gurrola-Ramos, Javier and Dalmau, Oscar and Alarcón, Teresa E},
  journal={IEEE Access},
  volume={9},
  pages={31742--31754},
  year={2021},
  publisher={IEEE},
  doi={10.1109/ACCESS.2021.3061062}
}
```

## Acknowledgments

[OpenCV Documentation](https://opencv.org/)
[PyTorch Documentation](https://pytorch.org/)

Special thanks to the contributors, and anyone who has provided feedback on this project.

Thanks for taking your time to get to the end :D
