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
   ```bash
   cd C:/type/path/here
   git clone https://github.com/gauravnirmalkumar/AIDenoisingAndEdgeEnhancement.git
   cd AIDenoisingAndEdgeEnhancement
   ```
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

### Gray Tones
We will compute the SSNR for three specific gray tones:
1. **Gray Tone 1:** (e.g., 50)
2. **Gray Tone 2:** (e.g., 128)
3. **Gray Tone 3:** (e.g., 200)

For each gray tone, we will calculate the SSNR for all the methods listed above.

### Results
| Method                     | Gray Tone 1 (50) | Gray Tone 2 (128) | Gray Tone 3 (200) |
|----------------------------|-------------------|--------------------|--------------------|
| Median Filter              | SSNR Value 1      | SSNR Value 2       | SSNR Value 3       |
| Bilateral Filter           | SSNR Value 4      | SSNR Value 5       | SSNR Value 6       |
| Gaussian Filter            | SSNR Value 7      | SSNR Value 8       | SSNR Value 9       |
| Laplacian Filter           | SSNR Value 10     | SSNR Value 11      | SSNR Value 12      |
| Edge Enhancement (A2)      | SSNR Value 13     | SSNR Value 14      | SSNR Value 15      |

## Edge Strength Calculation
For each of the methods implemented, we will compute the edge strength based on a gradient-based approach:

1. **Gradient-Based Edge Detection:**
   - Use Sobel or Prewitt operators to calculate gradients for both the x and y directions.
   - Compute the gradient magnitude:
    
   $\text{Magnitude} = \sqrt{G_x^2 + G_y^2}$
    
   
   Where $\( G_x \)$ and $\( G_y \)$ are the gradients in the x and y directions, respectively.

2. **Compute Edge Strength for Each Method:**
   - Calculate the average gradient magnitude across the image for each method.

### Results of Edge Strength Calculation
| Method                     | Edge Strength (Gradient Magnitude) |
|----------------------------|-------------------------------------|
| Median Filter              | Edge Strength 1                     |
| Bilateral Filter           | Edge Strength 2                     |
| Gaussian Filter            | Edge Strength 3                     |
| Laplacian Filter           | Edge Strength 4                     |
| Edge Enhancement (A2)      | Edge Strength 5                     |

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
