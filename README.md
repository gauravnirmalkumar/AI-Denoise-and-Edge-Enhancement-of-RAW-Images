# Image Denoising and Enhancement Techniques

## Overview
This repository contains the implementation of various image denoising and enhancement techniques for 12-bit RAW images. The goal of this project is to apply and compare methods such as AI-based denoising, median filtering, bilateral filtering, and edge enhancement techniques using a Laplacian filter. The output is a 24-bit RGB image, suitable for further processing and analysis.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- Implementation of AI-based denoising using pre-trained models.
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
   ```
   git clone https://github.com/yourusername/YourRepoName.git
   cd YourRepoName
Set up a virtual environment:



python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
Install the required packages:



pip install -r requirements.txt
Usage
To run the main script, use the following command:



python main.py
Input
The input for the program should be a 12-bit RAW image file located in the data/ directory.

Output
The denoised output will be saved in the output/ directory in 24-bit RGB format.

Pre-trained Model
You can find the pre-trained model here. Download the model and place it in the appropriate directory for use in the denoising process.

Results
Denoising Example
Below are some examples of the denoising results achieved with this project.

AI denoised image:

Commonly Used Algorithms
Bilateral filter:Gaussian filter (Implemented in Assignment 1):Median filter:

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
OpenCV Documentation
PyTorch Documentation
Special thanks to the contributors and anyone who has provided feedback on this project.

