# src/filters/gaussian_filter.py
import cv2

def apply_gaussian_filter(image, kernel_size=5, sigma=1.5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
