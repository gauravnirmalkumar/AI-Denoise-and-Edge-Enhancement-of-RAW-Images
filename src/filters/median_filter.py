# src/filters/median_filter.py
import cv2

def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)
