import cv2
import numpy as np

def statistical_range(original):

    blurred = cv2.GaussianBlur(original, (3, 3), 0)
    edge_image = np.zeros_like(blurred)
    rows, cols = blurred.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            block = blurred[i-1:i+2, j-1:j+2]
            pixel_range = np.max(block) - np.min(block)
            edge_image[i, j] = pixel_range
    return edge_image

def sobel(original):

    sobelx = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return np.uint8(np.clip(sobel, 0, 255))

def canny(original):

    return cv2.Canny(original, 100, 200)