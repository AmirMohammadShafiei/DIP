from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error
import numpy as np

def calculate_rmse(img1, img2):

    mse = mean_squared_error(img1, img2)
    return np.sqrt(mse)

def evaluate_image(original, processed):

    psnr_val = compare_psnr(original, processed)
    mse_val = mean_squared_error(original, processed)
    rmse_val = calculate_rmse(original, processed)
    
    return {
        'psnr': psnr_val,
        'mse': mse_val,
        'rmse': rmse_val
    }