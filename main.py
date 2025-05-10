import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog
import time
import os

def calculate_rmse(img1, img2):
    mse = mean_squared_error(img1, img2)
    return np.sqrt(mse)

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

def process(method_name, edge_image, original, output_dir, exec_time):
    filename = os.path.join(output_dir, f"edges_{method_name}.jpg")
    cv2.imwrite(filename, edge_image)
    psnr_val = compare_psnr(original, edge_image)
    mse_val = mean_squared_error(original, edge_image)
    rmse_val = calculate_rmse(original, edge_image)
    print(f"\n🔹 روش: {method_name}")
    print(f"📤 فایل ذخیره‌شده: {filename}")
    print(f"📊 PSNR: {psnr_val:.2f} dB")
    print(f"📊 MSE: {mse_val:.2f}")
    print(f"📊 RMSE: {rmse_val:.2f}")
    print(f"⏱ زمان اجرا: {exec_time:.4f} ثانیه")
    return filename

def get_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='انتخاب تصویر X-Ray',
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    return file_path

if __name__ == '__main__':
    print("در حال انتخاب فایل تصویر...")
    image_path = get_image_file()
    if not image_path:
        print("❌ تصویری انتخاب نشد.")
    else:
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            print("❌ بارگذاری تصویر با خطا مواجه شد.")
            exit()

        output_dir = os.path.dirname(image_path)
        print("\n▶ شروع پردازش الگوریتم‌ها...")

        t1 = time.time()
        edge_sobel = sobel(original)
        t2 = time.time()
        process("sobel", edge_sobel, original, output_dir, t2 - t1)

        t3 = time.time()
        edge_canny = canny(original)
        t4 = time.time()
        process("canny", edge_canny, original, output_dir, t4 - t3)

        t5 = time.time()
        edge_stat = statistical_range(original)
        t6 = time.time()
        process("statistical", edge_stat, original, output_dir, t6 - t5)

        total_time = t6 - t1
        print(f"\n✅ پردازش کامل شد.")
        print(f"⏱ زمان کل برای همه الگوریتم‌ها: {total_time:.2f} ثانیه")
