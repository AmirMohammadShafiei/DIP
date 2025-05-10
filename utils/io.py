import tkinter as tk
from tkinter import filedialog
import cv2
import os

def get_image_file():

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='انتخاب تصویر X-Ray',
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    return file_path

def load_image(image_path):

    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def save_edge_image(method_name, edge_image, output_dir):

    filename = os.path.join(output_dir, f"edges_{method_name}.jpg")
    cv2.imwrite(filename, edge_image)
    return filename

def process_and_report(method_name, edge_image, original, output_dir, exec_time, metrics):

    filename = save_edge_image(method_name, edge_image, output_dir)
    
    print(f"\n🔹 روش: {method_name}")
    print(f"📤 فایل ذخیره‌شده: {filename}")
    print(f"📊 PSNR: {metrics['psnr']:.2f} dB")
    print(f"📊 MSE: {metrics['mse']:.2f}")
    print(f"📊 RMSE: {metrics['rmse']:.2f}")
    print(f"⏱ زمان اجرا: {exec_time:.4f} ثانیه")
    
    return filename