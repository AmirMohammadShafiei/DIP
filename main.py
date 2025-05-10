
import time
import os


from edge_detection.algorithms import sobel, canny, statistical_range
from metrics.evaluation import evaluate_image
from utils.io import get_image_file, load_image, process_and_report


def main():

    print("در حال انتخاب فایل تصویر...")
    image_path = get_image_file()
    
    if not image_path:
        print("❌ تصویری انتخاب نشد.")
        return
    
    original = load_image(image_path)
    if original is None:
        print("❌ بارگذاری تصویر با خطا مواجه شد.")
        return
    
    output_dir = os.path.dirname(image_path)
    print("\n▶ شروع پردازش الگوریتم‌ها...")

    t1 = time.time()
    edge_sobel = sobel(original)
    t2 = time.time()
    metrics_sobel = evaluate_image(original, edge_sobel)
    process_and_report("sobel", edge_sobel, original, output_dir, t2 - t1, metrics_sobel)

    t3 = time.time()
    edge_canny = canny(original)
    t4 = time.time()
    metrics_canny = evaluate_image(original, edge_canny)
    process_and_report("canny", edge_canny, original, output_dir, t4 - t3, metrics_canny)
    t5 = time.time()
    edge_stat = statistical_range(original)
    t6 = time.time()
    metrics_stat = evaluate_image(original, edge_stat)
    process_and_report("statistical", edge_stat, original, output_dir, t6 - t5, metrics_stat)

    total_time = t6 - t1
    print(f"\n✅ پردازش کامل شد.")
    print(f"⏱ زمان کل برای همه الگوریتم‌ها: {total_time:.2f} ثانیه")


if __name__ == '__main__':
    main()
