from text_detector import TextDetector
import os
import argparse

def debug_detection(image_path):
    detector = TextDetector()
    ratio, debug_info = detector.detect_text_area(image_path, debug=True)
    print(f"\n处理图片: {os.path.basename(image_path)}")
    print(f"文字区域占比: {ratio:.2%}")
    if debug_info:
        print("调试信息:")
        for key, value in debug_info.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='文字区域检测调试工具')
    parser.add_argument('--image', type=str, help='单个图片路径')
    parser.add_argument('--dir', type=str, help='图片目录路径')
    args = parser.parse_args()

    if args.image:
        # 处理单个图片
        debug_detection(args.image)
    elif args.dir:
        # 处理目录下所有jpg图片
        for filename in sorted(os.listdir(args.dir)):
            if filename.lower().endswith('.jpg'):
                image_path = os.path.join(args.dir, filename)
                debug_detection(image_path)
    else:
        print("请指定图片路径或目录路径")
        print("示例:")
        print("处理单个图片:")
        print("python src/debug_text_detection.py --image /path/to/image.jpg")
        print("\n处理目录下所有图片:")
        print("python src/debug_text_detection.py --dir /path/to/images/dir")