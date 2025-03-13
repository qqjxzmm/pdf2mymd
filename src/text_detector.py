import cv2
import numpy as np
from PIL import Image
import os
from paddleocr import PaddleOCR

class TextDetector:
    def __init__(self):
        # 仅使用检测模型，不加载识别模型
        self.detector = PaddleOCR(use_angle_cls=True, lang='ch', rec=False, show_log=False)

    def detect_text_area(self, image_path, debug=False):
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                return 0.0, None
            
            # 使用PaddleOCR检测文本区域
            result = self.detector.ocr(image_path, rec=False, cls=False)
            
            # 处理检测结果
            height, width = image.shape[:2]
            total_area = height * width
            text_area = 0
            boxes = []
            
            # 安全地处理检测结果
            if result and isinstance(result, list) and len(result) > 0:
                # 确保结果不为空且格式正确
                if isinstance(result[0], list):
                    boxes = [np.array(box, dtype=np.int32) for box in result[0]]
                    text_area = sum(cv2.contourArea(box) for box in boxes)
            
            ratio = min(text_area / total_area, 1.0)
            
            # 调试模式：保存可视化结果
            if debug:
                debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # 在原图上绘制检测到的文字区域
                debug_img = image.copy()
                if boxes:
                    cv2.drawContours(debug_img, boxes, -1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(debug_dir, f'{base_name}_result.jpg'), debug_img)
                
                debug_info = {
                    'total_area': total_area,
                    'text_area': text_area,
                    'boxes_count': len(boxes),
                    'debug_dir': debug_dir
                }
                return ratio, debug_info
            
            return ratio, None
            
        except Exception as e:
            print(f"处理图片时出错 {image_path}: {str(e)}")
            return 0.0, None