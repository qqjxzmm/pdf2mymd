import easyocr
import cv2
import numpy as np
import os
from abc import ABC, abstractmethod
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk
import base64
from PIL import Image
import io
import cv2
import numpy as np
from openai import OpenAI
from text_detector import TextDetector
from dotenv import load_dotenv
from paddleocr import PaddleOCR  # 添加这行导入

class BaseOCR(ABC):
    """OCR基类，便于后续扩展其他OCR引擎"""
    @abstractmethod
    def process_image(self, image_path):
        pass

    @abstractmethod
    def get_text_area_ratio(self, image_path):
        pass

import os
from dotenv import load_dotenv

class MistralOCRProcessor(BaseOCR):
    def __init__(self):
        self.api_key = os.getenv('MISTRAL_API_KEY', '')
        if not self.api_key:
            raise ValueError("请设置 MISTRAL_API_KEY 环境变量")
        self.client = Mistral(api_key=self.api_key)

    def _compress_image(self, image_path, max_size_mb=1):
        """压缩图片到指定大小"""
        with Image.open(image_path) as img:
            # 计算当前质量
            buffered = io.BytesIO()
            img.save(buffered, format='JPEG', quality=85)
            size_mb = len(buffered.getvalue()) / (1024 * 1024)
            
            if size_mb > max_size_mb:
                # 计算新的尺寸
                ratio = (max_size_mb / size_mb) ** 0.5
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 转换为base64
            buffered = io.BytesIO()
            img.save(buffered, format='JPEG', quality=85)
            return base64.b64encode(buffered.getvalue()).decode()

    def process_image(self, image_path):
        """处理单张图片"""
        # 压缩并转换图片
        base64_image = self._compress_image(image_path)

        # 调用OCR API
        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            },
            include_image_base64=True
        )

        # 提取文本块信息
        results = []
        if response.pages and len(response.pages) > 0:
            page = response.pages[0]
            # 解析markdown内容
            lines = page.markdown.split('\n')
            img = Image.open(image_path)
            width, height = img.size
            y_pos = 0
            line_height = height / (len(lines) + 1)
            
            for line in lines:
                if line.strip():
                    # 创建更合理的边界框，确保是int32类型
                    y_pos += line_height
                    bbox = np.array([
                        [0, int(y_pos - line_height)],
                        [width, int(y_pos - line_height)],
                        [width, int(y_pos)],
                        [0, int(y_pos)]
                    ], dtype=np.int32)  # 指定数据类型为int32
                    text = line
                    confidence = 1.0
                    results.append((bbox, text, confidence))
            img.close()
        
        return results

    def get_text_area_ratio(self, image_path):
        """计算文字区域占比"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
                
            height, width = image.shape[:2]
            total_area = height * width
            
            # 使用OCR结果计算文本区域
            results = self.process_image(image_path)
            if not results:
                return 0.0
                
            # 计算所有文本框的总面积
            text_area = sum(cv2.contourArea(bbox) for bbox, _, _ in results)
            ratio = text_area / total_area
            return min(ratio, 1.0)  # 确保比例不超过1
            
        except Exception as e:
            print(f"处理图片时出错 {image_path}: {str(e)}")
            return 0.0

class EasyOCRProcessor(BaseOCR):
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim', 'en'])

    def process_image(self, image_path):
        """处理单张图片"""
        results = self.reader.readtext(image_path)
        return results

    def get_text_area_ratio(self, image_path):
        """计算图片中文字区域占比"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告：无法读取图片 {image_path}")
                return 0.0
                
            height, width = image.shape[:2]
            total_area = height * width
            
            results = self.reader.readtext(image_path)
            text_area = 0
            
            for bbox, _, _ in results:
                points = np.array(bbox, np.int32)
                text_area += cv2.contourArea(points)
                
            return text_area / total_area
            
        except Exception as e:
            print(f"处理图片时出错 {image_path}: {str(e)}")
            return 0.0

# 参见 https://bailian.console.aliyun.com/
class QwenOCRProcessor(BaseOCR):
    MODEL_CONFIGS = {
        'qwen-vl-ocr': {
            'prompt': "Read all the text in the image.",
            'model': "qwen-vl-ocr",
            'stream_required': False
        },
        'qwen2.5-vl-72b-instruct': {
            'prompt': "以下是一页文档的图像，返回文档的文本，就像你自然阅读它一样。尽量保留格式，不要虚构内容。",
            'model': "qwen-vl-plus",
            'stream_required': False
        },
        'qwen-vl-max': {
            'prompt': "以下是一页文档的图像，返回文档的文本，就像你自然阅读它一样。尽量保留格式，不要虚构内容。",
            'model': "qwen-vl-plus",
            'stream_required': False
        },
        'qwen2.5-vl-3b-instruct': {
            'prompt': "以下是一页文档的图像，返回文档的文本，就像你自然阅读它一样。尽量保留格式，不要虚构内容。",
            'model': "qwen-vl-plus",
            'stream_required': False
        },
        'qwen-omni-turbo-latest': {
            'prompt': "以下是一页文档的图像，返回文档的文本和整段中文翻译，就像你自然阅读它一样。尽量保留格式，不要虚构内容。",
            'model': "qwen-omni-turbo-latest",
            'stream_required': True
        }
    }

    def __init__(self, model_name='qwen-vl-ocr', custom_prompt=None):
        self.api_key = os.getenv('QWEN_API_KEY', '')
        if not self.api_key:
            raise ValueError("请设置 QWEN_API_KEY 环境变量")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.text_detector = TextDetector()
        self.last_call_time = 0
        self.min_interval = 0.2

        # 设置模型和提示词
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model_config = self.MODEL_CONFIGS[model_name].copy()
        if custom_prompt:
            self.model_config['prompt'] = custom_prompt

    def process_image(self, image_path):
        """处理单张图片"""
        import time
        
        # 限流控制
        current_time = time.time()
        time_diff = current_time - self.last_call_time
        if time_diff < self.min_interval:
            time.sleep(self.min_interval - time_diff)
        
        self.last_call_time = time.time()
        
        # 处理图片并转换为base64
        base64_image = self._process_image_size(image_path)
        
        # 准备API调用参数
        api_params = {
            'model': self.model_config['model'],
            'messages': [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "min_pixels": 28 * 28 * 4,
                        "max_pixels": 3000 * 784
                    },
                    {"type": "text", "text": self.model_config['prompt']}
                ]
            }]
        }

        # 对于需要流式输出的模型，添加必要的参数
        if self.model_config.get('stream_required', False):
            api_params.update({
                'stream': True,
                'stream_options': {"include_usage": True},
                'modalities': ["text"]
            })

        # 调用OCR API
        response = self.client.chat.completions.create(**api_params)

        # 解析响应
        results = []
        img = Image.open(image_path)
        width, height = img.size
        bbox = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)

        if self.model_config.get('stream_required', False):
            # 处理流式响应
            full_text = ""
            for chunk in response:
                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        full_text += chunk.choices[0].delta.content
            text = full_text
        else:
            # 处理普通响应
            if hasattr(response, 'choices') and response.choices:
                text = response.choices[0].message.content
            else:
                text = ""

        if text:
            results.append((bbox, text, 1.0))
        img.close()

        return results

    def _process_image_size(self, image_path):
        """处理图片尺寸并转换为base64"""
        import base64
        from PIL import Image
        import io

        # 读取图片
        with Image.open(image_path) as img:
            # 计算新的尺寸，确保在限制范围内
            max_pixels = 3000 * 784
            min_pixels = 28 * 28 * 4
            
            width, height = img.size
            pixels = width * height
            print(f"图片像素数: {pixels}")

            if pixels > max_pixels:
                scale = (max_pixels / pixels) ** 0.5
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"调整后的尺寸: {new_width}x{new_height}")
            elif pixels < min_pixels:
                scale = (min_pixels / pixels) ** 0.5
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8") # 这里需要解码为字符串，明确指定使用 UTF-8 编码进行解码，在处理包含非ASCII字符时更安全，事实上不加参数会导致大模型处理出问题
            
            return img_str

    def get_text_area_ratio(self, image_path):
        """计算图片中文字区域占比"""
        try:
            # 使用 TextDetector 进行文本区域检测
            ratio, debug_info = self.text_detector.detect_text_area(image_path)
            return ratio
            
        except Exception as e:
            print(f"处理图片时出错 {image_path}: {str(e)}")
            return 0.0

    