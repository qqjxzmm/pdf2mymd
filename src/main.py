"""PDF转Markdown工具

这个工具可以将PDF文件转换为Markdown格式，支持批量处理和分步执行。

使用方法：
0. 显示帮助信息
python src/main.py

1. 批量处理：
   python src/main.py --batch [--input-dir /path/to/pdf/folder]
   注：默认输入目录为 input/，可通过 --input-dir 指定其他目录

2. 处理单个文件：
   python src/main.py --pdf /path/to/your.pdf

3. 分步执行：
   - 步骤1 (仅PDF转图片):
     python src/main.py --batch --step 1
   - 步骤2 (仅OCR文字识别):
     python src/main.py --batch --step 2
   - 步骤3 (仅生成Markdown):
     python src/main.py --batch --step 3

4. 组合使用示例：
   - 处理指定目录下所有PDF的第一步：
     python src/main.py --batch --input-dir /path/to/pdf/folder --step 1
   - 处理单个文件的特定步骤：
     python src/main.py --pdf /path/to/your.pdf --step 1

输出目录结构：
output/
  └── pdf文件名/
      ├── images/         # 转换后的图片
      ├── ocr_results.json # OCR识别结果
      ├── md/             # 单页Markdown文件
      └── summary.md      # 汇总的Markdown文件
"""

import os
import argparse
import json
from pdf_processor import PDFProcessor
from ocr_processor import QwenOCRProcessor
from markdown_generator import MarkdownGenerator

class PDFToMarkdown:
    def __init__(self, output_base_dir, text_area_threshold=0.02):  # 调整默认阈值为2%
        self.output_base_dir = output_base_dir
        self.text_area_threshold = text_area_threshold
        self.pdf_processor = PDFProcessor(output_base_dir)
        self.ocr_processor = QwenOCRProcessor(model_name='qwen2.5-vl-72b-instruct')

    def process_image(self, image_path, output_dir=None):
        """处理单张图片
        Args:
            image_path: 图片路径
            output_dir: 输出目录，如果为None则使用图片所在目录
        """
        if output_dir is None:
            output_dir = os.path.dirname(image_path)

        # OCR识别
        text_ratio = self.ocr_processor.get_text_area_ratio(image_path)
        print(f"文字区域占比: {text_ratio:.2%}")
        
        if text_ratio > self.text_area_threshold:
            print("正在进行OCR识别...")
            ocr_results = self.ocr_processor.process_image(image_path)
            
            # 处理识别结果
            results = {
                'image_path': image_path,
                'results': [
                    {
                        'bbox': [float(x) for x in bbox.flatten()] if hasattr(bbox, 'flatten') else bbox,
                        'text': text,
                        'confidence': float(conf)
                    }
                    for bbox, text, conf in ocr_results
                ]
            }
            
            # 输出识别结果
            print("\n识别结果:")
            for result in results['results']:
                print(f"文本: {result['text']}\n")
            
            # 保存结果
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ocr_cache_path = os.path.join(output_dir, f'{base_name}_ocr_results.json')
            with open(ocr_cache_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 生成Markdown
            md_path = os.path.join(output_dir, f'{base_name}.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f'# {base_name}\n\n')
                for result in results['results']:
                    f.write(f'{result["text"]}\n\n')
            
            print(f"\n结果已保存：")
            print(f"OCR结果：{ocr_cache_path}")
            print(f"Markdown文件：{md_path}")
        else:
            print("文字区域太小，跳过OCR处理")

    def process_pdf(self, pdf_path, step=None):
        """处理单个PDF文件
        step: 
            1 - 仅PDF转图片
            2 - 仅OCR文字识别
            3 - 仅生成Markdown
            None - 执行所有步骤
        """
        image_paths = []
        output_dir = None
        
        # 步骤1：PDF转图片
        if step is None or step == 1:
            print(f"步骤1: 转换PDF文件 {pdf_path}")
            image_paths, output_dir = self.pdf_processor.convert_pdf_to_images(pdf_path)
            if step == 1:
                return
        
        # 如果从步骤2开始，需要获取已有的图片路径
        if step in [2, 3] and not image_paths:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join(self.output_base_dir, pdf_name)
            images_dir = os.path.join(output_dir, 'images')
            if os.path.exists(images_dir):
                # 获取所有jpg文件并按页码数字排序
                def get_page_num(filename):
                    return int(filename.replace('page_', '').replace('.jpg', ''))
                
                image_paths = sorted(
                    [os.path.join(images_dir, f) 
                     for f in os.listdir(images_dir) 
                     if f.endswith('.jpg')],
                    key=lambda x: get_page_num(os.path.basename(x))
                )
        
        # 步骤2：OCR识别
        ocr_results_dict = {}
        if step is None or step == 2:
            print(f"步骤2: OCR文字识别")
            for i, image_path in enumerate(image_paths, 1):
                text_ratio = self.ocr_processor.get_text_area_ratio(image_path)
                print(f"页面 {i} 文字区域占比: {text_ratio:.2%}")
                
                if text_ratio > self.text_area_threshold:
                    print(f"正在识别第 {i} 页...")
                    ocr_results = self.ocr_processor.process_image(image_path)
                    
                    # 立即处理并保存当前页的结果
                    page_results = {
                        'image_path': image_path,
                        'results': [
                            {
                                'bbox': [float(x) for x in bbox.flatten()] if hasattr(bbox, 'flatten') else bbox,
                                'text': text,
                                'confidence': float(conf)
                            }
                            for bbox, text, conf in ocr_results
                        ]
                    }
                    ocr_results_dict[i] = page_results
                    
                    # 立即输出当前页的识别结果
                    print(f"\n第 {i} 页识别结果:")
                    for result in page_results['results']:
                        print(f"文本: {result['text']}\n")
                    print("-" * 50)
                else:
                    print(f"跳过第 {i} 页 - 文字区域太小")
            
            # 保存完整的OCR结果到JSON文件
            if ocr_results_dict:
                ocr_cache_path = os.path.join(output_dir, 'ocr_results.json')
                print(f"\n保存OCR结果到: {ocr_cache_path}")
                with open(ocr_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(ocr_results_dict, f, ensure_ascii=False, indent=2)
            else:
                print("警告：没有找到任何可识别的文字内容")
            
            if step == 2:
                return
                
        # 步骤3：生成Markdown
        if step is None or step == 3:
            print(f"步骤3: 生成Markdown文件")
            if step == 3 and not ocr_results_dict:
                # 从缓存加载OCR结果
                ocr_cache_path = os.path.join(output_dir, 'ocr_results.json')
                if os.path.exists(ocr_cache_path):
                    with open(ocr_cache_path, 'r', encoding='utf-8') as f:
                        cached_results = json.load(f)
                        ocr_results_dict = {
                            int(k): {
                                'image_path': v['image_path'],
                                'results': [
                                    (r['bbox'], r['text'], r['confidence'])
                                    for r in v['results']
                                ]
                            }
                            for k, v in cached_results.items()
                        }
            
            md_generator = MarkdownGenerator(output_dir)
            for page_num, data in ocr_results_dict.items():
                md_generator.generate_page_md(
                    page_num, 
                    data['results'],
                    data['image_path']
                )
            md_generator.generate_summary(len(image_paths))

def main():
    parser = argparse.ArgumentParser(description='PDF转Markdown工具')
    parser.add_argument('--step', type=int, choices=[1, 2, 3], 
                       help='指定执行步骤: 1-PDF转图片, 2-OCR识别, 3-生成Markdown')
    parser.add_argument('--pdf', type=str, help='指定单个PDF文件路径')
    parser.add_argument('--image', type=str, help='指定单张图片路径')  # 新增参数
    parser.add_argument('--batch', action='store_true', help='批量处理PDF文件')
    # 修改这里的默认路径
    parser.add_argument('--input-dir', type=str, 
                       default="input",  # 改为相对路径
                       help='指定批量处理的输入目录，默认为 input/')
    
    parser.add_argument('--output-dir', type=str,
                       default="output",  # 改为相对路径
                       help='指定输出目录，默认使用图片所在目录')
    args = parser.parse_args()

    output_dir = args.output_dir or "/Users/jiangxin/Downloads/misocr/pdf2mymd/output"
    processor = PDFToMarkdown(output_dir)
    
    if args.image:
        # 处理单张图片
        if os.path.exists(args.image):
            processor.process_image(args.image, args.output_dir)
        else:
            print(f"错误：图片 {args.image} 不存在")
    elif args.pdf:
        # 处理PDF文件
        if os.path.exists(args.pdf):
            processor.process_pdf(args.pdf, args.step)
        else:
            print(f"错误：文件 {args.pdf} 不存在")
    elif args.batch:
        # 使用用户指定的输入目录
        pdf_dir = args.input_dir
        if not os.path.exists(pdf_dir):
            print(f"错误：输入目录 {pdf_dir} 不存在")
            return
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print("未找到PDF文件，请将PDF文件放入input目录")
            return
            
        print(f"找到 {len(pdf_files)} 个PDF文件")
        for file in pdf_files:
            pdf_path = os.path.join(pdf_dir, file)
            print(f"\n处理文件：{file}")
            processor.process_pdf(pdf_path, args.step)

if __name__ == "__main__":
    main()