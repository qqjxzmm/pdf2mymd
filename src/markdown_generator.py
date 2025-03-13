import os

class MarkdownGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.md_dir = os.path.join(output_dir, 'md')
        os.makedirs(self.md_dir, exist_ok=True)

    def generate_page_md(self, page_num, ocr_results, image_path):
        """生成单页MD文件"""
        md_path = os.path.join(self.md_dir, f'page_{page_num}.md')
        relative_image_path = os.path.relpath(image_path, self.output_dir)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            # 添加图片引用
            f.write(f'![第{page_num}页]({relative_image_path})\n\n')
            
            # 处理OCR结果
            if isinstance(ocr_results, list):
                # 如果是列表，直接处理
                for result in ocr_results:
                    if isinstance(result, tuple):
                        # 如果是元组，取文本部分
                        text = result[1] if len(result) > 1 else str(result)
                    elif isinstance(result, dict):
                        # 如果是字典，取text字段
                        text = result.get('text', '')
                    else:
                        text = str(result)
                    
                    if text.strip():  # 只写入非空文本
                        f.write(f'{text}\n\n')

    def generate_summary(self, total_pages):
        """生成汇总MD文件"""
        summary_path = os.path.join(self.output_dir, 'summary.md')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            for page_num in range(1, total_pages + 1):
                page_md = os.path.join(self.md_dir, f'page_{page_num}.md')
                if os.path.exists(page_md):
                    with open(page_md, 'r', encoding='utf-8') as page_f:
                        content = page_f.read()
                        if content.strip():  # 只添加非空页面
                            f.write(f'## 第 {page_num} 页\n\n')
                            f.write(content)
                            f.write('\n---\n\n')