from pdf2image import convert_from_path
import os

class PDFProcessor:
    def __init__(self, output_base_dir):
        self.output_base_dir = output_base_dir

    def convert_pdf_to_images(self, pdf_path, dpi=300):
        """将PDF转换为高分辨率图片"""
        try:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join(self.output_base_dir, pdf_name)
            images_dir = os.path.join(output_dir, 'images')
            
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'md'), exist_ok=True)
    
            # 分批处理PDF页面，每次处理10页
            batch_size = 10
            image_paths = []
            page_num = 1
            
            while True:
                try:
                    print(f"正在处理第 {page_num} - {page_num + batch_size - 1} 页...")
                    images = convert_from_path(
                        pdf_path,
                        dpi=dpi,
                        first_page=page_num,
                        last_page=page_num + batch_size - 1
                    )
                    
                    if not images:
                        break
                        
                    # 保存本批次的图片
                    for i, image in enumerate(images):
                        image_path = os.path.join(images_dir, f'page_{page_num + i}.jpg')
                        image.save(image_path, 'JPEG')
                        image_paths.append(image_path)
                    
                    page_num += len(images)
                    
                except Exception as e:
                    if "begin greater than end" in str(e):
                        break
                    raise e
            
            # 确保按照页码数字顺序排序
            def get_page_num(path):
                filename = os.path.basename(path)
                return int(filename.replace('page_', '').replace('.jpg', ''))
                
            image_paths.sort(key=get_page_num)
            print(f"总共处理了 {len(image_paths)} 页")
            
            return image_paths, output_dir
            
        except Exception as e:
            print(f"转换PDF时出错: {str(e)}")
            return [], None