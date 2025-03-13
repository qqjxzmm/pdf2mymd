# PDF2MyMD

一个将 PDF 文件转换为 Markdown 格式的工具，支持批量处理和分步执行。

## 功能特点

- 支持 PDF 转图片
- 多种 OCR 引擎支持（通义千问、EasyOCR 等）
- 支持批量处理
- 支持分步执行
- 支持单张图片处理

## 安装

```bash
git clone https://github.com/你的用户名/pdf2mymd.git
cd pdf2mymd
pip install -r requirements.txt

## 配置

1. 复制环境变量示例文件：
```bash
cp .env.example .env
2. 编辑 .env 文件，填入你的 API Keys：
```plaintext
MISTRAL_API_KEY=your_mistral_api_key_here
QWEN_API_KEY=your_qwen_api_key_here
 ```
 
## 使用方法
1. 批量处理：
2. 处理单个文件：
3. 处理单张图片：
更多使用说明请参考代码注释。

## 输出目录结构
```plaintext
output/
  └── pdf文件名/
      ├── images/         # 转换后的图片
      ├── ocr_results.json # OCR识别结果
      ├── md/             # 单页Markdown文件
      └── summary.md      # 汇总的Markdown文件
 ```

## 依赖说明
请查看 requirements.txt 文件。

## License
MIT
