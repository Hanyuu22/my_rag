"""临时脚本：将 data/raw/ 下的两个 GB 标准 PDF 入库到 gb_standards collection"""
import sys
sys.path.insert(0, '/home/hanyuu/rag_project')

from backend.services.pipeline import process_file

files = [
    'data/raw/GB+23971-2025.pdf',
    'data/raw/GB+46767-2025.pdf',
]

for f in files:
    print(f'\n{"="*50}')
    print(f'处理：{f}')
    print('='*50)
    try:
        r = process_file(f, collection_name='gb_standards', use_vlm_ocr=True)
        print(f'完成：{r}')
    except Exception as e:
        print(f'失败：{e}')
