"""
chunk 切分预览脚本 — 不入库，只展示切分结果

用法：
    python scripts/preview_chunks.py                    # 输出到终端（前30个）
    python scripts/preview_chunks.py --all              # 输出全部
    python scripts/preview_chunks.py --export           # 导出到 /mnt/d/Download/chunk_preview.txt
    python scripts/preview_chunks.py --min 50 --max 200 # 只看长度在50~200字之间的chunk
    python scripts/preview_chunks.py --short            # 只看 <50 字的短chunk（用于检查噪声）
"""
import sys
import argparse
sys.path.insert(0, '/home/hanyuu/rag_project')

from loaders.mineru_loader import MinerULoader
from splitters.markdown_splitter import split_documents
import statistics

CONTENT_LIST = (
    '/home/hanyuu/rag_project/data/mineru_output/'
    '【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节/auto/'
    'content_list_with_tables.json'
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='显示全部chunk')
    parser.add_argument('--export', action='store_true', help='导出到D盘txt文件')
    parser.add_argument('--short', action='store_true', help='只看 <50 字的chunk')
    parser.add_argument('--min', type=int, default=0, help='最小长度过滤')
    parser.add_argument('--max', type=int, default=99999, help='最大长度过滤')
    parser.add_argument('--n', type=int, default=30, help='终端显示数量（默认30）')
    args = parser.parse_args()

    print('加载并切分文档...')
    loader = MinerULoader(CONTENT_LIST)
    docs = loader.load()
    chunks = split_documents(docs)

    lengths = [len(c.page_content) for c in chunks]

    # 统计信息
    print(f'\n{"="*60}')
    print(f'切分完成：共 {len(chunks)} 个 chunk')
    print(f'最短: {min(lengths)} 字  最长: {max(lengths)} 字')
    print(f'中位数: {statistics.median(lengths):.0f} 字  平均: {statistics.mean(lengths):.0f} 字')
    print(f'分布：')
    for t in [30, 64, 128, 256, 512]:
        n = sum(1 for l in lengths if l <= t)
        print(f'  ≤{t:4d}字: {n:4d} 个 ({n/len(chunks)*100:.1f}%)')
    print(f'  >512字: {sum(1 for l in lengths if l > 512)} 个')
    print(f'{"="*60}\n')

    # 过滤
    if args.short:
        filtered = [(i, c) for i, c in enumerate(chunks) if len(c.page_content) < 50]
        print(f'[短chunk过滤] 共 {len(filtered)} 个 < 50字的chunk\n')
    else:
        filtered = [(i, c) for i, c in enumerate(chunks)
                    if args.min <= len(c.page_content) <= args.max]

    lines = []
    limit = len(filtered) if args.all or args.export else min(args.n, len(filtered))

    for idx, (i, chunk) in enumerate(filtered[:limit]):
        section = chunk.metadata.get('section_path', '')
        block_type = chunk.metadata.get('block_type', 'text')
        length = len(chunk.page_content)
        content = chunk.page_content

        sep = '─' * 60
        header = f'[{i+1:03d}] 长度={length}字  类型={block_type}'
        if section:
            header += f'\n      章节: {section}'

        lines.append(sep)
        lines.append(header)
        lines.append('')

        # 内容：终端截断显示，导出完整显示
        if args.export:
            lines.append(content)
        else:
            preview = content[:300] + ('...' if len(content) > 300 else '')
            lines.append(preview)
        lines.append('')

    output = '\n'.join(lines)

    if args.export:
        out_path = '/mnt/d/Download/chunk_preview.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f'切分统计：共{len(chunks)}个chunk，中位数{statistics.median(lengths):.0f}字\n\n')
            f.write(output)
        print(f'已导出到 {out_path}')
        print(f'（共导出 {limit} 个chunk）')
    else:
        print(output)
        if limit < len(filtered):
            print(f'... 还有 {len(filtered)-limit} 个未显示，用 --all 查看全部，或 --export 导出')

if __name__ == '__main__':
    main()
