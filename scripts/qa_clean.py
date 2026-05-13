"""
QA 数据集清理脚本

操作：
1. 删除 PDF 解析异常条目（目录页 TOC 残留）
2. 清理 source_text / ground_truth / question 中的 LaTeX 标记
3. 输出清理后的数据集

用法：
    python scripts/qa_clean.py \
        --input  data/qa_dataset/hydro_reg_qa.json \
        --output data/qa_dataset/hydro_reg_qa_clean.json
"""

import argparse
import json
import re
from pathlib import Path

# ── 要删除的 index 列表（人工确认的无效条目）──────────────────────────────────
DELETE_INDICES = {2, 3}  # 目录页 QA，source_text 含 TOC 残留

# ── LaTeX 清理规则 ────────────────────────────────────────────────────────────
# 1. 把 $...$ 块整体替换为空（内联公式）
INLINE_LATEX = re.compile(r'\$[^$]+\$')
# 2. 残余的命令前缀（如 \mathrm \times）
LATEX_CMD = re.compile(r'\\[a-zA-Z]+\s*\{?|\}')
# 3. 多余空格压缩
MULTI_SPACE = re.compile(r'  +')


def clean_latex(text: str) -> str:
    text = INLINE_LATEX.sub('', text)
    text = LATEX_CMD.sub('', text)
    text = MULTI_SPACE.sub(' ', text)
    return text.strip()


def process(input_path: str, output_path: str):
    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)

    original_count = len(items)
    cleaned = []
    latex_fixed = 0
    deleted = 0

    for idx, item in enumerate(items):
        # 删除指定 index
        if idx in DELETE_INDICES:
            print(f"  删除 [{idx}] p.{item.get('page')} {item.get('question','')[:40]}")
            deleted += 1
            continue

        # 清理 LaTeX
        changed = False
        for field in ("source_text", "ground_truth", "question"):
            original = item.get(field, "")
            cleaned_text = clean_latex(original)
            if cleaned_text != original:
                item[field] = cleaned_text
                changed = True
        if changed:
            latex_fixed += 1

        cleaned.append(item)

    # 保存
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"\n原始条数：{original_count}")
    print(f"删除条数：{deleted}")
    print(f"LaTeX清理条数：{latex_fixed}")
    print(f"输出条数：{len(cleaned)}")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/qa_dataset/hydro_reg_qa.json")
    parser.add_argument("--output", default="data/qa_dataset/hydro_reg_qa_clean.json")
    args = parser.parse_args()
    process(args.input, args.output)
