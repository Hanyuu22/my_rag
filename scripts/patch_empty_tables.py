"""
用 VLM 补全 MinerU 提取失败的空表格块。

扫描 content_list JSON，找出 type=table 且 text='' 且 table_body='' 的块，
用 pypdfium2 渲染对应页面区域，调用 DashScope VLM 提取 Markdown 表格行，
将结果写回 JSON（生成新文件，不覆盖原文件）。

用法：
    python scripts/patch_empty_tables.py
    python scripts/patch_empty_tables.py --dry-run    # 只显示将处理哪些块，不调用 VLM
    python scripts/patch_empty_tables.py --block 808  # 只处理指定块
"""

import argparse
import base64
import copy
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from openai import OpenAI

sys.path.insert(0, "/home/hanyuu/rag_project")
from config import DASHSCOPE_API_KEY

CONTENT_LIST = (
    "/home/hanyuu/rag_project/data/mineru_output/"
    "【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节/auto/"
    "content_list_with_tables.json"
)
PDF_PATH = (
    "/home/hanyuu/rag_project/data/mineru_output/"
    "【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节/auto/"
    "【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节_origin.pdf"
)
OUTPUT_JSON = (
    "/home/hanyuu/rag_project/data/mineru_output/"
    "【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节/auto/"
    "content_list_with_tables_patched.json"
)

# VLM 渲染 DPI 倍率
RENDER_SCALE = 3

# 在 bbox 四周扩展多少 pts，确保捕获完整行
BBOX_EXPAND_X = 20  # 水平方向扩展（基本不需要，表格通常横跨全页）
BBOX_EXPAND_Y_UP = 60   # 向上多扩展，确保抓到 bbox 上方的行（MinerU bbox 偏低的问题）
BBOX_EXPAND_Y_DOWN = 20


def _is_empty_table(block: dict) -> bool:
    return (
        block.get("type") == "table"
        and not block.get("text", "").strip()
        and not block.get("table_body", "").strip()
    )


def _render_region(doc: pdfium.PdfDocument, page_idx: int, bbox: list) -> bytes:
    """渲染 PDF 页面的指定区域，返回 PNG bytes。"""
    page = doc[page_idx]
    w_pts = page.get_width()
    h_pts = page.get_height()

    x1, y1, x2, y2 = bbox
    # 扩展 bbox
    x1 = max(0, x1 - BBOX_EXPAND_X)
    y1 = max(0, y1 - BBOX_EXPAND_Y_UP)
    x2 = min(w_pts, x2 + BBOX_EXPAND_X)
    y2 = min(h_pts, y2 + BBOX_EXPAND_Y_DOWN)

    bm = page.render(scale=RENDER_SCALE, rotation=0)
    img = bm.to_pil()

    ix1 = int(x1 * RENDER_SCALE)
    iy1 = int(y1 * RENDER_SCALE)
    ix2 = int(x2 * RENDER_SCALE)
    iy2 = int(y2 * RENDER_SCALE)
    cropped = img.crop((ix1, iy1, ix2, iy2))

    page.close()
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def _is_sep(line: str) -> bool:
    return bool(re.match(r"^\|[\s\-:]+(\|[\s\-:]+)+\|?$", line.strip()))


def _col_count(line: str) -> int:
    return line.count("|") - 1 if line.strip().startswith("|") else 0


def _first_cell(line: str) -> str:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    return parts[0] if parts else ""


def _clean_vlm_output(text: str) -> str:
    """
    去除多余内容：
    1. 按空行分段，取第一段
    2. 去掉分隔行
    3. 遇到以下情况截断（遇到新表格的标志）：
       a. 列数变化 >= 4（表格结构改变）
       b. 已出现过数字首列行后，出现纯中文/字母首列行（表头）
    """
    sections = re.split(r"\n\s*\n", text.strip())
    rows = [l for l in sections[0].split("\n") if l.strip().startswith("|")]
    rows = [l for l in rows if not _is_sep(l)]

    if not rows:
        return text.strip()

    ref_cols = _col_count(rows[0])
    seen_numeric_first = False
    clean = []
    for r in rows:
        c = _col_count(r)
        fc = _first_cell(r)

        # 列数突变
        if abs(c - ref_cols) >= 3:
            break

        # 曾经出现过数字首列行，现在出现非数字、非空的首列行 → 下一张表的表头
        if seen_numeric_first and fc and not fc[0].isdigit():
            break

        if fc and fc[0].isdigit():
            seen_numeric_first = True

        clean.append(r)
    return "\n".join(clean)


def _vlm_extract_table(img_bytes: bytes, context_caption: str, client: OpenAI) -> str:
    """调用 DashScope VLM，将图片中的表格内容提取为 Markdown 行。"""
    b64 = base64.b64encode(img_bytes).decode()
    prompt = (
        f"这是一段表格图片（表格名称参考：{context_caption or '未知'}）。"
        "请将图中所有表格数据行提取为 Markdown 表格格式（仅输出数据行，不要重复表头行，"
        "不要添加解释文字）。每行格式：| 列1 | 列2 | ... |"
    )
    resp = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只列出空块，不调用 VLM")
    parser.add_argument("--block", type=int, default=None, help="只处理指定 block 索引")
    args = parser.parse_args()

    with open(CONTENT_LIST, encoding="utf-8") as f:
        data = json.load(f)

    # 找所有空表格块
    empty_blocks = [
        (i, b) for i, b in enumerate(data)
        if _is_empty_table(b) and (args.block is None or i == args.block)
    ]

    if not empty_blocks:
        print("没有找到空表格块。")
        return

    print(f"找到 {len(empty_blocks)} 个空表格块：")
    for i, b in empty_blocks:
        cap = b.get("table_caption", [])
        print(f"  [{i}] page={b.get('page_idx')} bbox={b.get('bbox')} caption={cap}")

    if args.dry_run:
        return

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    doc = pdfium.PdfDocument(PDF_PATH)
    patched = copy.deepcopy(data)
    patch_count = 0

    for i, b in empty_blocks:
        page_idx = b.get("page_idx", 0)
        bbox = b.get("bbox", [0, 0, 800, 200])
        caption = " ".join(b.get("table_caption", []))

        # 找前一个同页或相邻页的非空表格块，获取 caption 作为上下文
        context_caption = caption
        if not context_caption:
            for j in range(i - 1, max(0, i - 10), -1):
                prev = data[j]
                if prev.get("type") == "table" and prev.get("table_caption"):
                    prev_page = prev.get("page_idx", -1)
                    if abs(prev_page - page_idx) <= 1:
                        context_caption = " ".join(prev.get("table_caption", []))
                        break

        print(f"\n处理块 [{i}] page={page_idx} context='{context_caption}'...")

        try:
            img_bytes = _render_region(doc, page_idx, bbox)

            # 保存调试图片
            debug_path = f"/tmp/patch_block_{i}.png"
            with open(debug_path, "wb") as f:
                f.write(img_bytes)
            print(f"  裁剪区域已保存到 {debug_path}")

            raw = _vlm_extract_table(img_bytes, context_caption, client)
            extracted = _clean_vlm_output(raw)
            print(f"  VLM 原始输出：\n{raw}\n")
            print(f"  清理后结果：\n{extracted}\n")

            patched[i]["text"] = extracted
            patch_count += 1
        except Exception as e:
            print(f"  处理块 [{i}] 失败: {e}")

    doc.close()

    if patch_count > 0:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(patched, f, ensure_ascii=False, indent=2)
        print(f"\n已修补 {patch_count} 个块，结果写入: {OUTPUT_JSON}")
    else:
        print("没有成功修补任何块。")


if __name__ == "__main__":
    main()
