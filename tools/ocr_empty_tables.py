"""
对 MinerU content_list.json 中 text='' 的 table block 用 qwen-vl-plus 补充 OCR。

流程：
  1. 读取 content_list.json，找出所有 type='table' 且 text='' 的 block
  2. 对每个 block 的 img_path，用 qwen-vl-plus 识别表格内容（Markdown 格式）
  3. 把识别结果写回 text 字段
  4. 保存为新的 content_list_with_tables.json（不覆盖原文件）
  5. 打印补充前后的统计信息

用法：
    conda activate mineru_2.5
    cd ~/rag_project
    python tools/ocr_empty_tables.py
"""

import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────

CONTENT_LIST_PATH = Path(
    "/home/hanyuu/rag_project/data/mineru_output"
    "/【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节/auto"
    "/【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节_content_list.json"
)
OUTPUT_PATH = CONTENT_LIST_PATH.parent / "content_list_with_tables.json"
IMAGE_BASE = CONTENT_LIST_PATH.parent  # img_path 相对于此目录

VLM_MODEL = "qwen-vl-plus"
REQUEST_INTERVAL = 1.0   # 请求间隔（秒），避免触发限流


# ── VLM 调用 ──────────────────────────────────────────────────────────────

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def image_to_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ocr_table_image(img_path: Path, page: int) -> str:
    """用 qwen-vl-plus 识别表格图片，返回 Markdown 格式文本"""
    b64 = image_to_base64(img_path)
    suffix = img_path.suffix.lstrip(".")  # jpg / png

    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{suffix};base64,{b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "这是一张来自石油化工工艺技术规程的表格图片（第 "
                            f"{page} 页）。\n"
                            "请将表格中的所有内容完整转录为 Markdown 表格格式，"
                            "保留所有数值、单位、行列结构。\n"
                            "只输出 Markdown 表格，不要任何解释或前缀。"
                        ),
                    },
                ],
            }
        ],
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


# ── 主流程 ────────────────────────────────────────────────────────────────

def run_ocr(content_list_path: Path, output_path: Path) -> bool:
    """
    对 content_list.json 中空表格调用 VLM OCR，结果保存到 output_path。

    Args:
        content_list_path: 原始 content_list.json 路径
        output_path: 输出路径（通常是同目录下 content_list_with_tables.json）

    Returns:
        True 表示至少处理了一张表格并保存成功
    """
    image_base = content_list_path.parent

    print(f"读取：{content_list_path.name}")
    with open(content_list_path, encoding="utf-8") as f:
        blocks = json.load(f)

    empty_tables = [
        (i, b) for i, b in enumerate(blocks)
        if b.get("type") == "table" and not b.get("text", "").strip()
        and b.get("img_path")
    ]

    total = len([b for b in blocks if b.get("type") == "table"])
    print(f"表格总数：{total}，其中 text 为空：{len(empty_tables)}")

    if not empty_tables:
        print("没有需要补充的表格，跳过。")
        return False

    print(f"\n开始 VLM OCR（共 {len(empty_tables)} 张，模型：{VLM_MODEL}）\n")

    success = 0
    for idx, (block_idx, block) in enumerate(empty_tables):
        page = block.get("page_idx", 0) + 1
        img_rel = block["img_path"]
        img_path = image_base / img_rel

        print(f"[{idx+1}/{len(empty_tables)}] p.{page}  {img_rel[-20:]}", end="  ")

        if not img_path.exists():
            print("❌ 图片不存在，跳过")
            continue

        try:
            t0 = time.time()
            text = ocr_table_image(img_path, page)
            elapsed = time.time() - t0
            blocks[block_idx]["text"] = text
            success += 1
            preview = text[:60].replace("\n", " ")
            print(f"✅ {elapsed:.1f}s  {preview}...")
        except Exception as e:
            err = str(e)
            if any(k in err for k in ["quota", "Quota", "insufficient", "balance", "429"]):
                print(f"❌ API 额度不足，终止剩余处理：{err}")
                break
            print(f"❌ 失败（跳过）：{err}")

        if idx < len(empty_tables) - 1:
            time.sleep(REQUEST_INTERVAL)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)

    print(f"\n完成：{success}/{len(empty_tables)} 张表格补充成功")
    print(f"已保存到：{output_path.name}")
    return success > 0


def main():
    run_ocr(CONTENT_LIST_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
