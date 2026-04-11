"""
批量出题脚本：从 GB 标准文档生成 RAG 测试集

流程：
  1. 读取指定目录下所有已解析的 content_list.json（MinerU 输出）
  2. 提取文档文本（跳过图片块，保留表格 OCR 结果）
  3. 调用 qwen-plus API，按四类题型出题
  4. 保存为 RAGAS 兼容格式 JSON

用法：
  conda activate mineru_2.5
  cd ~/rag_project
  python scripts/generate_qa_dataset.py \\
      --input_dir data/gb_standards \\
      --output    data/qa_dataset/gb_qa.json \\
      --questions_per_doc 10 \\
      --max_docs 200

输出格式（RAGAS 兼容）：
  [
    {
      "question": "...",
      "ground_truth": "...",
      "reference_doc": "xxx.pdf",
      "source_text": "...",   ← 答案来源原文段落
      "question_type": "factual|reasoning|negative|comparison",
      "difficulty": "single_hop|multi_hop"
    },
    ...
  ]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────

LLM_MODEL = "qwen-plus"
REQUEST_INTERVAL = 1.5      # 请求间隔（秒），避免限流
MAX_DOC_TOKENS = 6000       # 每份文档发送的最大字符数（避免超出上下文）
RETRY_LIMIT = 3             # 单次出题失败最大重试次数

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

# ── Prompt ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个专业的石油化工领域知识问答测试集构建专家。
你的任务是根据提供的技术文档内容，生成高质量的问答对，用于评估 RAG 系统的检索和生成能力。"""

QA_PROMPT_TEMPLATE = """请根据以下技术文档内容，生成 {n_questions} 个问答对。

【文档内容】
{doc_text}

【出题要求】
请生成以下四种类型的问题（按比例分配）：
1. factual（事实抽取，{n_factual}题）：答案可以直接从文档某一段落找到
2. reasoning（推理综合，{n_reasoning}题）：需要综合文档多处信息才能回答
3. negative（否定/例外，{n_negative}题）：涉及"不适用/禁止/例外/不在范围"等情况
4. comparison（对比，{n_comparison}题）：涉及两个指标、工艺、条件的区别或对比

【格式要求】
严格按以下 JSON 数组格式输出，不要有任何其他文字：

[
  {{
    "question": "问题内容",
    "ground_truth": "标准答案（完整、准确，包含关键数值和单位）",
    "source_text": "答案所在的原文段落（原文引用，不超过200字）",
    "question_type": "factual|reasoning|negative|comparison",
    "difficulty": "single_hop|multi_hop"
  }}
]

【注意事项】
- 问题必须有明确答案，不能出"请介绍一下..."这类开放题
- ground_truth 必须忠于原文，不能凭空添加信息
- 如果文档内容不足以出某类题，可以适当调整各类题型数量
- 不要出关于文档编号、发布日期、起草单位等元信息的题目"""


# ── 文档文本提取 ──────────────────────────────────────────────────────────

def extract_text_from_content_list(content_list_path: Path, max_chars: int = MAX_DOC_TOKENS) -> str:
    """
    从 content_list.json（或 content_list_with_tables.json）提取文本。
    优先使用 VLM 增强版（含表格 OCR 结果）。
    """
    # 优先使用含表格 OCR 的版本
    enhanced = content_list_path.parent / "content_list_with_tables.json"
    target = enhanced if enhanced.exists() else content_list_path

    with open(target, encoding="utf-8") as f:
        blocks = json.load(f)

    texts = []
    total = 0
    for block in blocks:
        btype = block.get("type", "")
        text = block.get("text", "").strip()

        if not text:
            continue
        if btype == "image":
            continue  # 跳过无法转文字的图片块

        # 表格加标记，方便大模型理解这是结构化数据
        if btype == "table":
            text = f"[表格]\n{text}\n[/表格]"

        texts.append(text)
        total += len(text)
        if total >= max_chars:
            break

    return "\n\n".join(texts)


# ── 出题调用 ──────────────────────────────────────────────────────────────

def generate_qa(doc_text: str, n_questions: int = 10) -> list[dict]:
    """
    调用 qwen-plus 对单份文档出题，返回问答对列表。
    失败时自动重试 RETRY_LIMIT 次。
    """
    # 按比例分配题型
    n_factual    = max(1, round(n_questions * 0.3))
    n_reasoning  = max(1, round(n_questions * 0.3))
    n_negative   = max(1, round(n_questions * 0.2))
    n_comparison = n_questions - n_factual - n_reasoning - n_negative

    prompt = QA_PROMPT_TEMPLATE.format(
        doc_text=doc_text,
        n_questions=n_questions,
        n_factual=n_factual,
        n_reasoning=n_reasoning,
        n_negative=n_negative,
        n_comparison=n_comparison,
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,    # 低温，保证答案稳定
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content.strip()

            # 容错：截取 JSON 数组部分（模型有时会多输出解释文字）
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("响应中未找到 JSON 数组")

            qa_list = json.loads(raw[start:end])

            # 基本校验
            valid = []
            for qa in qa_list:
                if all(k in qa for k in ["question", "ground_truth", "source_text", "question_type"]):
                    valid.append(qa)
            return valid

        except Exception as e:
            print(f"    ⚠️ 第{attempt}次尝试失败：{e}")
            if attempt < RETRY_LIMIT:
                time.sleep(2)

    return []


# ── 主流程 ────────────────────────────────────────────────────────────────

def find_content_lists(input_dir: Path) -> list[tuple[str, Path]]:
    """
    在 input_dir 下递归查找所有 *_content_list.json 文件。
    返回 [(doc_name, path), ...]
    """
    results = []
    for p in sorted(input_dir.rglob("*_content_list.json")):
        doc_name = p.stem.replace("_content_list", "")
        results.append((doc_name, p))
    return results


def run(input_dir: str, output_path: str, questions_per_doc: int, max_docs: int):
    input_dir   = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有进度（断点续传）
    existing: list[dict] = []
    done_docs: set[str]  = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        done_docs = {qa["reference_doc"] for qa in existing}
        print(f"[断点续传] 已完成 {len(done_docs)} 份文档，共 {len(existing)} 条问答对")

    # 收集待处理文档
    all_docs = find_content_lists(input_dir)
    todo = [(name, path) for name, path in all_docs if name not in done_docs]
    todo = todo[:max_docs]

    print(f"\n待处理：{len(todo)} 份文档（共发现 {len(all_docs)} 份）")
    print(f"每份出题：{questions_per_doc} 题")
    print(f"预计 token 消耗：~{len(todo) * questions_per_doc * 400 // 1000}k\n")

    all_qa = list(existing)

    for idx, (doc_name, cl_path) in enumerate(todo):
        print(f"[{idx+1}/{len(todo)}] {doc_name[:50]}", end="  ")

        # 提取文档文本
        doc_text = extract_text_from_content_list(cl_path)
        if len(doc_text) < 200:
            print("⚠️ 文本过短，跳过")
            continue

        # 出题
        t0 = time.time()
        qa_list = generate_qa(doc_text, questions_per_doc)
        elapsed = time.time() - t0

        if not qa_list:
            print("❌ 出题失败，跳过")
            continue

        # 补充文档来源信息
        for qa in qa_list:
            qa["reference_doc"] = doc_name

        all_qa.extend(qa_list)
        print(f"✅ {len(qa_list)}题  {elapsed:.1f}s")

        # 实时保存（断点续传）
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=2)

        if idx < len(todo) - 1:
            time.sleep(REQUEST_INTERVAL)

    # 统计
    print(f"\n完成！共 {len(all_qa)} 条问答对，保存至 {output_path}")
    type_counts = {}
    for qa in all_qa:
        t = qa.get("question_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print("题型分布：", type_counts)


# ── 入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量出题：GB 标准文档 → RAG 测试集")
    parser.add_argument("--input_dir",        default="data/gb_standards/mineru_output",
                        help="MinerU 输出目录（含 *_content_list.json）")
    parser.add_argument("--output",           default="data/qa_dataset/gb_qa.json",
                        help="输出 JSON 路径")
    parser.add_argument("--questions_per_doc",type=int, default=10,
                        help="每份文档出题数量（默认 10）")
    parser.add_argument("--max_docs",         type=int, default=200,
                        help="最多处理多少份文档（默认 200）")
    args = parser.parse_args()

    run(args.input_dir, args.output, args.questions_per_doc, args.max_docs)
