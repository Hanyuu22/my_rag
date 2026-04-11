"""
动态出题脚本 v2：按文档长度分配题数，目标总量 2000 题

改进点（相比 generate_qa_dataset.py）：
  1. 动态题数：根据文档文本量自动分配 5~15 题，而非固定 10 题
  2. 目标总量：达到 --target 后自动停止
  3. 增加 topic_tag 字段：自动从问题内容判断主题分类
  4. 专业标注格式：符合 RAGAS + 人工标注双向兼容

输出格式（每条）：
  {
    "question":      "问题",
    "ground_truth":  "标准答案（含数值单位）",
    "source_text":   "原文引用（≤200字）",
    "question_type": "factual|reasoning|negative|comparison",
    "difficulty":    "single_hop|multi_hop",
    "topic_tag":     "工艺参数|安全规程|质量标准|设备操作|其他",
    "reference_doc": "文档名",
    "n_questions_allocated": 10   ← 该文档被分配的题数（便于分析）
  }

用法：
  conda activate mineru_2.5
  cd ~/rag_project

  python scripts/generate_qa_v2.py \\
      --input_dir data/raw/mineru_output \\
      --output    data/qa_dataset/gb_qa_2000.json \\
      --target    2000
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────

LLM_MODEL        = "qwen-plus"
REQUEST_INTERVAL = 1.5
MAX_DOC_CHARS    = 6000
RETRY_LIMIT      = 3

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


# ── 动态题数分配 ──────────────────────────────────────────────────────────────

def calc_n_questions(doc_text: str, target_total: int, remaining: int) -> int:
    """
    根据文档长度决定出题数，同时考虑剩余配额。
    短文档少出，长文档多出，不超过剩余配额。
    """
    length = len(doc_text)
    if length < 1500:
        n = 5
    elif length < 3000:
        n = 8
    elif length < 5000:
        n = 12
    else:
        n = 15
    return min(n, remaining)


# ── Topic 标签（出题后由 LLM 标注） ──────────────────────────────────────────

TOPIC_KEYWORDS = {
    "工艺参数":  ["温度", "压力", "流量", "浓度", "转化率", "收率", "反应", "催化"],
    "安全规程":  ["安全", "危险", "禁止", "警告", "防护", "事故", "应急", "泄漏"],
    "质量标准":  ["指标", "标准", "规格", "限值", "合格", "检验", "测定", "分析"],
    "设备操作":  ["操作", "启动", "停车", "维修", "检修", "设备", "仪表", "阀门"],
}

def auto_tag_topic(question: str, answer: str) -> str:
    text = question + answer
    for tag, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return tag
    return "其他"


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个专业的石油化工领域知识问答测试集构建专家。
根据技术文档生成高质量问答对，用于评估 RAG 系统的检索和生成能力。"""

QA_PROMPT_TEMPLATE = """根据以下石油化工技术文档，生成 {n_questions} 个问答对。

【文档内容】
{doc_text}

【题型分配】
- factual（事实抽取，{n_factual}题）：答案可直接从文档某段找到
- reasoning（推理综合，{n_reasoning}题）：需综合文档多处信息
- negative（否定/例外，{n_negative}题）：涉及"不适用/禁止/例外/不在范围"
- comparison（对比，{n_comparison}题）：涉及两个指标/工艺/条件的区别

【格式要求】
严格输出 JSON 数组，不要有任何其他文字：

[
  {{
    "question":      "问题（必须有明确答案，不出开放题）",
    "ground_truth":  "标准答案（完整准确，包含关键数值和单位）",
    "source_text":   "答案所在原文段落（原文引用，≤200字）",
    "question_type": "factual|reasoning|negative|comparison",
    "difficulty":    "single_hop|multi_hop"
  }}
]

【注意】
- ground_truth 必须忠于原文，不能凭空添加
- 不出文档编号、发布日期、起草单位等元信息题目
- 如内容不足以出某类题，可调整各类比例"""


# ── 文档文本提取 ──────────────────────────────────────────────────────────────

def extract_text(content_list_path: Path, max_chars: int = MAX_DOC_CHARS) -> str:
    """
    提取文档文本，随机选取起始位置覆盖文档不同部分。
    短文档（总字数 < max_chars * 1.5）直接取全文。
    长文档随机取一段，避免永远只出前几页的题。
    """
    enhanced = content_list_path.parent / "content_list_with_tables.json"
    target = enhanced if enhanced.exists() else content_list_path

    with open(target, encoding="utf-8") as f:
        blocks = json.load(f)

    # 先提取所有有效 block 的文本
    all_texts = []
    for block in blocks:
        if block.get("type") == "image":
            continue
        text = block.get("text", "").strip()
        if not text:
            continue
        if block.get("type") == "table":
            text = f"[表格]\n{text}\n[/表格]"
        all_texts.append(text)

    full_text = "\n\n".join(all_texts)
    total_len = len(full_text)

    # 短文档直接返回全文
    if total_len <= max_chars * 1.5:
        return full_text[:max_chars]

    # 长文档：随机选取起始位置（留出末尾，确保能取够 max_chars）
    max_start = total_len - max_chars
    start = random.randint(0, max_start)
    # 对齐到段落边界（找最近的换行符），避免从句子中间开始
    newline_pos = full_text.find("\n", start)
    if newline_pos != -1 and newline_pos - start < 200:
        start = newline_pos + 1
    return full_text[start: start + max_chars]


# ── 出题调用 ──────────────────────────────────────────────────────────────────

def generate_qa(doc_text: str, n_questions: int) -> list[dict]:
    n_factual    = max(1, round(n_questions * 0.30))
    n_reasoning  = max(1, round(n_questions * 0.30))
    n_negative   = max(1, round(n_questions * 0.20))
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
                temperature=0.3,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content.strip()
            start, end = raw.find("["), raw.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("未找到 JSON 数组")
            json_str = raw[start:end]
            # 修复1：LaTeX 反斜杠转义问题（\sqrt → \\sqrt）
            json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)
            # 修复2：中文弯引号替换为直引号，避免破坏 JSON 字符串边界
            json_str = json_str.replace('\u201c', '\\"').replace('\u201d', '\\"')
            qa_list = json.loads(json_str)
            return [qa for qa in qa_list
                    if all(k in qa for k in ["question", "ground_truth", "source_text", "question_type"])]
        except Exception as e:
            print(f"    ⚠️ 第{attempt}次失败：{e}")
            if attempt < RETRY_LIMIT:
                time.sleep(2)
    return []


# ── 主流程 ────────────────────────────────────────────────────────────────────

def find_content_lists(input_dir: Path) -> list[tuple[str, Path]]:
    results = []
    for p in sorted(input_dir.rglob("*_content_list.json")):
        doc_name = p.stem.replace("_content_list", "")
        results.append((doc_name, p))
    return results


def run(input_dir: str, output_path: str, target: int):
    input_dir   = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传
    existing: list[dict] = []
    done_docs: set[str]  = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        done_docs = {qa["reference_doc"] for qa in existing}
        print(f"[断点续传] 已完成 {len(done_docs)} 份，共 {len(existing)} 条")

    all_docs = find_content_lists(input_dir)
    todo = [(name, path) for name, path in all_docs if name not in done_docs]
    remaining = target - len(existing)

    print(f"\n目标总题数：{target}")
    print(f"已有题数：{len(existing)}")
    print(f"还需：{remaining} 题")
    print(f"待处理文档：{len(todo)} 份\n")

    if remaining <= 0:
        print("已达到目标题数，无需继续。")
        return

    all_qa = list(existing)

    for idx, (doc_name, cl_path) in enumerate(todo):
        if remaining <= 0:
            print(f"\n已达到 {target} 题目标，停止。")
            break

        doc_text = extract_text(cl_path)
        if len(doc_text) < 300:
            print(f"[{idx+1}] {doc_name[:45]}  ⚠️ 文本过短，跳过")
            continue

        n = calc_n_questions(doc_text, target, remaining)
        print(f"[{idx+1}/{len(todo)}] {doc_name[:45]}  文本{len(doc_text)}字 → 出{n}题", end="  ")

        t0 = time.time()
        qa_list = generate_qa(doc_text, n)
        elapsed = time.time() - t0

        if not qa_list:
            print("❌ 出题失败")
            continue

        # 补充元数据
        for qa in qa_list:
            qa["reference_doc"] = doc_name
            qa["topic_tag"] = auto_tag_topic(qa["question"], qa["ground_truth"])
            qa["n_questions_allocated"] = n

        all_qa.extend(qa_list)
        remaining -= len(qa_list)
        print(f"✅ {len(qa_list)}题  {elapsed:.1f}s  (累计{len(all_qa)}/{target})")

        # 实时保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=2)

        if idx < len(todo) - 1 and remaining > 0:
            time.sleep(REQUEST_INTERVAL)

    # 统计报告
    print(f"\n{'='*55}")
    print(f"完成！共 {len(all_qa)} 条，保存至 {output_path}")

    type_counts  = {}
    diff_counts  = {}
    topic_counts = {}
    for qa in all_qa:
        t = qa.get("question_type", "unknown")
        d = qa.get("difficulty",    "unknown")
        k = qa.get("topic_tag",     "其他")
        type_counts[t]  = type_counts.get(t,  0) + 1
        diff_counts[d]  = diff_counts.get(d,  0) + 1
        topic_counts[k] = topic_counts.get(k, 0) + 1

    print("\n题型分布：",  type_counts)
    print("难度分布：",    diff_counts)
    print("主题分布：",    topic_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动态出题：GB 标准文档 → 2000 题测试集")
    parser.add_argument("--input_dir", default="data/raw/mineru_output", help="MinerU 输出目录")
    parser.add_argument("--output",    default="data/qa_dataset/gb_qa_2000.json", help="输出路径")
    parser.add_argument("--target",    type=int, default=2000, help="目标总题数（默认 2000）")
    args = parser.parse_args()

    run(args.input_dir, args.output, args.target)
