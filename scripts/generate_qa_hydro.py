"""
加氢裂化工艺规程 QA 生成脚本

将长文档按固定步长切成多段，每段出题，目标 200 题。
断点续传：已生成的段跳过。

用法：
  conda activate mineru_2.5
  cd ~/rag_project
  python scripts/generate_qa_hydro.py \
      --output data/qa_dataset/hydro_qa_200.json \
      --target 200
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────
CONTENT_LIST_PATH = (
    "data/mineru_output/【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节"
    "/auto/content_list_with_tables.json"
)
LLM_MODEL        = "qwen-plus"
SEGMENT_CHARS    = 5000    # 每段字数
STEP_CHARS       = 4000    # 滑动步长（重叠 1000 字，避免边界截断）
N_PER_SEGMENT    = 15      # 每段出题数
REQUEST_INTERVAL = 1.5
RETRY_LIMIT      = 3

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

# ── Prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的石油化工领域知识问答测试集构建专家，
擅长根据加氢裂化装置工艺技术规程生成高质量问答对，用于评估 RAG 系统的检索和生成能力。"""

QA_PROMPT_TEMPLATE = """根据以下加氢裂化装置工艺技术规程片段，生成 {n} 个问答对。

【文档片段】
{text}

【题型分配】
- factual（事实抽取，{n_factual}题）：答案可直接从文档某段找到，如参数数值、操作步骤
- reasoning（推理综合，{n_reasoning}题）：需综合文档多处信息，如工艺条件之间的关联
- negative（禁止/例外，{n_negative}题）：涉及"不允许/禁止/不应/不得"等约束条件
- comparison（对比，{n_comparison}题）：涉及不同工况、参数或操作方式的比较

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
- 优先出与工艺参数（温度、压力、流量、氢油比）相关的题目
- 不出文档编号、发布日期、起草单位等元信息题目"""


TOPIC_KEYWORDS = {
    "工艺参数":  ["温度", "压力", "流量", "浓度", "氢油比", "空速", "转化率", "收率", "反应"],
    "安全规程":  ["安全", "危险", "禁止", "警告", "防护", "事故", "应急", "泄漏", "不得", "不允许"],
    "质量标准":  ["指标", "标准", "规格", "限值", "合格", "检验", "测定", "分析"],
    "设备操作":  ["操作", "启动", "停车", "维修", "检修", "设备", "仪表", "阀门", "开工", "停工"],
}

def auto_tag(question: str, answer: str) -> str:
    text = question + answer
    for tag, kws in TOPIC_KEYWORDS.items():
        if any(k in text for k in kws):
            return tag
    return "其他"


# ── 文档分段 ──────────────────────────────────────────────────────────────

def load_segments(content_list_path: str, seg_chars: int, step_chars: int) -> list[dict]:
    path = Path(content_list_path)
    with open(path, encoding="utf-8") as f:
        blocks = json.load(f)

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
    total = len(full_text)
    print(f"文档总字数：{total}")

    segments = []
    start = 0
    seg_idx = 0
    while start < total:
        end = min(start + seg_chars, total)
        seg_text = full_text[start:end]
        # 只取有实质内容的段（过滤纯空白 / 纯标点）
        if len(seg_text.strip()) > 200:
            segments.append({"seg_idx": seg_idx, "text": seg_text, "start": start, "end": end})
            seg_idx += 1
        start += step_chars

    print(f"切分为 {len(segments)} 段（步长 {step_chars} 字，段长 {seg_chars} 字）")
    return segments


# ── LLM 出题 ──────────────────────────────────────────────────────────────

def generate_qa(seg_text: str, n: int) -> list[dict]:
    n_f = max(1, round(n * 0.35))
    n_r = max(1, round(n * 0.25))
    n_n = max(1, round(n * 0.20))
    n_c = n - n_f - n_r - n_n

    prompt = QA_PROMPT_TEMPLATE.format(
        text=seg_text, n=n,
        n_factual=n_f, n_reasoning=n_r, n_negative=n_n, n_comparison=n_c
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
            s, e = raw.find("["), raw.rfind("]") + 1
            if s == -1 or e == 0:
                raise ValueError("未找到 JSON 数组")
            json_str = raw[s:e]
            json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)
            json_str = json_str.replace('\u201c', '\\"').replace('\u201d', '\\"')
            qa_list = json.loads(json_str)
            return [qa for qa in qa_list
                    if all(k in qa for k in ["question", "ground_truth", "source_text", "question_type"])]
        except Exception as ex:
            print(f"    ⚠️ 第{attempt}次失败：{ex}")
            if attempt < RETRY_LIMIT:
                time.sleep(2)
    return []


# ── 主流程 ────────────────────────────────────────────────────────────────

def run(output_path: str, target: int):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传
    existing: list[dict] = []
    done_segs: set[int]  = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        done_segs = {qa["seg_idx"] for qa in existing if "seg_idx" in qa}
        print(f"[断点续传] 已有 {len(existing)} 条，已处理段：{sorted(done_segs)}")

    segments = load_segments(CONTENT_LIST_PATH, SEGMENT_CHARS, STEP_CHARS)
    todo = [s for s in segments if s["seg_idx"] not in done_segs]
    remaining = target - len(existing)

    print(f"\n目标：{target} 题，已有：{len(existing)} 题，还需：{remaining} 题")
    print(f"待处理段：{len(todo)}/{len(segments)}\n")

    if remaining <= 0:
        print("已达到目标题数。")
        return

    all_qa = list(existing)

    for s in todo:
        if remaining <= 0:
            break
        n = min(N_PER_SEGMENT, remaining)
        print(f"[段{s['seg_idx']}] 字符 {s['start']}~{s['end']}  出 {n} 题 ...", end="  ")
        t0 = time.time()
        qa_list = generate_qa(s["text"], n)
        elapsed = time.time() - t0

        if not qa_list:
            print("❌ 出题失败")
            continue

        for qa in qa_list:
            qa["seg_idx"]     = s["seg_idx"]
            qa["reference_doc"] = "240万吨加氢裂化装置工艺技术规程"
            qa["topic_tag"]   = auto_tag(qa["question"], qa["ground_truth"])

        all_qa.extend(qa_list)
        remaining -= len(qa_list)
        print(f"✅ {len(qa_list)} 题  {elapsed:.1f}s  （累计 {len(all_qa)}/{target}）")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=2)

        if remaining > 0:
            time.sleep(REQUEST_INTERVAL)

    # 统计
    print(f"\n{'='*55}")
    print(f"完成！共 {len(all_qa)} 条，保存至 {output_path}")
    type_counts  = {}
    topic_counts = {}
    for qa in all_qa:
        t = qa.get("question_type", "?")
        k = qa.get("topic_tag", "其他")
        type_counts[t]  = type_counts.get(t, 0) + 1
        topic_counts[k] = topic_counts.get(k, 0) + 1
    print("题型：", type_counts)
    print("主题：", topic_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/qa_dataset/hydro_qa_200.json")
    parser.add_argument("--target", type=int, default=200)
    args = parser.parse_args()
    run(args.output, args.target)
