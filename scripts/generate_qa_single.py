"""
单文档逐页 QA 生成脚本

针对单个 MinerU 解析后的文档，按页分组文本，
每页调用 Qwen 生成若干问答对，支持断点续传。

用法：
    python scripts/generate_qa_single.py \
        --content_list "data/raw/mineru_output/【保密】.../auto/content_list_with_tables.json" \
        --output data/qa_dataset/hydro_reg_qa.json \
        --questions_per_page 4
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

LLM_MODEL        = "qwen-plus"
RETRY_LIMIT      = 3
REQUEST_INTERVAL = 2.0   # 请求间隔（秒），避免触发限速
MIN_PAGE_CHARS   = 150   # 页面文本少于此值时与下一页合并

SYSTEM_PROMPT = (
    "你是一个专业的石油化工领域问答数据集构建专家。"
    "请根据给定的文档段落，生成高质量的问答对。"
    "要求：\n"
    "1. 问题必须能从段落中直接找到答案，不要超出文本范围\n"
    "2. 答案要准确、简洁，包含关键数值和单位\n"
    "3. 问题要多样化，覆盖不同类型\n"
    "4. 只输出 JSON 数组，不要任何解释"
)

QA_PROMPT = """\
请根据以下文档段落（第{page}页）生成 {n} 个问答对。

题型分配：
- factual（事实查询，参数/数值/定义）：{n_factual} 个
- reasoning（需要理解推断）：{n_reasoning} 个
- negative（判断某条件是否适用/不适用）：{n_negative} 个

输出格式：严格 JSON 对象，包含 items 数组，每个元素字段：
- question: 问题
- ground_truth: 标准答案（含关键数值和单位）
- source_text: 原文引用（≤150字，数学公式用文字替代，如 CH3、H2S，不要使用 LaTeX 或反斜杠）
- question_type: factual / reasoning / negative
- difficulty: single_hop / multi_hop

示例：
{{"items": [{{"question": "...", "ground_truth": "...", "source_text": "...", "question_type": "factual", "difficulty": "single_hop"}}]}}

文档段落：
{text}
"""


# ── 工具函数 ───────────────────────────────────────────────────────────────

def load_content_list(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_page(blocks: list) -> dict:
    """将 blocks 按 page_idx 分组，只保留 text 和 table 类型"""
    pages: dict[int, list[str]] = {}
    for block in blocks:
        if block.get("type") not in ("text", "table"):
            continue
        page = block.get("page_idx", 0)
        text = block.get("text", "").strip()
        if not text:
            continue
        pages.setdefault(page, []).append(text)
    return pages


def build_page_segments(pages: dict, min_chars: int = MIN_PAGE_CHARS) -> list[dict]:
    """
    将页面文本整合为段落列表。
    内容太少的页面与下一页合并，避免信息量不足。
    返回: [{"pages": [0,1], "text": "..."}]
    """
    segments = []
    sorted_pages = sorted(pages.keys())
    i = 0
    while i < len(sorted_pages):
        page = sorted_pages[i]
        text = "\n".join(pages[page])
        # 内容太少则向后合并
        while len(text) < min_chars and i + 1 < len(sorted_pages):
            i += 1
            next_page = sorted_pages[i]
            text = text + "\n" + "\n".join(pages[next_page])
            page = next_page  # 以最后合并页作为代表页
        if len(text.strip()) >= 50:  # 过滤掉几乎空白的页
            segments.append({"page": page, "text": text.strip()})
        i += 1
    return segments


def calc_type_split(n: int) -> tuple[int, int, int]:
    """计算题型分配：factual / reasoning / negative"""
    n_factual  = max(1, round(n * 0.5))
    n_reasoning = max(0, round(n * 0.3))
    n_negative  = max(0, n - n_factual - n_reasoning)
    return n_factual, n_reasoning, n_negative


def parse_json_from_response(text: str, debug: bool = False) -> list:
    """从 LLM 回复中提取问答列表，支持 {items:[...]} 和裸数组两种格式"""
    text = text.strip()
    # 修复常见问题
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # 中文引号
    # 将非法 JSON 转义（如 \m \s \( 等 LaTeX 残留）替换为空，保留合法转义
    text = re.sub(r'\\(?!["\\/bfnrtu0-9])', '', text)

    # 优先尝试解析完整 JSON（{items:[...]} 格式）
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "items" in obj:
            return obj["items"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # 降级：提取 [ ... ] 数组片段
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    if debug:
        print(f"\n    [DEBUG 实际返回前300字]: {text[:300]}\n")
    return []


def auto_tag_topic(question: str, answer: str) -> str:
    text = question + answer
    if any(k in text for k in ["温度", "压力", "流量", "浓度", "密度", "参数", "操作条件"]):
        return "工艺参数"
    if any(k in text for k in ["安全", "危险", "防护", "应急", "泄漏", "火灾"]):
        return "安全规程"
    if any(k in text for k in ["质量", "指标", "标准", "规格", "合格"]):
        return "质量标准"
    if any(k in text for k in ["设备", "装置", "阀门", "泵", "换热器", "反应器"]):
        return "设备操作"
    return "其他"


# ── LLM 调用 ───────────────────────────────────────────────────────────────

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def generate_qa(page: int, text: str, n: int) -> list:
    """对单个页面段落调用 LLM 生成 n 个 QA，带重试"""
    n_factual, n_reasoning, n_negative = calc_type_split(n)
    prompt = QA_PROMPT.format(
        page=page + 1,  # 显示给 LLM 时从 1 开始
        n=n,
        n_factual=n_factual,
        n_reasoning=n_reasoning,
        n_negative=n_negative,
        text=text[:4000],  # 防止超出 token 限制
    )
    for attempt in range(RETRY_LIMIT):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            items = parse_json_from_response(raw, debug=(attempt == RETRY_LIMIT - 1))
            if items:
                return items
            print(f"  ⚠ 第{page+1}页解析失败，重试({attempt+1}/{RETRY_LIMIT})")
        except Exception as e:
            print(f"  ⚠ 第{page+1}页请求出错：{e}，重试({attempt+1}/{RETRY_LIMIT})")
        time.sleep(2)
    return []


# ── 主流程 ─────────────────────────────────────────────────────────────────

def run(content_list_path: str, output_path: str, questions_per_page: int):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：加载已完成的 QA 和已处理的页号
    qa_list: list = []
    done_pages: set = set()
    if output.exists():
        with open(output, encoding="utf-8") as f:
            qa_list = json.load(f)
        done_pages = {item["page"] for item in qa_list if "page" in item}
        print(f"断点续传：已有 {len(qa_list)} 条，已处理页：{sorted(done_pages)}")

    # 读取 content_list，按页分组
    blocks   = load_content_list(content_list_path)
    pages    = group_by_page(blocks)
    segments = build_page_segments(pages)
    doc_name = Path(content_list_path).parent.parent.name  # 取文档目录名

    total_segs = len(segments)
    remaining  = [s for s in segments if s["page"] not in done_pages]
    print(f"文档：{doc_name}")
    print(f"有效段落：{total_segs} 个，待处理：{len(remaining)} 个")
    print(f"预计生成：{len(remaining) * questions_per_page} 条\n")

    for idx, seg in enumerate(remaining):
        page = seg["page"]
        text = seg["text"]
        print(f"[{idx+1}/{len(remaining)}] 第{page+1}页（{len(text)}字）", end="  ")

        items = generate_qa(page, text, questions_per_page)
        if not items:
            print("跳过（无有效输出）")
            continue

        # 补充元数据
        for item in items:
            item["page"]          = page
            item["reference_doc"] = doc_name
            item["topic_tag"]     = auto_tag_topic(
                item.get("question", ""), item.get("ground_truth", "")
            )
            # 兼容性：确保字段存在
            item.setdefault("question_type", "factual")
            item.setdefault("difficulty",    "single_hop")
            item.setdefault("source_text",   "")

        qa_list.extend(items)
        print(f"生成 {len(items)} 条，累计 {len(qa_list)} 条")

        # 实时保存
        with open(output, "w", encoding="utf-8") as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=2)

        if idx < len(remaining) - 1:
            time.sleep(REQUEST_INTERVAL)

    # 统计
    print(f"\n完成！共 {len(qa_list)} 条")
    from collections import Counter
    print("题型分布:", dict(Counter(q.get("question_type", "") for q in qa_list)))
    print("难度分布:", dict(Counter(q.get("difficulty", "")    for q in qa_list)))
    print("主题分布:", dict(Counter(q.get("topic_tag", "")     for q in qa_list)))
    print(f"输出文件：{output}")


# ── 入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单文档逐页 QA 生成")
    parser.add_argument(
        "--content_list",
        default=(
            "data/raw/mineru_output/"
            "【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节/"
            "auto/content_list_with_tables.json"
        ),
        help="MinerU 输出的 content_list_with_tables.json 路径",
    )
    parser.add_argument(
        "--output",
        default="data/qa_dataset/hydro_reg_qa.json",
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--questions_per_page",
        type=int,
        default=4,
        help="每页生成题数（默认 4）",
    )
    args = parser.parse_args()
    run(args.content_list, args.output, args.questions_per_page)
