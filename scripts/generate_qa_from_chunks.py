"""
从 Chroma collection chunks 生成 QA 数据集

每个 chunk 作为一个上下文，调用 Qwen 生成问答对。
ground_truth_chunk_id 直接取该 chunk 的 chunk_id，无需事后标注。

用法：
    python scripts/generate_qa_from_chunks.py \
        --collection hydro_crack \
        --output data/qa_dataset/hydro_crack_qa.json \
        --questions_per_chunk 2

    # 跳过 block_type=table/equation 的 chunk（只用文本）
    python scripts/generate_qa_from_chunks.py \
        --collection hydro_crack \
        --output data/qa_dataset/hydro_crack_qa.json \
        --text_only

    # 断点续传：已有输出文件时跳过已完成的 chunk
    python scripts/generate_qa_from_chunks.py \
        --collection hydro_crack \
        --output data/qa_dataset/hydro_crack_qa.json \
        --resume
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

LLM_MODEL        = "qwen-plus"
RETRY_LIMIT      = 3
REQUEST_INTERVAL = 1.5
MIN_CHUNK_CHARS  = 80   # 太短的 chunk 跳过（无法生成有意义的题）

SYSTEM_PROMPT = (
    "你是一个专业的石油化工领域问答数据集构建专家。"
    "请根据给定的文档片段，生成高质量的问答对。"
    "要求：\n"
    "1. 问题必须能从片段中直接找到答案，不要超出文本范围\n"
    "2. 答案要准确、简洁，包含关键数值和单位\n"
    "3. 只输出 JSON，不要任何解释"
)

QA_PROMPT = """\
请根据以下文档片段生成 {n} 个问答对。

题型分配（尽量按比例，不够时可只取部分类型）：
- factual（事实查询，参数/数值/定义）：优先
- reasoning（需要理解推断）：其次
- negative（判断某条件是否适用/不适用）：最后

输出格式：严格 JSON 对象，包含 items 数组，每个元素字段：
- question: 问题
- ground_truth: 标准答案（含关键数值和单位）
- source_text: 原文引用（≤120字，直接摘录，不要修改原文）
- question_type: factual / reasoning / negative
- difficulty: single_hop / multi_hop

示例：
{{"items": [{{"question": "...", "ground_truth": "...", "source_text": "...", "question_type": "factual", "difficulty": "single_hop"}}]}}

文档片段：

{chunk_text}
"""


def _call_llm(client: OpenAI, chunk_text: str, n: int) -> list[dict]:
    prompt = QA_PROMPT.format(n=n, chunk_text=chunk_text[:1500])
    for attempt in range(RETRY_LIMIT):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            # 去掉 markdown 代码块
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            items = data.get("items", data) if isinstance(data, dict) else data
            return items if isinstance(items, list) else []
        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(2)
            else:
                print(f"    [失败] {e}")
    return []


def load_chunks_from_collection(collection_name: str) -> list[dict]:
    """从 Chroma collection 读取所有 chunk（含 metadata 和 document）。"""
    import chromadb
    from config import CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    col = client.get_collection(collection_name)
    result = col.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        chunks.append({"text": doc, "metadata": meta})
    return chunks


def run(collection: str, output_path: str, questions_per_chunk: int,
        text_only: bool, resume: bool):
    output_path = Path(output_path)
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    # 断点续传：读取已有输出
    existing: list[dict] = []
    done_chunk_ids: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        done_chunk_ids = {q["ground_truth_chunk_id"] for q in existing
                          if q.get("ground_truth_chunk_id")}
        print(f"断点续传：已有 {len(existing)} 条，跳过 {len(done_chunk_ids)} 个 chunk")

    print(f"加载 collection: {collection}")
    chunks = load_chunks_from_collection(collection)
    print(f"共 {len(chunks)} 个 chunk")

    # 过滤
    if text_only:
        chunks = [c for c in chunks if c["metadata"].get("block_type") == "text"]
        print(f"  仅文本 chunk：{len(chunks)} 个")

    results = list(existing)
    skip_count = 0
    gen_count  = 0

    for i, chunk in enumerate(chunks):
        chunk_id   = chunk["metadata"].get("chunk_id", f"chunk_{i}")
        chunk_text = chunk["text"].strip()
        page       = chunk["metadata"].get("page", -1)
        block_type = chunk["metadata"].get("block_type", "text")

        if chunk_id in done_chunk_ids:
            skip_count += 1
            continue

        if len(chunk_text) < MIN_CHUNK_CHARS:
            skip_count += 1
            continue

        print(f"  [{i+1}/{len(chunks)}] page={page} type={block_type} "
              f"len={len(chunk_text)} ...", end=" ", flush=True)

        items = _call_llm(client, chunk_text, questions_per_chunk)
        if not items:
            print("跳过")
            skip_count += 1
            time.sleep(REQUEST_INTERVAL)
            continue

        for item in items:
            item["ground_truth_chunk_id"] = chunk_id
            item["page"]       = page
            item["block_type"] = block_type
            item["collection"] = collection
            item["category"]   = block_type
            results.append(item)

        gen_count += 1
        print(f"生成 {len(items)} 条")

        # 每 20 个 chunk 保存一次（断点续传）
        if gen_count % 20 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(REQUEST_INTERVAL)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成：生成 {len(results)} 条问答，跳过 {skip_count} 个 chunk")
    print(f"输出：{output_path}")

    # 统计
    from collections import Counter
    types = Counter(q.get("question_type", "?") for q in results)
    print(f"题型分布：{dict(types)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection",          default="hydro_crack")
    parser.add_argument("--output",              default="data/qa_dataset/hydro_crack_qa.json")
    parser.add_argument("--questions_per_chunk", type=int, default=2)
    parser.add_argument("--text_only",           action="store_true")
    parser.add_argument("--resume",              action="store_true")
    args = parser.parse_args()

    run(args.collection, args.output, args.questions_per_chunk,
        args.text_only, args.resume)


if __name__ == "__main__":
    main()
