"""
RAPTOR 摘要节点构建器

原理：
  对已入库 collection 的每个文档，将其所有 chunk 按顺序拼接，
  调用 LLM 生成 1~3 个段落级摘要节点，追加回同一 collection。
  摘要节点带有 node_type="raptor_summary" 元数据，便于评估时区分。

特性：
  - 不需要重新入库原始文档
  - 按文档分组：同一 source 文件的 chunks 聚在一起生成摘要
  - 长文档自动分段摘要（每段最多 3000 字）
  - 摘要节点 chunk_id 格式：raptor_{source}_{i}
  - 断点续传：已有同名摘要节点的文档跳过

用法：
  conda activate mineru_2.5
  cd ~/rag_project
  python splitters/raptor_builder.py --collection gb_standards_512
  python splitters/raptor_builder.py --collection gb_standards_1024
"""

import sys
import uuid
import time
from typing import List

sys.path.insert(0, "/home/hanyuu/rag_project")


# ── LLM 调用 ──────────────────────────────────────────────────────────────

def _build_llm_client():
    from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL
    from openai import OpenAI
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


SUMMARY_SYSTEM = "你是一个专业的技术文档摘要专家，擅长提炼石油化工领域标准文档的核心内容。"

SUMMARY_PROMPT = """请对以下石油化工技术文档片段生成一段简洁的摘要（150~300字），
要求：
1. 涵盖片段中的主要技术指标、参数范围和关键规定
2. 保留具体数值和单位（如温度、压力、浓度等）
3. 用第三人称陈述，不加主观评价
4. 输出纯文字，不要 Markdown 格式

【文档片段】
{text}

【摘要】"""


def _call_llm_summary(client, text: str, retries: int = 3) -> str:
    prompt = SUMMARY_PROMPT.format(text=text[:3000])
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    ⚠️ LLM 调用失败（第{attempt}次）：{e}")
            if attempt < retries:
                time.sleep(2)
    return ""


# ── 核心逻辑 ──────────────────────────────────────────────────────────────

def build_raptor_nodes(
    collection_name: str,
    max_chunk_chars: int = 3000,   # 每个摘要节点覆盖的原文字数上限
    request_interval: float = 1.0, # LLM 调用间隔（s）
):
    """
    从 collection 中读取所有 chunks，按文档分组生成摘要节点，追加回同一 collection。
    """
    import chromadb
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from chains.rag_chain import get_embeddings, get_collection_embedding_model
    from config import CHROMA_PERSIST_DIR

    # 加载向量库
    model_name = get_collection_embedding_model(collection_name)
    embeddings  = get_embeddings(model_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    collection = vectorstore._collection

    # 读取所有已有文档（含 RAPTOR 节点）
    all_data = collection.get(include=["documents", "metadatas"])
    all_docs      = all_data["documents"]
    all_metas     = all_data["metadatas"]
    total         = len(all_docs)
    print(f"Collection '{collection_name}' 共 {total} 条\n")

    # 找出已有 RAPTOR 摘要节点所对应的 source，跳过重复
    done_sources: set[str] = set()
    source_chunks: dict[str, list[str]] = {}   # source → [text, ...]

    for doc, meta in zip(all_docs, all_metas):
        node_type = meta.get("node_type", "")
        source    = meta.get("source", meta.get("file_name", "unknown"))
        if node_type == "raptor_summary":
            done_sources.add(source)
            continue
        # 收集普通 chunk，保持顺序（按 chunk_id 排序会更准确，这里用追加顺序近似）
        source_chunks.setdefault(source, []).append(doc)

    all_sources = sorted(source_chunks.keys())
    todo_sources = [s for s in all_sources if s not in done_sources]
    print(f"文档总数：{len(all_sources)}，已生成摘要：{len(done_sources)}，待处理：{len(todo_sources)}")

    if not todo_sources:
        print("全部文档已有摘要节点，跳过。")
        return

    llm_client  = _build_llm_client()
    added_total = 0

    for i, source in enumerate(todo_sources):
        chunks_text = source_chunks[source]
        # 按 max_chunk_chars 将所有 chunk 拼成若干段，每段生成一个摘要
        segments = _split_into_segments(chunks_text, max_chunk_chars)
        source_short = source[:50]
        print(f"\n[{i+1}/{len(todo_sources)}] {source_short}  "
              f"({len(chunks_text)} chunks → {len(segments)} 段摘要)")

        summary_docs  = []
        summary_metas = []

        for seg_i, seg_text in enumerate(segments):
            summary = _call_llm_summary(llm_client, seg_text)
            if not summary:
                print(f"  段 {seg_i+1}/{len(segments)} 摘要生成失败，跳过")
                continue

            chunk_id = f"raptor_{_safe_id(source)}_{seg_i}"
            summary_docs.append(summary)
            summary_metas.append({
                "source":      source,
                "chunk_id":    chunk_id,
                "node_type":   "raptor_summary",
                "seg_index":   seg_i,
                "n_segs":      len(segments),
                "orig_chunks": len(chunks_text),
            })
            print(f"  段 {seg_i+1}/{len(segments)}  {len(summary)}字摘要  ✅")
            if seg_i < len(segments) - 1:
                time.sleep(request_interval)

        if summary_docs:
            # 追加到 Chroma（不重建，直接 add）
            vectorstore.add_texts(texts=summary_docs, metadatas=summary_metas)
            added_total += len(summary_docs)
            print(f"  → 追加 {len(summary_docs)} 个摘要节点")

    print(f"\n{'='*55}")
    print(f"RAPTOR 完成：共追加 {added_total} 个摘要节点")
    print(f"Collection '{collection_name}' 当前总量：{collection.count()}")


def _split_into_segments(chunks: List[str], max_chars: int) -> List[str]:
    """将 chunk 列表按字数上限合并成若干段"""
    segments = []
    current  = []
    current_len = 0
    for c in chunks:
        if current_len + len(c) > max_chars and current:
            segments.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(c)
        current_len += len(c)
    if current:
        segments.append("\n\n".join(current))
    return segments


def _safe_id(source: str) -> str:
    """将 source 文件名转成安全的 ID 前缀"""
    return source.replace("/", "_").replace("\\", "_").replace(" ", "_")[:40]


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAPTOR：为已有 collection 追加摘要节点")
    parser.add_argument("--collection",       default="gb_standards_512",  help="目标 collection")
    parser.add_argument("--max_chunk_chars",  type=int, default=3000,       help="每段摘要覆盖原文字数上限")
    parser.add_argument("--interval",         type=float, default=1.0,      help="LLM 调用间隔（秒）")
    args = parser.parse_args()

    build_raptor_nodes(
        collection_name=args.collection,
        max_chunk_chars=args.max_chunk_chars,
        request_interval=args.interval,
    )
