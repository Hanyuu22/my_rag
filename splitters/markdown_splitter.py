"""
Markdown 结构化切分器

策略：
1. 先按 Markdown 标题层级切（MarkdownHeaderTextSplitter）
   → 每个 chunk 知道自己属于哪个章节
2. 超长段落再按字符数二次切（RecursiveCharacterTextSplitter）
   → 保证每块不超过 CHUNK_SIZE
3. table / equation 类型不切分，整块保留
   → 避免行列关系被破坏

输入：List[Document]（来自 MinerULoader）
输出：List[Document]（chunk 级别，metadata 更丰富）
"""

import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")
from config import CHUNK_SIZE, CHUNK_OVERLAP


# Markdown 标题层级定义
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def split_documents(docs: List[Document]) -> List[Document]:
    """
    对 MinerULoader 输出的 Document 列表做结构化切分。

    - text 类型：先按标题切，再按长度切
    - table / equation 类型：整块保留，不切分
    """
    text_docs = [d for d in docs if d.metadata.get("block_type") == "text"]
    preserve_docs = [d for d in docs if d.metadata.get("block_type") in ("table", "equation")]

    # ── Step 1：按 Markdown 标题切分 text 块 ──────────────────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,   # 保留标题行，chunk 里能看到自己的章节
    )

    header_chunks: List[Document] = []
    for doc in text_docs:
        splits = header_splitter.split_text(doc.page_content)
        for chunk in splits:
            # 合并原始 metadata（source、page、block_type）和标题 metadata（h1、h2...）
            merged_meta = {**doc.metadata, **chunk.metadata}
            header_chunks.append(
                Document(page_content=chunk.page_content, metadata=merged_meta)
            )

    # ── Step 2：超长段落二次切分 ──────────────────────────────────────────
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    final_text_chunks = char_splitter.split_documents(header_chunks)

    # ── Step 3：合并（text chunk + 不切的 table/equation）────────────────
    all_chunks = final_text_chunks + preserve_docs

    # 过滤空块
    all_chunks = [c for c in all_chunks if c.page_content.strip()]

    # ── Step 4：补充结构化元数据 ──────────────────────────────────────────
    _enrich_metadata(all_chunks)

    return all_chunks


def _enrich_metadata(chunks: List[Document]) -> None:
    """
    原地补充三类元数据：
      - chunk_id:      每个 chunk 的唯一 ID（UUID）
      - section_path:  由 h1/h2/h3/h4 拼成的层级路径，如 "第3章 > 3.2 安全规程"
      - prev/next_chunk_id: 相邻 chunk 的 ID，用于检索后上下文扩展
    """
    # 先给每个 chunk 分配唯一 ID
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    # 拼 section_path
    for chunk in chunks:
        parts = [
            chunk.metadata.get("h1"),
            chunk.metadata.get("h2"),
            chunk.metadata.get("h3"),
            chunk.metadata.get("h4"),
        ]
        chunk.metadata["section_path"] = " > ".join(p for p in parts if p)

    # 连接前后 chunk ID（按列表顺序，跨 source 不连）
    for i, chunk in enumerate(chunks):
        src = chunk.metadata.get("source", "")

        prev_chunk = chunks[i - 1] if i > 0 else None
        next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

        # 跨文档不连（source 不同的 chunk 不互为邻居）
        chunk.metadata["prev_chunk_id"] = (
            prev_chunk.metadata["chunk_id"]
            if prev_chunk and prev_chunk.metadata.get("source", "") == src
            else ""
        )
        chunk.metadata["next_chunk_id"] = (
            next_chunk.metadata["chunk_id"]
            if next_chunk and next_chunk.metadata.get("source", "") == src
            else ""
        )


def split_plain_documents(docs: List[Document]) -> List[Document]:
    """
    对非 MinerU 来源的普通 Document（如 xlsx 行、网页等）做简单字符切分。
    没有 block_type 元数据时用这个。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if c.page_content.strip()]
