"""
Markdown 结构化切分器 v2

文本切分策略（贪心积累）：
  H1/H2 标题边界 → 无条件新建 chunk（语义强边界，不加 overlap）
  H3/H4 标题边界 → 贪心积累，累计不超过 CHUNK_SIZE 时合并；
                    超过时切断，在切点处为下一 chunk 保留 overlap 字符前缀
  单节内容超长   → RecursiveCharacterTextSplitter 兜底截断（含 overlap）

表格合并策略（三级判断）：
  Case 1：续块首行以 | 开头（无标题）+ 列数匹配(±1) + 相邻页
  Case 2：当前块无数据行（仅表头+分隔行）+ 列数匹配 + 相邻页（跨页标题误贴）
  Case 3：当前块末尾序号 N，续块起始序号 N+1 + 列数匹配 + 相邻页
"""

import re
import uuid
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")
from config import CHUNK_SIZE, CHUNK_OVERLAP

# token 计数（bge-large-zh-v1.5 的实际限制是 512 tokens）
# 用 tokenizer 计数确保 chunk 不超过模型输入上限
try:
    from transformers import AutoTokenizer as _AutoTokenizer
    _tokenizer = _AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
    def _token_len(text: str) -> int:
        return len(_tokenizer.encode(text, add_special_tokens=True))
except Exception:
    _token_len = len  # 兜底退化为字符数


HEADERS_TO_SPLIT_ON = [
    ("#",    "h1"),
    ("##",   "h2"),
    ("###",  "h3"),
    ("####", "h4"),
]

# H1/H2 为强边界（无条件新 chunk），H3/H4 为弱边界（贪心积累）
STRONG_BOUNDARY_LEVELS = {1, 2}

# MinerU 输出的编号标题（如 "7.1.1 安全基本概念"）没有 # 前缀，需要预处理注入
_H2_RE = re.compile(r'^\d{1,2}\s+[一-鿿A-Z(（]')          # e.g. "7 HSE..."
_H3_RE = re.compile(r'^\d{1,2}\.\d{1,2}\s+[一-鿿A-Z(（]') # e.g. "7.1 安全..."
_H4_RE = re.compile(r'^\d{1,2}(?:\.\d+){2,}\s+\S')                 # e.g. "7.1.1 安全基本概念"
_MAX_HEADING_LEN = 60


def _inject_heading_markers(text: str) -> str:
    """
    将 MinerU 输出中独立的编号标题段落（单行、≤60字）
    加上 ##/###/#### 前缀，使 MarkdownHeaderTextSplitter 能识别。
    """
    parts = text.split("\n\n")
    out = []
    for p in parts:
        s = p.strip()
        if "\n" not in s and 1 < len(s) <= _MAX_HEADING_LEN:
            if _H4_RE.match(s):
                p = "#### " + s
            elif _H3_RE.match(s):
                p = "### " + s
            elif _H2_RE.match(s):
                p = "## " + s
        out.append(p)
    return "\n\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def split_documents(docs: List[Document]) -> List[Document]:
    text_docs     = [d for d in docs if d.metadata.get("block_type") == "text"]
    table_docs    = [d for d in docs if d.metadata.get("block_type") == "table"]
    eq_docs       = [d for d in docs if d.metadata.get("block_type") == "equation"]

    # ── 表格：三级判断合并跨页块 ──────────────────────────────────────────
    merged_tables = _merge_consecutive_tables(table_docs)

    # ── 文本：拼整体 → 标题切分 → 贪心积累 ───────────────────────────────
    source = text_docs[0].metadata.get("source", "") if text_docs else ""
    full_text = "\n\n".join(d.page_content for d in text_docs)
    full_text = _inject_heading_markers(full_text)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    raw_chunks = header_splitter.split_text(full_text)
    section_chunks: List[Document] = [
        Document(
            page_content=c.page_content,
            metadata={"source": source, "block_type": "text", **c.metadata},
        )
        for c in raw_chunks
    ]

    # 合并孤立标题（<30字的极短节）
    section_chunks = _merge_short_chunks(section_chunks, min_length=30)

    # 贪心积累 + 超长节内截断
    final_text_chunks = _greedy_accumulate(section_chunks)

    # ── 合并所有类型 ──────────────────────────────────────────────────────
    all_chunks = final_text_chunks + merged_tables + eq_docs
    all_chunks = [c for c in all_chunks if c.page_content.strip()]

    _enrich_metadata(all_chunks)
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# 文本：贪心积累
# ─────────────────────────────────────────────────────────────────────────────

def _heading_level(chunk: Document) -> int:
    """返回 chunk 的最深标题层级（1-4），无标题返回 0。"""
    for level, key in [(4, "h4"), (3, "h3"), (2, "h2"), (1, "h1")]:
        if chunk.metadata.get(key):
            return level
    return 0


def _greedy_accumulate(
    sections: List[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    贪心积累：
      强边界（H1/H2）→ 无条件切断，新 chunk 从头开始
      弱边界（H3/H4）→ 整节积累，加入后超限则回退：当前节作为下一 chunk 起点
      单节超长       → RecursiveCharacterTextSplitter 截断（内部 overlap 可接受）
    各节之间无跨节 overlap，保证章节边界干净。
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_token_len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    result: List[Document] = []
    pending: List[Document] = []
    pending_size: int = 0

    def flush() -> None:
        nonlocal pending, pending_size
        if not pending:
            return
        merged = "\n".join(c.page_content for c in pending)
        meta   = {**pending[0].metadata}
        if _token_len(merged) <= chunk_size:
            result.append(Document(page_content=merged, metadata=meta))
        else:
            result.extend(char_splitter.split_documents(
                [Document(page_content=merged, metadata=meta)]
            ))
        pending = []
        pending_size = 0

    for chunk in sections:
        level = _heading_level(chunk)
        size  = _token_len(chunk.page_content)

        if level in STRONG_BOUNDARY_LEVELS:
            flush()
            pending = [chunk]
            pending_size = size
        else:
            # H3/H4 或无标题：整节积累，超限则回退
            if pending and pending_size + size > chunk_size:
                flush()
                pending = [chunk]
                pending_size = size
            else:
                pending.append(chunk)
                pending_size += size

    flush()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 表格：三级判断合并
# ─────────────────────────────────────────────────────────────────────────────

def _count_columns(content: str) -> int:
    """
    取所有 | 行中最大有效列数。
    排除非空格子占少数的行（MinerU 常在跨页表尾填充大量空 | 占位符）。
    """
    max_cols = 0
    for line in content.split("\n"):
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = s.split("|")
        non_empty = sum(1 for c in cells if c.strip())
        if non_empty > len(cells) / 2:
            max_cols = max(max_cols, s.count("|") - 1)
    return max_cols


def _is_separator_line(line: str) -> bool:
    return bool(re.match(r"^\|[\s\-:]+(\|[\s\-:]+)+\|?$", line.strip()))


def _table_data_rows(content: str) -> List[str]:
    """返回表格中的数据行（去掉分隔行，去掉首行表头行）。"""
    pipe_lines = [l for l in content.split("\n") if l.strip().startswith("|")]
    # 第一个非分隔行为表头行，之后的非分隔行为数据行
    header_seen = False
    data: List[str] = []
    for line in pipe_lines:
        if _is_separator_line(line):
            header_seen = True
            continue
        if not header_seen:
            continue   # 还在表头之前
        data.append(line)
    return data


def _table_has_no_data(content: str) -> bool:
    """当前块只有表头+分隔行，没有数据行。"""
    return len(_table_data_rows(content)) == 0


def _is_table_continuation(doc: Document) -> bool:
    """
    首行以 | 开头且紧跟行中无分隔行 → 原始数据续块。
    若首行后紧跟分隔行（|---|），说明这是带自己表头的新表格，不是续块。
    """
    lines = [l for l in doc.page_content.strip().split("\n") if l.strip()]
    if not lines or not lines[0].startswith("|"):
        return False
    for line in lines[1:4]:
        if _is_separator_line(line):
            return False
    return True


def _extract_data_lines_for_merge(content: str) -> List[str]:
    """提取续块的数据行（跳过分隔行）用于追加到主块。"""
    return [l for l in content.split("\n")
            if l.strip().startswith("|") and not _is_separator_line(l)]


def _last_seq_number(content: str) -> Optional[int]:
    """返回表格最后一条数据行的第一列（如果是纯数字）。"""
    rows = _table_data_rows(content)
    for row in reversed(rows):
        parts = [p.strip() for p in row.split("|") if p.strip()]
        if parts and parts[0].isdigit():
            return int(parts[0])
    return None


def _first_seq_number(content: str) -> Optional[int]:
    """返回第一条非分隔 | 行的首列（如果是纯整数）。
    不区分分隔行前后，直接取首个有效数字行，避免 MinerU 在续块中插入分隔行导致误判。
    """
    for line in content.split("\n"):
        s = line.strip()
        if not s.startswith("|") or _is_separator_line(s):
            continue
        parts = [p.strip() for p in s.split("|") if p.strip()]
        if parts and parts[0].isdigit():
            return int(parts[0])
    return None


def _first_pipe_row(content: str) -> str:
    """返回内容中第一个以 | 开头的行。"""
    for line in content.split("\n"):
        if line.strip().startswith("|"):
            return line.strip()
    return ""


def _pipe_row_is_data(row: str) -> bool:
    """
    判断一个 | 行是否为数据行（超过半数格子以数字或测量符号开头）。
    覆盖：374、10.0/1.0、~84、>100、<-50、1.2~6.0 等测量值。
    """
    if not row:
        return False
    cells = [c.strip() for c in row.split("|") if c.strip()]
    if not cells:
        return False
    measurement = sum(
        1 for c in cells if re.match(r"^[-~<>≥≤\d]", c)
    )
    return measurement > len(cells) / 2


def _should_merge_tables(
    current: Document, nxt: Document
) -> Tuple[bool, str]:
    """
    判断 nxt 是否应合并入 current。
    返回 (should_merge, case_label)。
    """
    curr_page = current.metadata.get("page", -1)
    nxt_page  = nxt.metadata.get("page", -1)
    if abs(nxt_page - curr_page) > 1:
        return False, ""

    curr_cols = _count_columns(current.page_content)
    nxt_cols  = _count_columns(nxt.page_content)
    cols_ok   = abs(curr_cols - nxt_cols) <= 1

    # Case 1：续块无标题（首行以 | 开头）
    if _is_table_continuation(nxt) and cols_ok:
        return True, "C1-no-caption"

    # Case 2：当前块无数据行（只有表头+分隔行），列数匹配
    if _table_has_no_data(current.page_content) and cols_ok:
        return True, "C2-no-data"

    # Case 3：序号连续（末尾 N → 首行 N+1），列数匹配
    if cols_ok:
        last = _last_seq_number(current.page_content)
        first = _first_seq_number(nxt.page_content)
        if last is not None and first is not None and first == last + 1:
            return True, "C3-seq"

    # Case 4：续块有标题，但其"表头行"实为数值数据（MinerU 跨页误封装）
    # 典型例：表9-1末尾续行被 MinerU 标注为"表9-2"，首个 | 行全是温度/参数值
    # 列数容差放宽至 ±5：9-3/9-4 模式中可见列（3）与续行实际列（8）差距较大
    cols_ok_loose = abs(curr_cols - nxt_cols) <= 5
    if cols_ok_loose and _pipe_row_is_data(_first_pipe_row(nxt.page_content)):
        return True, "C4-numeric-header"

    return False, ""


def _extract_caption(content: str) -> str:
    """提取表格块的标题行（第一个非 | 开头的非空行）。"""
    for line in content.split("\n"):
        s = line.strip()
        if s and not s.startswith("|"):
            return s
    return ""


def _merge_consecutive_tables(table_docs: List[Document]) -> List[Document]:
    if not table_docs:
        return []

    result: List[Document] = []
    current = table_docs[0]
    pending_caption: str = ""  # Case 4 合并时保存被丢弃的真实表名

    for nxt in table_docs[1:]:
        should, case = _should_merge_tables(current, nxt)
        if should:
            # 任何合并情况下，若 nxt 自带 caption，该 caption 实为下一独立表的表名
            nxt_caption = _extract_caption(nxt.page_content)
            if nxt_caption:
                pending_caption = nxt_caption
            data_lines = _extract_data_lines_for_merge(nxt.page_content)
            merged_content = current.page_content.rstrip() + "\n" + "\n".join(data_lines)
            current = Document(page_content=merged_content, metadata=current.metadata)
        else:
            result.append(current)
            current = nxt
            # 将上一次合并时保存的表名贴到本块头部（如果本块自身没有标题）
            if pending_caption and not _extract_caption(current.page_content):
                current = Document(
                    page_content=pending_caption + "\n" + current.page_content,
                    metadata=current.metadata,
                )
            pending_caption = ""

    # 最后一块
    if pending_caption and not _extract_caption(current.page_content):
        current = Document(
            page_content=pending_caption + "\n" + current.page_content,
            metadata=current.metadata,
        )
    result.append(current)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────────────────────────────────────

def _merge_short_chunks(chunks: List[Document], min_length: int = 30) -> List[Document]:
    if not chunks:
        return chunks
    result: List[Document] = []
    pending: Optional[Document] = None
    for chunk in chunks:
        if pending is not None:
            merged = pending.page_content.rstrip() + "\n" + chunk.page_content.lstrip()
            chunk = Document(
                page_content=merged,
                metadata={**pending.metadata, **chunk.metadata},
            )
            pending = None
        if _token_len(chunk.page_content.strip()) < min_length:
            pending = chunk
        else:
            result.append(chunk)
    if pending is not None:
        result.append(pending)
    return result


def _enrich_metadata(chunks: List[Document]) -> None:
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    for chunk in chunks:
        parts = [chunk.metadata.get(k) for k in ("h1", "h2", "h3", "h4")]
        chunk.metadata["section_path"] = " > ".join(p for p in parts if p)

    for i, chunk in enumerate(chunks):
        src  = chunk.metadata.get("source", "")
        prev = chunks[i - 1] if i > 0 else None
        nxt  = chunks[i + 1] if i < len(chunks) - 1 else None
        chunk.metadata["prev_chunk_id"] = (
            prev.metadata["chunk_id"]
            if prev and prev.metadata.get("source", "") == src else ""
        )
        chunk.metadata["next_chunk_id"] = (
            nxt.metadata["chunk_id"]
            if nxt and nxt.metadata.get("source", "") == src else ""
        )


# ─────────────────────────────────────────────────────────────────────────────
# 兼容接口
# ─────────────────────────────────────────────────────────────────────────────

def split_documents_with_parents(docs: List[Document]) -> List[Document]:
    return split_documents(docs)


def split_plain_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return [c for c in splitter.split_documents(docs) if c.page_content.strip()]
