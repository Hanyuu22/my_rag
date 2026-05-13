"""
用 Qwen 从知识库中自动提取专业术语，生成 jieba userdict。

用法：
    python scripts/build_kg_userdict.py --collection <库名> [--batch 10] [--topk 8]

输出：
    data/kg_cache/<collection>_userdict.txt   （jieba load_userdict 格式）

格式（每行）：
    术语 频率 词性
    催化裂化装置 10 n
"""

import sys
import re
import json
import argparse
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")

CACHE_DIR = Path("/home/hanyuu/rag_project/data/kg_cache")

_BATCH_PROMPT = """\
以下是来自专业技术手册的若干文本片段。
请从中提取**领域专有术语**（设备名、工艺名、化学品名、技术方法名等），
不要包含普通动词、形容词或通用名词。
每行输出一个术语，不要编号，不要解释，不要重复：

{texts}
"""


def _call_llm(llm, texts: list[str]) -> list[str]:
    combined = "\n---\n".join(t[:300] for t in texts)
    try:
        raw = llm.invoke(_BATCH_PROMPT.format(texts=combined)).content
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return []
    terms = []
    for line in raw.splitlines():
        line = line.strip().strip("，。、·•-– ")
        # 过滤：长度 2-15，不含标点，不是纯数字
        if 2 <= len(line) <= 15 and not re.search(r"[，。！？\d]", line):
            terms.append(line)
    return terms


def build_userdict(collection: str, batch_size: int = 10, topk: int = 8):
    from config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL
    from chains.rag_chain import get_embeddings, get_collection_embedding_model
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{collection}_userdict.txt"

    # 加载 chunks
    emb_model = get_collection_embedding_model(collection)
    vectorstore = Chroma(
        collection_name=collection,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=get_embeddings(emb_model),
    )
    results = vectorstore._collection.get(include=["documents"])
    docs = results["documents"]
    print(f"[userdict] 共 {len(docs)} 个 chunk，batch={batch_size}")

    llm = ChatOpenAI(
        model=LLM_MODEL, temperature=0,
        api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL,
    )

    all_terms: dict[str, int] = {}  # term → 出现次数（跨 batch 累计）

    total_batches = (len(docs) + batch_size - 1) // batch_size
    for i in range(0, len(docs), batch_size):
        batch = docs[i: i + batch_size]
        terms = _call_llm(llm, batch)
        for t in terms:
            all_terms[t] = all_terms.get(t, 0) + 1
        batch_no = i // batch_size + 1
        print(f"  [{batch_no}/{total_batches}] 本批提取 {len(terms)} 个术语，累计 {len(all_terms)} 个")

    # 按出现频次排序，频次越高权重越大（jieba 用频率值影响分词优先级）
    sorted_terms = sorted(all_terms.items(), key=lambda x: -x[1])

    lines = []
    for term, freq in sorted_terms:
        weight = min(freq * 10, 100)   # 映射到 jieba 频率权重
        lines.append(f"{term} {weight} n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[userdict] 完成，共 {len(lines)} 个术语 → {out_path}")
    print("前 20 个术语：")
    for line in lines[:20]:
        print(" ", line)

    # 同时把原始 kg_cache 清掉，强制下次重建图
    kg_cache = CACHE_DIR / f"{collection}_kg.json"
    if kg_cache.exists():
        kg_cache.unlink()
        print(f"[userdict] 已清除旧 KG 缓存 {kg_cache.name}，下次请求将重建图")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="知识库名称")
    parser.add_argument("--batch",  type=int, default=10, help="每批 chunk 数（默认 10）")
    parser.add_argument("--topk",   type=int, default=8,  help="每 chunk 提取术语数（暂未使用，由 LLM 自行决定）")
    args = parser.parse_args()

    build_userdict(args.collection, batch_size=args.batch)
