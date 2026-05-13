"""
QA 数据集自动标注 ground_truth_chunk_id

原理：
  对每条 QA 的 source_text，在 hydro_crack collection 里做向量检索，
  优先在同页（±1页）范围内找最相似的 chunk，
  将其 chunk_id 写入 ground_truth_chunk_id 字段。

这是"银标准"（自动标注），准确率取决于 source_text 与 chunk 的重叠度。

用法：
    python scripts/annotate_chunk_ids.py \
        --input  data/qa_dataset/hydro_reg_qa_clean.json \
        --output data/qa_dataset/hydro_crack_qa_annotated.json \
        --collection hydro_crack
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")
from config import CHROMA_PERSIST_DIR
from chains.rag_chain import get_embeddings, get_collection_embedding_model


def annotate(input_path: str, output_path: str, collection_name: str):
    import chromadb

    # 加载 QA 数据集
    with open(input_path, encoding="utf-8") as f:
        qa_list = json.load(f)
    print(f"加载 QA 数据集：{len(qa_list)} 条")

    # 初始化 ChromaDB 和 Embedding
    print("加载 Embedding 模型...")
    emb_model = get_collection_embedding_model(collection_name)
    embeddings = get_embeddings(emb_model)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    col = client.get_collection(collection_name)
    print(f"collection: {collection_name}，共 {col.count()} chunks\n")

    success = 0
    fallback_global = 0
    failed = 0

    for i, qa in enumerate(qa_list):
        source_text = qa.get("source_text", "").strip()
        page = qa.get("page")

        if not source_text:
            qa["ground_truth_chunk_id"] = None
            failed += 1
            continue

        # 生成 source_text 的 embedding
        query_vec = embeddings.embed_query(source_text)

        chunk_id = None

        # 第一步：同页 ±1 范围内检索
        if page is not None:
            pages_to_try = [p for p in [page - 1, page, page + 1] if p >= 0]
            try:
                res = col.query(
                    query_embeddings=[query_vec],
                    n_results=1,
                    where={"page": {"$in": pages_to_try}},
                    include=["metadatas", "distances"],
                )
                if res["ids"] and res["ids"][0]:
                    chunk_id = res["metadatas"][0][0].get("chunk_id")
                    success += 1
            except Exception:
                pass

        # 第二步：全库检索（兜底）
        if chunk_id is None:
            try:
                res = col.query(
                    query_embeddings=[query_vec],
                    n_results=1,
                    include=["metadatas", "distances"],
                )
                if res["ids"] and res["ids"][0]:
                    chunk_id = res["metadatas"][0][0].get("chunk_id")
                    fallback_global += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        qa["ground_truth_chunk_id"] = chunk_id

        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(qa_list)}")

    # 保存
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print(f"\n标注完成：")
    print(f"  同页命中：{success} 条")
    print(f"  全库兜底：{fallback_global} 条")
    print(f"  失败     ：{failed} 条")
    print(f"  输出文件：{output_path}")

    # 抽查前3条
    print("\n── 抽查前3条 ──")
    for qa in qa_list[:3]:
        print(f"  p.{qa.get('page')} [{qa.get('question_type')}] {qa.get('question','')[:50]}")
        print(f"    chunk_id: {qa.get('ground_truth_chunk_id')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/qa_dataset/hydro_reg_qa_clean.json")
    parser.add_argument("--output",     default="data/qa_dataset/hydro_crack_qa_annotated.json")
    parser.add_argument("--collection", default="hydro_crack")
    args = parser.parse_args()
    annotate(args.input, args.output, args.collection)
