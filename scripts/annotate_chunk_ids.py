"""
chunk_id 标注脚本：为 QA 测试集补充 ground truth chunk 信息

原理：
  用每条 QA 的 source_text 向 Chroma 做语义检索，
  Top-1 的 chunk_id 作为 ground_truth_chunk_id。
  同时保存 Top-3 备用（应对切分边界导致的 off-by-one 问题）。

用法：
  conda activate mineru_2.5
  cd ~/rag_project
  python scripts/annotate_chunk_ids.py \
      --input  data/qa_dataset/gb_qa_1000.json \
      --output data/qa_dataset/gb_qa_1000_annotated.json \
      --collection gb_standards
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")
from chains.rag_chain import get_embeddings, get_collection_embedding_model

import chromadb

# ── 配置 ──────────────────────────────────────────────────────────────────────

CHROMA_DIR   = "data/chroma_db"
TOP_K        = 3       # 保存 Top-3，评估时可选 @1/@3/@5
MIN_SCORE    = 0.75    # 相似度低于此值视为未找到对应 chunk
SLEEP        = 0.05    # 防止 Embedding 过热

# ── 主流程 ────────────────────────────────────────────────────────────────────

def annotate(input_path: str, output_path: str, collection_name: str):
    input_path  = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, encoding="utf-8") as f:
        qa_list = json.load(f)

    print(f"载入 {len(qa_list)} 条 QA，开始标注 chunk_id...\n")

    # 加载 Embedding 模型（和入库时同一个）
    model_name = get_collection_embedding_model(collection_name)
    print(f"Embedding 模型：{model_name}")
    embeddings = get_embeddings(model_name)

    # 连接 Chroma
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(collection_name)
    print(f"Chroma collection '{collection_name}'：{collection.count()} 条\n")

    hit = 0
    miss = 0

    for i, qa in enumerate(qa_list):
        # 已标注过的跳过（断点续传）
        if qa.get("ground_truth_chunk_id"):
            hit += 1
            continue

        source_text = qa.get("source_text", "").strip()
        if not source_text:
            qa["ground_truth_chunk_id"]    = None
            qa["ground_truth_chunk_ids"]   = []
            qa["ground_truth_chunk_score"] = 0.0
            miss += 1
            continue

        # 向 Chroma 查询
        try:
            query_emb = embeddings.embed_query(source_text)
            results   = collection.query(
                query_embeddings=[query_emb],
                n_results=TOP_K,
                include=["distances", "metadatas", "documents"]
            )

            ids       = results["ids"][0]
            distances = results["distances"][0]
            docs      = results["documents"][0]
            metas     = results["metadatas"][0]

            # Chroma 返回的是 L2 距离，转成相似度分（近似）
            scores = [max(0.0, 1 - d / 2) for d in distances]

            if scores[0] < MIN_SCORE:
                qa["ground_truth_chunk_id"]    = None
                qa["ground_truth_chunk_ids"]   = []
                qa["ground_truth_chunk_score"] = scores[0]
                miss += 1
            else:
                # 优先用 metadata 里的 chunk_id（新库有），否则用 Chroma 内部 ID（旧库兜底）
                qa["ground_truth_chunk_id"]    = metas[0].get("chunk_id") or ids[0]
                qa["ground_truth_chunk_ids"]   = [
                    m.get("chunk_id") or i for m, i in zip(metas[:TOP_K], ids[:TOP_K])
                ]
                qa["ground_truth_chunk_score"] = round(scores[0], 4)
                qa["ground_truth_chunk_text"]  = docs[0][:200]
                qa["ground_truth_source"]      = metas[0].get("source", "")
                hit += 1

        except Exception as e:
            print(f"  ⚠️ [{i+1}] 查询失败：{e}")
            qa["ground_truth_chunk_id"]    = None
            qa["ground_truth_chunk_ids"]   = []
            qa["ground_truth_chunk_score"] = 0.0
            miss += 1

        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(qa_list)}  命中{hit} 未命中{miss}")
            # 实时保存
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(qa_list, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP)

    # 最终保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"完成！命中：{hit}，未命中：{miss}")
    print(f"命中率：{hit / len(qa_list) * 100:.1f}%")
    print(f"已保存至：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/qa_dataset/gb_qa_1000.json")
    parser.add_argument("--output",     default="data/qa_dataset/gb_qa_1000_annotated.json")
    parser.add_argument("--collection", default="gb_standards")
    args = parser.parse_args()

    annotate(args.input, args.output, args.collection)
