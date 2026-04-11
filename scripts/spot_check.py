"""
人工抽检脚本：输出"问题 + 检索 Top-5 + 标注 ground truth"供人眼对比

用途：
  验证自动标注噪声假设——检索结果包含正确答案但被标注误判的比例

用法：
  conda activate mineru_2.5
  cd ~/rag_project

  # 随机抽 30 题，从 512 库检索
  python scripts/spot_check.py \
      --input data/qa_dataset/gb_qa_50docs_annotated_512.json \
      --collection gb_standards_512 \
      --n 30

  # 指定 random seed 保证可复现
  python scripts/spot_check.py --seed 42 --n 20

输出：
  终端逐题打印（方便边看边标）
  data/results/spot_check_result.json（记录每题的人工判断）
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")


def load_retriever(collection_name: str, top_k: int):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from chains.rag_chain import get_embeddings, get_collection_embedding_model
    from retrievers.hybrid_retriever import build_hybrid_retriever
    from config import CHROMA_PERSIST_DIR

    model_name  = get_collection_embedding_model(collection_name)
    embeddings  = get_embeddings(model_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )
    results = vectorstore._collection.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
    print(f"  向量库加载完成，共 {len(chunks)} 个 chunks")
    retriever = build_hybrid_retriever(chunks, vectorstore, k=top_k)
    return retriever, vectorstore._collection


def find_gt_chunk(collection, gt_id: str) -> str:
    """从 Chroma 里取出标注 ground truth chunk 的文本"""
    if not gt_id:
        return "(无标注)"
    try:
        # 先按 metadata chunk_id 查
        res = collection.get(where={"chunk_id": gt_id}, include=["documents"])
        if res["documents"]:
            return res["documents"][0]
    except Exception:
        pass
    try:
        # 兜底：用 Chroma 内部 ID
        res = collection.get(ids=[gt_id], include=["documents"])
        if res["documents"]:
            return res["documents"][0]
    except Exception:
        pass
    return "(未找到对应 chunk)"


def run(input_path: str, collection_name: str, n: int, seed: int, top_k: int, output: str):
    with open(input_path, encoding="utf-8") as f:
        qa_list = json.load(f)

    # 只取有标注的题
    valid = [qa for qa in qa_list if qa.get("ground_truth_chunk_id")]
    print(f"共 {len(qa_list)} 题，有标注 {len(valid)} 题")

    random.seed(seed)
    sample = random.sample(valid, min(n, len(valid)))
    print(f"随机抽取 {len(sample)} 题（seed={seed}）\n")

    print("加载检索器...")
    retriever, collection = load_retriever(collection_name, top_k)
    print()

    records = []

    for idx, qa in enumerate(sample):
        question  = qa["question"]
        gt_id     = qa["ground_truth_chunk_id"]
        gt_ids    = set(qa.get("ground_truth_chunk_ids", [gt_id]))
        gt_score  = qa.get("ground_truth_chunk_score", 0)
        gt_text   = qa.get("ground_truth_chunk_text", "") or find_gt_chunk(collection, gt_id)

        # 检索
        try:
            docs = retriever.invoke(question)
        except Exception as e:
            print(f"[{idx+1}] 检索失败：{e}\n")
            continue

        retrieved_ids = [d.metadata.get("chunk_id", "") for d in docs]

        # 判断是否命中（自动）
        auto_hit_rank = None
        for pos, cid in enumerate(retrieved_ids, 1):
            if cid and cid in gt_ids:
                auto_hit_rank = pos
                break

        # ── 打印 ──────────────────────────────────────────────
        sep = "═" * 70
        print(sep)
        print(f"[{idx+1}/{len(sample)}]  自动判断: {'✅ Hit@' + str(auto_hit_rank) if auto_hit_rank else '❌ Miss'}")
        print(f"问题：{question}")
        print(f"题型：{qa.get('question_type','?')}  难度：{qa.get('difficulty','?')}  主题：{qa.get('topic_tag','?')}")
        print()

        print(f"── 标注 ground truth（chunk_id={gt_id[:20]}...  标注相似度={gt_score:.3f}）")
        print(f"   {gt_text[:300].replace(chr(10), ' ')}")
        print()

        print(f"── 检索 Top-{min(top_k, len(docs))} 结果")
        for rank, doc in enumerate(docs[:top_k], 1):
            cid   = doc.metadata.get("chunk_id", "")
            src   = doc.metadata.get("source", doc.metadata.get("file_name", ""))[:30]
            hit   = "★" if cid in gt_ids else " "
            text  = doc.page_content[:200].replace("\n", " ")
            print(f"  [{rank}]{hit} {src}  {text}")
        print()

        # 请用户输入判断（可跳过，直接回车）
        print("人工判断（回车=跳过，y=检索正确但标注错了，n=检索确实错了，?=不确定）：", end="", flush=True)
        try:
            verdict = input().strip().lower() or "skip"
        except EOFError:
            verdict = "skip"

        records.append({
            "idx":          idx + 1,
            "question":     question,
            "question_type":qa.get("question_type", ""),
            "difficulty":   qa.get("difficulty", ""),
            "topic_tag":    qa.get("topic_tag", ""),
            "gt_chunk_id":  gt_id,
            "gt_score":     gt_score,
            "gt_text":      gt_text[:300],
            "auto_hit_rank":auto_hit_rank,
            "retrieved_ids":retrieved_ids[:top_k],
            "retrieved_texts":[d.page_content[:200] for d in docs[:top_k]],
            "verdict":      verdict,   # y/n/?/skip
        })
        print()

    # ── 统计 ──────────────────────────────────────────────────
    print("═" * 70)
    total    = len(records)
    auto_hit = sum(1 for r in records if r["auto_hit_rank"] is not None)
    manual_y = sum(1 for r in records if r["verdict"] == "y")   # 检索对但标注错
    manual_n = sum(1 for r in records if r["verdict"] == "n")   # 检索确实错
    skip     = sum(1 for r in records if r["verdict"] == "skip")

    print(f"\n抽检结果统计（共 {total} 题）")
    print(f"  自动命中（标注匹配）：{auto_hit}/{total} = {auto_hit/total:.2%}")
    print(f"  人工判断-检索正确但标注错：{manual_y}")
    print(f"  人工判断-检索确实错：      {manual_n}")
    print(f"  人工判断-不确定/跳过：     {skip + sum(1 for r in records if r['verdict'] == '?')}")

    if total > 0:
        # 修正后估算真实 hit rate
        corrected_hit = auto_hit + manual_y
        print(f"\n  修正后命中（自动+人工修正）：{corrected_hit}/{total} = {corrected_hit/total:.2%}")
        print(f"  标注噪声估算：{manual_y/total:.1%} 的题目因标注错误被低估")

    # 保存
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "collection":    collection_name,
            "n_sampled":     total,
            "auto_hit":      auto_hit,
            "manual_y":      manual_y,
            "manual_n":      manual_n,
            "corrected_hit": auto_hit + manual_y,
            "records":       records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至：{out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人工抽检：验证自动标注噪声")
    parser.add_argument("--input",      default="data/qa_dataset/gb_qa_50docs_annotated_512.json")
    parser.add_argument("--collection", default="gb_standards_512")
    parser.add_argument("--n",          type=int,   default=30,   help="抽检题数（默认30）")
    parser.add_argument("--seed",       type=int,   default=42,   help="随机种子（默认42）")
    parser.add_argument("--top_k",      type=int,   default=5,    help="检索 Top-K（默认5）")
    parser.add_argument("--output",     default="data/results/spot_check_result.json")
    args = parser.parse_args()

    run(args.input, args.collection, args.n, args.seed, args.top_k, args.output)
