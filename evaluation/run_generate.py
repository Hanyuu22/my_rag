"""
Step 1 — Golden Set 草稿生成器

读取 golden_set_questions.json，对每道题跑一次完整的 RAG graph，
把生成的答案和检索到的 contexts 写入 draft_golden_set.json。

之后人工审核 draft_golden_set.json，修改 reference 字段中不准确的答案，
再用 run_ragas_eval.py 进行 RAGAS 批量评估。

用法：
    conda activate mineru_2.5
    cd ~/rag_project
    python evaluation/run_generate.py --collection <集合名称>

    # 例：
    python evaluation/run_generate.py --collection hydro_manual
    python evaluation/run_generate.py --collection hydro_manual --start 25 --end 50
"""

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Golden Set 草稿生成")
    parser.add_argument("--collection",  required=True,  help="Chroma collection 名称")
    parser.add_argument("--input",       default="evaluation/golden_set_questions.json",
                        help="输入题目文件（默认：evaluation/golden_set_questions.json）")
    parser.add_argument("--output",      default="evaluation/draft_golden_set.json",
                        help="输出草稿文件（默认：evaluation/draft_golden_set.json）")
    parser.add_argument("--start",       type=int, default=1,   help="从第几题开始（id，含，默认 1）")
    parser.add_argument("--end",         type=int, default=9999, help="到第几题结束（id，含，默认全部）")
    parser.add_argument("--top_k",       type=int,   default=10,  help="Hybrid 检索 Top-K（默认 10）")
    parser.add_argument("--reranker_n",  type=int,   default=3,   help="Reranker Top-N（默认 3）")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM Temperature（默认 0.1）")
    return parser.parse_args()


# ── 初始化 RAG 图 ─────────────────────────────────────────────────────────────

def build_graph(collection: str, top_k: int, reranker_n: int, temperature: float):
    print(f"\n[初始化] 加载知识库 '{collection}'，Top-K={top_k}，Reranker-N={reranker_n} ...")
    t0 = time.time()

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI
    from config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL
    from chains.rag_chain import get_embeddings
    from retrievers.hybrid_retriever import build_hybrid_retriever
    from retrievers.reranker import BGEReranker
    from graphs.rag_graph import build_rag_graph

    vectorstore = Chroma(
        collection_name=collection,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=get_embeddings(),
    )
    results = vectorstore._collection.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
    print(f"  向量库加载完成，共 {len(chunks)} 个 chunks")

    hybrid_retriever = build_hybrid_retriever(chunks, vectorstore, k=top_k)
    reranker = BGEReranker(top_n=reranker_n)
    llm = ChatOpenAI(
        model=LLM_MODEL, temperature=temperature,
        api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL,
    )
    graph = build_rag_graph(hybrid_retriever, reranker, llm)
    print(f"  RAG 图构建完成（耗时 {time.time()-t0:.1f}s）\n")
    return graph


# ── 运行单题 ──────────────────────────────────────────────────────────────────

def run_one(graph, question: str) -> dict:
    """返回 {answer, contexts, scores, route, elapsed}"""
    t0 = time.time()
    result = graph.invoke({
        "question": question,
        "score_threshold": 0.5,
        "max_retry_limit": 2,
    })
    elapsed = round(time.time() - t0, 2)

    docs   = result.get("docs",   [])
    scores = result.get("scores", [])
    return {
        "answer":   result.get("answer", ""),
        "contexts": [d.page_content for d in docs],
        "scores":   scores,
        "route":    result.get("route", "retrieve"),
        "elapsed":  elapsed,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 读取题目
    q_path = Path(args.input)
    if not q_path.exists():
        print(f"[错误] 题目文件不存在：{q_path}")
        sys.exit(1)
    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)

    # 过滤范围
    questions = [q for q in questions if args.start <= q["id"] <= args.end]
    print(f"[计划] 共 {len(questions)} 道题，collection='{args.collection}'")

    # 加载已有草稿（支持断点续跑）
    out_path = Path(args.output)
    existing: dict[int, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["id"]] = item
        print(f"[断点续跑] 已有 {len(existing)} 条草稿，跳过已完成的题目")

    # 过滤掉已跑过的（id 已存在且 answer 非空）
    todo = [q for q in questions if not (q["id"] in existing and existing[q["id"]].get("answer"))]
    print(f"[计划] 需要运行 {len(todo)} 道题\n")

    if not todo:
        print("所有题目已有草稿，无需重跑。")
        return

    # 构建图（只在有题目要跑时才加载，节省 GPU 显存）
    graph = build_graph(args.collection, args.top_k, args.reranker_n, args.temperature)

    # 逐题运行
    results = dict(existing)  # 先复制已有草稿
    for i, q in enumerate(todo, 1):
        qid      = q["id"]
        category = q["category"]
        question = q["question"]

        print(f"[{i:2d}/{len(todo)}] #{qid} [{category}]")
        print(f"  Q: {question}")

        try:
            out = run_one(graph, question)
            results[qid] = {
                "id":        qid,
                "category":  category,
                "question":  question,
                # reference：Fallback/路由题保留原始说明，其余留空待人工确认
                "reference": q.get("reference", ""),
                "answer":    out["answer"],
                "contexts":  out["contexts"],
                "scores":    out["scores"],
                "route":     out["route"],
                "elapsed":   out["elapsed"],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  ✅ 路由={out['route']}  Top1={out['scores'][0] if out['scores'] else 'N/A':.3f}  耗时={out['elapsed']}s")
            # 截断答案预览（避免刷屏）
            preview = out["answer"].replace("\n", " ")[:120]
            print(f"  A: {preview}{'...' if len(out['answer']) > 120 else ''}\n")

        except Exception as e:
            print(f"  ❌ 错误：{e}\n")
            results[qid] = {
                "id":       qid,
                "category": category,
                "question": question,
                "reference": q.get("reference", ""),
                "answer":   "",
                "contexts": [],
                "scores":   [],
                "route":    "error",
                "elapsed":  0,
                "error":    str(e),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        # 每题完成后立即保存（防止中途崩溃丢失）
        _save(out_path, results)

    print(f"\n[完成] 草稿已保存到：{out_path}")
    print("请打开该文件，检查并修正 'reference' 字段中不准确的答案，")
    print("然后运行：python evaluation/run_ragas_eval.py --input evaluation/draft_golden_set.json\n")


def _save(path: Path, results: dict):
    """按 id 排序后写入 JSON"""
    ordered = [results[k] for k in sorted(results.keys())]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
