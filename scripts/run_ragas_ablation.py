"""
chunk_size 消融实验 — RAGAS 端到端评估

对指定 collection 运行 RAG graph，收集 answer + contexts，
然后用 RAGAS 计算四指标。

用法：
    conda activate mineru_2.5
    cd ~/rag_project

    # 先小样本验证
    python scripts/run_ragas_ablation.py --collection hydro_crack     --max 30
    python scripts/run_ragas_ablation.py --collection hydro_crack_256 --max 30
    python scripts/run_ragas_ablation.py --collection hydro_crack_1024 --max 30

    # 全量
    python scripts/run_ragas_ablation.py --collection hydro_crack
    python scripts/run_ragas_ablation.py --collection hydro_crack_256
    python scripts/run_ragas_ablation.py --collection hydro_crack_1024
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, "/home/hanyuu/rag_project")

QA_FILE    = "data/qa_dataset/hydro_crack_qa.json"
OUTPUT_DIR = Path("data/results/ragas_ablation")
INTERVAL   = 0.5   # 每题之间的间隔（秒），避免 LLM 限速


# ── RAG 调用 ──────────────────────────────────────────────────────────────────

def build_graph(collection: str, top_k: int, reranker_n: int):
    print(f"初始化 RAG graph: collection={collection}, top_k={top_k}, reranker_n={reranker_n}")
    from langchain_chroma import Chroma
    from config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL
    from chains.rag_chain import get_embeddings, get_collection_embedding_model
    from retrievers.hybrid_retriever import build_hybrid_retriever
    from retrievers.reranker import BGEReranker
    from graphs.rag_graph import build_rag_graph

    emb_model  = get_collection_embedding_model(collection)
    embeddings = get_embeddings(emb_model)
    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )
    all_docs_result = vectorstore._collection.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document as LCDoc
    all_docs = [
        LCDoc(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs_result["documents"], all_docs_result["metadatas"])
    ]
    retriever = build_hybrid_retriever(all_docs, vectorstore, k=top_k)
    reranker  = BGEReranker(top_n=reranker_n)

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=0.1,
    )
    graph = build_rag_graph(hybrid_retriever=retriever, reranker=reranker, llm=llm)
    print("  graph 就绪")
    return graph


def call_rag(question: str, graph, reranker_enabled: bool = True) -> dict:
    try:
        result = graph.invoke({
            "question":            question,
            "score_threshold":     0.3,
            "max_retry_limit":     1,
            "reranker_enabled":    reranker_enabled,
            "fallback_enabled":    False,
            "tools_enabled":       False,
            "cross_lingual_enabled": False,
            "token_callback":      None,
        })
        answer   = result.get("answer", "")
        docs     = result.get("docs", [])
        contexts = [d.page_content for d in docs[:3]]
        return {"answer": answer, "contexts": contexts, "ok": True}
    except Exception as e:
        return {"answer": "", "contexts": [], "ok": False, "error": str(e)}


# ── RAGAS 评估 ────────────────────────────────────────────────────────────────

def _safe(val):
    try:
        v = float(val)
        return None if (v != v) else round(v, 4)   # NaN check
    except Exception:
        return None


def run_ragas(samples: list[dict]) -> list[dict]:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from chains.rag_chain import get_llm, get_embeddings

    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

    for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
        m.llm = ragas_llm
    for m in [answer_relevancy]:
        m.embeddings = ragas_emb

    run_cfg = RunConfig(max_workers=1, timeout=120)

    # 有 reference 的题：四指标；无 reference：只跑两指标
    full_idx, partial_idx = [], []
    for i, s in enumerate(samples):
        (full_idx if s.get("reference") else partial_idx).append(i)

    scores = [None] * len(samples)

    def _evaluate_group(indices, metrics):
        if not indices:
            return
        ds = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input   = samples[i]["question"],
                response     = samples[i]["answer"],
                retrieved_contexts = samples[i]["contexts"],
                reference    = samples[i].get("reference") or None,
            )
            for i in indices
        ])
        result = evaluate(ds, metrics=metrics, run_config=run_cfg)
        df = result.to_pandas()
        for j, i in enumerate(indices):
            scores[i] = {
                "faithfulness":      _safe(df.get("faithfulness",      [None]*len(indices))[j]),
                "answer_relevancy":  _safe(df.get("answer_relevancy",  [None]*len(indices))[j]),
                "context_precision": _safe(df.get("context_precision", [None]*len(indices))[j]),
                "context_recall":    _safe(df.get("context_recall",    [None]*len(indices))[j]),
            }

    _evaluate_group(full_idx,    [faithfulness, answer_relevancy, context_precision, context_recall])
    _evaluate_group(partial_idx, [faithfulness, answer_relevancy])

    return scores


def _nanmean(vals):
    clean = [v for v in vals if v is not None]
    return round(float(np.mean(clean)), 4) if clean else None


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection",  required=True)
    parser.add_argument("--qa_file",     default=QA_FILE)
    parser.add_argument("--max",         type=int, default=None, help="最多评估 N 题（随机采样）")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--top_k",       type=int, default=10)
    parser.add_argument("--reranker_n",  type=int, default=3)
    parser.add_argument("--no_reranker", action="store_true", help="关闭 reranker（用于 chunk_size 消融）")
    args = parser.parse_args()

    with open(args.qa_file, encoding="utf-8") as f:
        qa_list = json.load(f)

    if args.max and args.max < len(qa_list):
        random.seed(args.seed)
        qa_list = random.sample(qa_list, args.max)
        print(f"采样 {args.max} 题（seed={args.seed}）")

    print(f"共 {len(qa_list)} 题，collection={args.collection}")

    graph = build_graph(args.collection, args.top_k, args.reranker_n)

    # ── Step1: 生成答案 ───────────────────────────────────────────────────────
    samples = []
    fail = 0
    for i, qa in enumerate(qa_list):
        t0  = time.time()
        res = call_rag(qa["question"], graph, reranker_enabled=not args.no_reranker)
        elapsed = round(time.time() - t0, 2)

        if not res["ok"] or not res["answer"]:
            fail += 1

        samples.append({
            "question":   qa["question"],
            "answer":     res["answer"],
            "contexts":   res["contexts"],
            "reference":  qa.get("ground_truth", ""),
            "category":   qa.get("question_type", "factual"),
            "elapsed":    elapsed,
            "ok":         res["ok"],
        })

        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(qa_list)}  失败={fail}")
        time.sleep(INTERVAL)

    print(f"\nRAG 生成完成：{len(samples)} 题，失败 {fail} 题")

    # 过滤掉 answer 为空的
    eval_samples = [s for s in samples if s["answer"]]
    print(f"参与 RAGAS 评估：{len(eval_samples)} 题")

    # ── Step2: RAGAS 评估 ─────────────────────────────────────────────────────
    print("\n运行 RAGAS ...")
    scores = run_ragas(eval_samples)

    for s, sc in zip(eval_samples, scores):
        s.update(sc or {})

    # ── Step3: 汇总 ───────────────────────────────────────────────────────────
    def _agg(key):
        return _nanmean([s.get(key) for s in eval_samples])

    aggregate = {
        "collection":        args.collection,
        "total":             len(samples),
        "evaluated":         len(eval_samples),
        "faithfulness":      _agg("faithfulness"),
        "answer_relevancy":  _agg("answer_relevancy"),
        "context_precision": _agg("context_precision"),
        "context_recall":    _agg("context_recall"),
    }

    # 按题型分组
    from collections import defaultdict
    by_type: dict[str, list] = defaultdict(list)
    for s in eval_samples:
        by_type[s["category"]].append(s)

    by_type_agg = {}
    for t, items in by_type.items():
        by_type_agg[t] = {
            "n":                 len(items),
            "faithfulness":      _nanmean([s.get("faithfulness")      for s in items]),
            "answer_relevancy":  _nanmean([s.get("answer_relevancy")  for s in items]),
            "context_precision": _nanmean([s.get("context_precision") for s in items]),
            "context_recall":    _nanmean([s.get("context_recall")    for s in items]),
        }

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate":    aggregate,
        "by_type":      by_type_agg,
        "per_question": eval_samples,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_no_reranker" if args.no_reranker else ""
    out_path = OUTPUT_DIR / f"{args.collection}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 打印结果 ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  RAGAS 结果 — {args.collection}")
    print("="*55)
    print(f"  Faithfulness:    {aggregate['faithfulness']}")
    print(f"  AnswerRelevancy: {aggregate['answer_relevancy']}")
    print(f"  ContextPrec:     {aggregate['context_precision']}")
    print(f"  ContextRecall:   {aggregate['context_recall']}")
    print(f"\n  按题型：")
    for t, v in by_type_agg.items():
        print(f"    [{t}] n={v['n']}  "
              f"Faith={v['faithfulness']}  Relev={v['answer_relevancy']}  "
              f"Prec={v['context_precision']}  Rec={v['context_recall']}")
    print(f"\n  结果保存至: {out_path}")
    print("="*55)


if __name__ == "__main__":
    main()
