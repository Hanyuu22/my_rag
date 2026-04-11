"""
GB 标准测试集 RAGAS 评估脚本

将 gb_qa_annotated_*.json 直接接入 RAGAS 四指标评估，
支持 A1/A2/B1/B2 各 collection 对比。

流程：
  1. 读取标注题目（ground_truth 作为 reference）
  2. 对每道题用 Hybrid Retriever 检索 + LLM 生成答案
  3. 送入 RAGAS 计算四指标
  4. 保存结果到 data/results/ragas_{collection}.json

用法：
  conda activate mineru_2.5
  cd ~/rag_project

  # 评估 A1（512 chunk）
  python scripts/eval_ragas_gb.py --collection gb_standards_512 \
      --input data/qa_dataset/gb_qa_annotated_512.json

  # 评估 A2（1024 chunk）
  python scripts/eval_ragas_gb.py --collection gb_standards_1024 \
      --input data/qa_dataset/gb_qa_annotated_1024.json

  # 只跑前 50 题快速验证
  python scripts/eval_ragas_gb.py --collection gb_standards_512 \
      --input data/qa_dataset/gb_qa_annotated_512.json --limit 50

  # 跳过生成步骤，直接用已有草稿跑 RAGAS
  python scripts/eval_ragas_gb.py --collection gb_standards_512 \
      --draft data/results/draft_gb_standards_512.json --ragas-only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/home/hanyuu/rag_project")


# ── 步骤 1：加载检索器 + LLM ──────────────────────────────────────────────

def build_components(collection_name: str, top_k: int):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI
    from chains.rag_chain import get_embeddings, get_collection_embedding_model
    from retrievers.hybrid_retriever import build_hybrid_retriever
    from config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL

    model_name  = get_collection_embedding_model(collection_name)
    embeddings  = get_embeddings(model_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )
    all_data = vectorstore._collection.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    print(f"  向量库 '{collection_name}'：{len(chunks)} chunks")
    retriever = build_hybrid_retriever(chunks, vectorstore, k=top_k)

    llm = ChatOpenAI(
        model=LLM_MODEL, temperature=0.1,
        api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL,
    )
    return retriever, llm


# ── 步骤 2：生成答案 + contexts ───────────────────────────────────────────

ANSWER_PROMPT = """\
你是石油化工领域的专业助手。请根据以下参考资料回答问题，答案要简洁准确，保留关键数值和单位。
如果参考资料中没有相关信息，请回答"根据现有资料无法确定"。

【参考资料】
{context}

【问题】
{question}

【答案】"""


def generate_answers(qa_list: list, retriever, llm, draft_path: Path) -> list:
    """
    对每道题检索 + 生成，结果实时写入 draft_path（断点续传）。
    返回 draft 列表。
    """
    # 读已有草稿
    existing: dict[int, dict] = {}
    if draft_path.exists():
        with open(draft_path, encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["idx"]] = item
        print(f"  断点续传：已有 {len(existing)} 条草稿")

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)

    results = dict(existing)

    todo = [
        (i, qa) for i, qa in enumerate(qa_list)
        if i not in existing or not existing[i].get("answer")
    ]
    print(f"  待生成：{len(todo)} 题\n")

    for step, (i, qa) in enumerate(todo, 1):
        question  = qa["question"]
        reference = qa.get("ground_truth", "")

        try:
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            messages = prompt.format_messages(context=context, question=question)
            answer = llm.invoke(messages).content.strip()
        except Exception as e:
            print(f"  ❌ [{step}/{len(todo)}] 失败：{e}")
            answer   = ""
            docs     = []
            context  = ""

        results[i] = {
            "idx":           i,
            "question":      question,
            "answer":        answer,
            "reference":     reference,
            "contexts":      [d.page_content for d in docs],
            "question_type": qa.get("question_type", ""),
            "difficulty":    qa.get("difficulty", ""),
            "topic_tag":     qa.get("topic_tag", ""),
        }

        if step % 10 == 0 or step == len(todo):
            _save_draft(draft_path, results)
            print(f"  [{step}/{len(todo)}] 已保存草稿")

        if step < len(todo):
            time.sleep(0.5)   # 防止 API 限速

    _save_draft(draft_path, results)
    return [results[k] for k in sorted(results.keys())]


def _save_draft(path: Path, results: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [results[k] for k in sorted(results.keys())]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


# ── 步骤 3：RAGAS 评估 ────────────────────────────────────────────────────

def run_ragas(samples: list, collection_name: str) -> dict:
    import numpy as np
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from chains.rag_chain import get_llm, get_embeddings, get_collection_embedding_model

    model_name = get_collection_embedding_model(collection_name)
    ragas_llm  = LangchainLLMWrapper(get_llm())
    ragas_emb  = LangchainEmbeddingsWrapper(get_embeddings(model_name))
    run_config = RunConfig(timeout=120, max_retries=2, max_wait=30)

    # 过滤掉 answer 为空的
    valid = [s for s in samples if s.get("answer") and s.get("contexts")]
    print(f"\n[RAGAS] 有效样本：{len(valid)}/{len(samples)}")

    eval_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("reference", ""),
        )
        for s in valid
    ]
    dataset = EvaluationDataset(samples=eval_samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm, embeddings=ragas_emb,
        show_progress=True, raise_exceptions=False,
        run_config=run_config,
    )

    def safe_mean(key):
        try:
            vals = result[key]
            m = float(np.nanmean(vals))
            return round(m, 4) if not np.isnan(m) else None
        except Exception:
            return None

    scores = {
        "faithfulness":      safe_mean("faithfulness"),
        "answer_relevancy":  safe_mean("answer_relevancy"),
        "context_precision": safe_mean("context_precision"),
        "context_recall":    safe_mean("context_recall"),
    }

    # 按题型分组
    group_scores = {}
    for key in ["question_type", "difficulty", "topic_tag"]:
        groups: dict[str, list] = {}
        for i, s in enumerate(valid):
            g = s.get(key, "unknown")
            groups.setdefault(g, []).append(i)
        group_scores[key] = {}
        for g, idxs in groups.items():
            group_scores[key][g] = {
                metric: round(float(np.nanmean([result[metric][i] for i in idxs])), 4)
                for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                if metric in result
            }

    return {"aggregate": scores, "group_scores": group_scores, "n_evaluated": len(valid)}


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GB 测试集 RAGAS 评估")
    parser.add_argument("--collection", default="gb_standards_512")
    parser.add_argument("--input",      default="data/qa_dataset/gb_qa_annotated_512.json")
    parser.add_argument("--limit",      type=int, default=None, help="只跑前 N 题（默认全部）")
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--draft",      default=None, help="草稿文件路径（默认自动按 collection 命名）")
    parser.add_argument("--ragas-only", action="store_true", help="跳过生成，直接读草稿跑 RAGAS")
    args = parser.parse_args()

    draft_path = Path(args.draft) if args.draft else \
        Path(f"data/results/draft_{args.collection}.json")
    out_path   = Path(f"data/results/ragas_{args.collection}.json")

    if not args.ragas_only:
        # 读题目
        with open(args.input, encoding="utf-8") as f:
            qa_list = json.load(f)
        valid_qa = [qa for qa in qa_list if qa.get("ground_truth_chunk_id")]
        if args.limit:
            valid_qa = valid_qa[:args.limit]
        print(f"题目：{len(valid_qa)} 道（collection={args.collection}）")

        print("\n[步骤1] 加载检索器...")
        retriever, llm = build_components(args.collection, args.top_k)

        print("\n[步骤2] 生成答案...")
        samples = generate_answers(valid_qa, retriever, llm, draft_path)
    else:
        print(f"[跳过生成] 读取草稿：{draft_path}")
        with open(draft_path, encoding="utf-8") as f:
            samples = json.load(f)

    print("\n[步骤3] RAGAS 评估...")
    ragas_result = run_ragas(samples, args.collection)

    # 打印
    agg = ragas_result["aggregate"]
    n   = ragas_result["n_evaluated"]
    print(f"\n{'='*55}")
    print(f"RAGAS 评估结果  collection={args.collection}  n={n}")
    print(f"{'='*55}")
    print(f"  Faithfulness:      {agg['faithfulness']}")
    print(f"  Answer Relevancy:  {agg['answer_relevancy']}")
    print(f"  Context Precision: {agg['context_precision']}")
    print(f"  Context Recall:    {agg['context_recall']}")

    for group_key, groups in ragas_result["group_scores"].items():
        print(f"\n  ── 按 {group_key} 分组 ──")
        for g, s in sorted(groups.items()):
            print(f"    {g:<18} CR={s.get('context_recall','?')}  CP={s.get('context_precision','?')}"
                  f"  Faith={s.get('faithfulness','?')}")

    # 保存
    out = {
        "collection":    args.collection,
        "evaluated_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_evaluated":   n,
        **ragas_result,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{out_path}")


if __name__ == "__main__":
    main()
