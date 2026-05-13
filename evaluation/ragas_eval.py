"""
Level 4 — RAGAS 评估封装

指标说明：
  Faithfulness        答案是否有事实依据（声明都能在 context 中找到支撑）
  AnswerRelevancy     答案是否切题（与问题的相关性）
  ContextPrecision    检索到的 context 是否都有用（精确率，有 ground truth 版）
  ContextRecall       正确答案所需的信息是否都被检索到（召回率，需 ground truth）

用法：
    from evaluation.ragas_eval import run_ragas

    samples = [
        {
            "question": "冷氢的作用是什么？",
            "answer": "...",
            "contexts": ["...", "..."],
            "reference": "..."   # ground truth，用于 context_recall / context_precision
        },
        ...
    ]
    result = run_ragas(samples)
    # {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
"""

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")

from typing import List, Dict, Any

import numpy as np
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def run_ragas(
    samples: List[Dict[str, Any]],
    show_progress: bool = True,
    return_per_sample: bool = False,
) -> Dict[str, Any]:
    """
    对一批样本运行 RAGAS 四指标评估。

    Args:
        samples: 样本列表，每个样本包含：
            - question (str): 用户问题
            - answer (str): 系统生成的答案
            - contexts (List[str]): 检索到的 context 文本列表
            - reference (str): ground truth 参考答案
        show_progress: 是否显示进度条
        return_per_sample: 为 True 时额外返回 per_sample 字段（每道题的逐条分数列表）

    Returns:
        四个指标均值的 dict；若 return_per_sample=True，额外包含
        "per_sample": {"faithfulness": [...], ...}
    """
    from chains.rag_chain import get_llm, get_embeddings

    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

    # strictness=1：每道题只生成 1 次，避免 DashScope 不支持 n>1 导致的大量警告
    answer_relevancy.strictness = 1

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    eval_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("reference", ""),
        )
        for s in samples
    ]

    dataset = EvaluationDataset(samples=eval_samples)

    run_config = RunConfig(
        timeout=120,
        max_retries=2,
        max_wait=30,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        show_progress=show_progress,
        raise_exceptions=False,
        run_config=run_config,
    )

    per_sample = {}
    scores = {}
    for key in METRIC_KEYS:
        try:
            vals = [float(v) for v in result[key]]
            per_sample[key] = vals
            mean = float(np.nanmean(vals))
            scores[key] = round(mean, 4) if not np.isnan(mean) else None
        except (KeyError, Exception):
            per_sample[key] = []
            scores[key] = None

    if return_per_sample:
        scores["per_sample"] = per_sample
    return scores


def aggregate_group_scores(
    samples: List[Dict[str, Any]],
    per_sample: Dict[str, List[float]],
    group_key: str,
) -> Dict[str, Dict]:
    """
    从已有的逐条分数按 group_key 分组聚合，不再重复调用 LLM。

    Args:
        samples: 与 run_ragas 传入相同的样本列表（含 group_key 字段）
        per_sample: run_ragas 返回的 per_sample 字典
        group_key: 分组字段名，如 "question_type"

    Returns:
        {group_name: {"n": int, "faithfulness": float, ...}}
    """
    from collections import defaultdict

    group_indices: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        group_indices[s.get(group_key, "unknown")].append(i)

    result = {}
    for gname, indices in sorted(group_indices.items()):
        group_scores = {}
        for key in METRIC_KEYS:
            vals_all = per_sample.get(key, [])
            vals = [vals_all[i] for i in indices if i < len(vals_all)]
            mean = float(np.nanmean(vals)) if vals else float("nan")
            group_scores[key] = round(mean, 4) if not np.isnan(mean) else None
        result[gname] = {"n": len(indices), **group_scores}
    return result
