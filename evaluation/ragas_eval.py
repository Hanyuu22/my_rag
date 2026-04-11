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


def run_ragas(
    samples: List[Dict[str, Any]],
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    对一批样本运行 RAGAS 四指标评估。

    Args:
        samples: 样本列表，每个样本包含：
            - question (str): 用户问题
            - answer (str): 系统生成的答案
            - contexts (List[str]): 检索到的 context 文本列表
            - reference (str): ground truth 参考答案
        show_progress: 是否显示进度条

    Returns:
        四个指标均值的 dict
    """
    from chains.rag_chain import get_llm, get_embeddings

    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

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
        timeout=120,       # 单个评估任务超时秒数（默认 60s，DashScope 有时较慢）
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

    # result[key] 返回每个样本的分数列表，取均值
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        try:
            vals = result[key]  # List[float]，NaN 表示该样本评估失败
            mean = float(np.nanmean(vals))
            scores[key] = round(mean, 4) if not np.isnan(mean) else None
        except (KeyError, Exception):
            scores[key] = None

    return scores
