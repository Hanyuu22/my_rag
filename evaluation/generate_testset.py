"""
Golden Set 自动生成（RAGAS TestsetGenerator）

从已有的 Chroma 向量库中读取 chunks，用 RAGAS TestsetGenerator
自动生成问题 + 标准参考答案，无需人工标注。

生成的题目类型：
  SingleHopSpecific    — 直接从单段文本回答的精确问题（60%）
  MultiHopAbstract     — 需要跨段落抽象推理的问题（20%）
  MultiHopSpecific     — 需要跨段落精确检索的问题（20%）

用法：
    conda activate mineru_2.5
    cd ~/rag_project

    # 生成 30 题（推荐先试小规模）
    python evaluation/generate_testset.py --collection hydro_manual --size 30

    # 生成 50 题（完整版）
    python evaluation/generate_testset.py --collection hydro_manual --size 50

    # 限制喂入 chunks 数量（节省 API 费用，默认取前 300 个 chunks）
    python evaluation/generate_testset.py --collection hydro_manual --size 30 --max-chunks 200

注意：
  TestsetGenerator 要对每个 chunk 做多次 LLM 调用来构建知识图谱，
  chunks 越多 = API 费用越高、时间越长。建议：
    --max-chunks 200~300 足够生成 30~50 道高质量题目
"""

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")

import argparse
import json
import random
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="RAGAS 自动生成测试集")
    parser.add_argument("--collection",  required=True,  help="Chroma collection 名称")
    parser.add_argument("--size",        type=int, default=30,  help="生成题目数量（默认 30）")
    parser.add_argument("--max-chunks",  type=int, default=300, help="喂给 Generator 的最大 chunk 数（默认 300）")
    parser.add_argument("--output",      default="evaluation/golden_set_auto.json",
                        help="输出文件（默认 evaluation/golden_set_auto.json）")
    parser.add_argument("--seed",        type=int, default=42,  help="随机种子（默认 42）")
    return parser.parse_args()


def load_chunks_from_chroma(collection: str, max_chunks: int):
    """从 Chroma 读取 chunks，转成 LangChain Document 列表"""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from config import CHROMA_PERSIST_DIR
    from chains.rag_chain import get_embeddings

    print(f"[加载] 从 Chroma collection '{collection}' 读取 chunks ...")
    vectorstore = Chroma(
        collection_name=collection,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=get_embeddings(),
    )
    results = vectorstore._collection.get(include=["documents", "metadatas"])
    all_chunks = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
    total = len(all_chunks)
    print(f"  共 {total} 个 chunks")

    # 过滤掉过短的 chunk（太短的 chunk 很难生成有意义的问题）
    filtered = [c for c in all_chunks if len(c.page_content.strip()) >= 50]
    print(f"  过滤后（≥50字）：{len(filtered)} 个 chunks")

    # 随机采样（保证覆盖全文，而不是只取前 N 个）
    if len(filtered) > max_chunks:
        random.seed(42)
        filtered = random.sample(filtered, max_chunks)
        print(f"  随机采样 {max_chunks} 个 chunks 用于生成")

    return filtered


def build_ragas_wrappers():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from chains.rag_chain import get_llm, get_embeddings

    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())
    return ragas_llm, ragas_emb


def run_generator(chunks, size: int, ragas_llm, ragas_emb):
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )
    from ragas.testset.persona import Persona
    from ragas.run_config import RunConfig

    # 中文 persona：让 LLM 以中文用户视角出题
    personas = [
        Persona(
            name="石化工艺工程师",
            role_description="一位熟悉加氢裂化装置的工艺工程师，使用中文提问，关注操作规程、工艺参数和安全规定。",
        ),
        Persona(
            name="装置操作员",
            role_description="加氢裂化装置的现场操作人员，使用中文提问，关注开停工步骤、设备操作和异常处理。",
        ),
    ]

    # 中文出题上下文
    llm_context = (
        "本知识库包含一份中文石油化工技术文档：240万吨加氢裂化装置工艺技术规程。"
        "请务必用中文生成所有问题和答案，不要使用英文或拼音。"
    )

    # 题型分布
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.60),
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm),  0.20),
        (MultiHopSpecificQuerySynthesizer(llm=ragas_llm),  0.20),
    ]

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_emb,
        persona_list=personas,
        llm_context=llm_context,
    )

    run_config = RunConfig(
        timeout=180,
        max_retries=3,
        max_wait=60,
    )

    print(f"\n[生成] 开始生成 {size} 道题目（chunk 数={len(chunks)}）...")
    print("  这一步会进行大量 LLM 调用来构建知识图谱，预计需要 5~15 分钟，请耐心等待。\n")

    testset = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=size,
        query_distribution=query_distribution,
        run_config=run_config,
        raise_exceptions=False,
    )
    return testset


def save_testset(testset, output_path: str):
    """把 RAGAS Testset 转成我们的 JSON 格式"""
    df = testset.to_pandas()
    print(f"\n[结果] 成功生成 {len(df)} 道题")
    print(df[["user_input", "synthesizer_name"]].to_string(index=False))

    items = []
    for i, row in df.iterrows():
        items.append({
            "id":        i + 1,
            "category":  row.get("synthesizer_name", "auto"),
            "question":  row.get("user_input", ""),
            "reference": row.get("reference_answer", row.get("reference", "")),
            # 以下字段在跑 run_ragas_eval.py 时填入
            "answer":    "",
            "contexts":  [],
            "scores":    [],
            "route":     "",
            "elapsed":   None,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] 测试集已保存到：{out_path}")
    print("下一步：")
    print(f"  1. 跑 RAG 生成答案：python evaluation/run_generate.py --collection <name> --input {output_path} --output evaluation/draft_auto.json")
    print(f"  2. RAGAS 评估：     python evaluation/run_ragas_eval.py --input evaluation/draft_auto.json")
    return items


def main():
    args = parse_args()
    random.seed(args.seed)

    chunks = load_chunks_from_chroma(args.collection, args.max_chunks)
    ragas_llm, ragas_emb = build_ragas_wrappers()

    try:
        testset = run_generator(chunks, args.size, ragas_llm, ragas_emb)
        save_testset(testset, args.output)
    except Exception as e:
        print(f"\n[错误] 生成失败：{e}")
        print("\n常见原因及解决方法：")
        print("  1. API 超时 → 减少 --max-chunks（比如改为 150）")
        print("  2. chunks 太短无法生成题目 → 检查知识库内容")
        print("  3. LLM 拒绝输出 → 检查 API key 和配额")
        raise


if __name__ == "__main__":
    main()
