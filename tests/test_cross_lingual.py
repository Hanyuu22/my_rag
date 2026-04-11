"""
跨语言检索对比实验：单语 query vs 双语 query（CrossLingualRetriever）

对比三种方案在混合中英文库中的检索表现：
  方案1 baseline：bge-large-zh + 单语 query（现有系统）
  方案2 m3 only： bge-m3 + 单语 query
  方案3 cross：   bge-m3 + 双语 query（CrossLingualRetriever）

测试重点：C/D 场景（跨语言），看方案3是否能修复命中率。

运行：
  conda activate mineru_2.5
  cd ~/rag_project
  python tests/test_cross_lingual.py
"""

import sys, os, time
import numpy as np
sys.path.insert(0, "/home/hanyuu/rag_project")
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7897"

# ── 测试语料（与上次实验相同） ─────────────────────────────────────────────

ZH_DOCS = [
    "加氢裂化反应器R101采用多床层结构，床层间设有冷氢注入点，通过注入冷氢控制床层温升，防止超温。",
    "精制反应器R102主要作用是脱硫、脱氮，将原料中的含硫化合物转化为硫化氢，通过高压分离器排出系统。",
    "循环氢压缩机是维持反应系统氢分压的关键设备，正常运行时需保证出口压力和流量在设计范围内。",
    "分馏塔将加氢裂化产物按沸点分离，塔顶得到石脑油，侧线抽出航煤和柴油，塔底为未转化尾油。",
    "装置开工前需进行氮气置换，将系统含氧量降至0.5%以下，防止引入氢气时发生爆炸。",
    "催化剂硫化采用二甲基二硫醚（DMDS）作为硫化剂，在规定温度程序下完成活性金属的预硫化。",
]

EN_DOCS = [
    "The hydrocracking reactor R101 uses a multi-bed structure with cold hydrogen injection points between beds to control temperature rise and prevent overheating.",
    "The hydrotreating reactor R102 is primarily designed for desulfurization and denitrogenation, converting sulfur compounds to H2S which is removed via high-pressure separators.",
    "The recycle hydrogen compressor is critical for maintaining hydrogen partial pressure in the reaction system; outlet pressure and flow must stay within design specifications.",
    "The fractionation column separates hydrocracking products by boiling point: naphtha at the top, jet fuel and diesel as side draws, and unconverted tail oil at the bottom.",
    "Before startup, nitrogen purging reduces the oxygen content below 0.5% to prevent explosion when hydrogen is introduced into the system.",
    "Catalyst sulfiding uses dimethyl disulfide (DMDS) as the sulfiding agent, pre-sulfiding active metals following a specified temperature program.",
]

ALL_DOCS = ZH_DOCS + EN_DOCS

QUERIES = [
    ("反应器床层温度控制和冷氢的作用",              "ZH", 0,  "A: ZH→ZH"),
    ("催化剂预硫化的方法和硫化剂",                  "ZH", 5,  "A: ZH→ZH"),
    ("temperature control in hydrocracking reactor using quench hydrogen", "EN", 6,  "B: EN→EN"),
    ("catalyst presulfiding procedure and sulfiding agent",                "EN", 11, "B: EN→EN"),
    ("desulfurization reactor removes sulfur compounds to H2S",            "EN", 1,  "C: EN→ZH"),
    ("nitrogen purging before startup oxygen removal",                     "EN", 4,  "C: EN→ZH"),
    ("分馏塔产品分布，石脑油航煤柴油",              "ZH", 9,  "D: ZH→EN"),
    ("循环氢压缩机出口压力控制",                    "ZH", 8,  "D: ZH→EN"),
]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_simple_retriever(encode_fn, docs):
    """返回一个简单的向量检索函数 query→ranked docs"""
    doc_vecs = encode_fn(docs)
    def retrieve(query: str, top_k: int = 10):
        q_vec = encode_fn([query])[0]
        sims = [(cosine_sim(q_vec, dv), i) for i, dv in enumerate(doc_vecs)]
        sims.sort(reverse=True)
        return [(score, docs[idx]) for score, idx in sims[:top_k]]
    return retrieve


def rrf_merge(result_lists, k=60, top_k=10):
    """RRF 融合多路检索结果，返回排序后的 (score, doc_idx, doc) 列表"""
    scores = {}
    idx_map = {}
    for result_list in result_lists:
        for rank, (_, doc) in enumerate(result_list, start=1):
            key = doc[:80]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            idx_map[key] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, text) for key, score in ranked[:top_k] for text in [idx_map[key]]]


def run_experiment(name, encode_fn, use_cross_lingual=False, llm=None):
    print(f"\n{'='*60}")
    print(f"  方案: {name}")
    print(f"{'='*60}")

    doc_vecs = encode_fn(ALL_DOCS)
    retrieve_fn = build_simple_retriever(encode_fn, ALL_DOCS)

    results = []
    for query, lang, gt_idx, scenario in QUERIES:
        if use_cross_lingual:
            # 翻译 query
            from retrievers.cross_lingual_retriever import detect_language, translate_query
            q_lang = detect_language(query)
            try:
                trans_query = translate_query(query, q_lang, llm)
            except Exception:
                trans_query = query

            # 双路检索
            results_orig  = retrieve_fn(query)
            results_trans = retrieve_fn(trans_query)

            # RRF 合并（只用文本做 key）
            merged_scores = {}
            for rank, (_, doc_text) in enumerate(results_orig, 1):
                merged_scores[doc_text[:80]] = merged_scores.get(doc_text[:80], 0.0) + 1.0 / (60 + rank)
            for rank, (_, doc_text) in enumerate(results_trans, 1):
                merged_scores[doc_text[:80]] = merged_scores.get(doc_text[:80], 0.0) + 1.0 / (60 + rank)

            top1_text = max(merged_scores, key=merged_scores.get)
            top1_idx  = next(i for i, d in enumerate(ALL_DOCS) if d[:80] == top1_text)
            top1_score = merged_scores[top1_text]
            gt_score = merged_scores.get(ALL_DOCS[gt_idx][:80], 0.0)
            extra = f" → 译文: {trans_query[:35]}…"
        else:
            trans_query = ""
            top_results = retrieve_fn(query)
            top1_score, top1_text = top_results[0]
            top1_idx = next(i for i, d in enumerate(ALL_DOCS) if d[:80] == top1_text[:80])

            q_vec = encode_fn([query])[0]
            gt_score = cosine_sim(q_vec, encode_fn([ALL_DOCS[gt_idx]])[0])
            extra = ""

        hit = (top1_idx == gt_idx)
        flag = "✅" if hit else "❌"
        print(f"  {flag} [{scenario}] top1={top1_score:.3f} gt={gt_score:.3f}{extra}")
        print(f"     query: {query[:50]}")

        results.append({
            "scenario": scenario[0],
            "hit": hit,
            "top1_score": top1_score,
            "gt_score": gt_score,
        })

    from collections import defaultdict
    by_sc = defaultdict(list)
    for r in results:
        by_sc[r["scenario"]].append(r)

    print(f"\n  场景命中率：")
    for sc in sorted(by_sc):
        sc_res = by_sc[sc]
        hits = sum(r["hit"] for r in sc_res)
        avg_gt = np.mean([r["gt_score"] for r in sc_res])
        print(f"    场景{sc}: {hits}/{len(sc_res)} 命中 | avg_gt={avg_gt:.3f}")

    total = sum(r["hit"] for r in results)
    print(f"\n  总命中率: {total}/{len(results)} = {total/len(results):.1%}")
    return results


def print_summary(r1, r2, r3):
    from collections import defaultdict
    print(f"\n{'='*70}")
    print(f"  三方案对比汇总")
    print(f"{'='*70}")
    print(f"  {'场景':<6} {'基准(zh单语)':>12} {'m3单语':>12} {'m3双语(跨语)':>14}")
    print(f"  {'-'*50}")

    def by_sc(results):
        d = defaultdict(list)
        for r in results:
            d[r["scenario"]].append(r)
        return d

    d1, d2, d3 = by_sc(r1), by_sc(r2), by_sc(r3)
    for sc in "ABCD":
        if sc not in d1:
            continue
        h1 = sum(r["hit"] for r in d1[sc])
        h2 = sum(r["hit"] for r in d2[sc])
        h3 = sum(r["hit"] for r in d3[sc])
        n  = len(d1[sc])
        g1 = np.mean([r["gt_score"] for r in d1[sc]])
        g2 = np.mean([r["gt_score"] for r in d2[sc]])
        g3 = np.mean([r["gt_score"] for r in d3[sc]])
        print(f"  场景{sc}   {h1}/{n} gt={g1:.3f}   {h2}/{n} gt={g2:.3f}   {h3}/{n} gt={g3:.3f}")

    tot1 = sum(r["hit"] for r in r1)
    tot2 = sum(r["hit"] for r in r2)
    tot3 = sum(r["hit"] for r in r3)
    n = len(r1)
    print(f"  {'-'*50}")
    print(f"  {'总计':<6} {tot1}/{n} = {tot1/n:.1%}   {tot2}/{n} = {tot2/n:.1%}   {tot3}/{n} = {tot3/n:.1%}")
    print()
    # 结论
    if tot3 > tot2:
        print(f"  ✅ 双语 query 策略有效：命中率 {tot2/n:.1%} → {tot3/n:.1%} (+{(tot3-tot2)/n*100:.0f}pp)")
    else:
        print(f"  ⚠️  双语 query 未能进一步提升命中率，gt_score 变化见上表")


if __name__ == "__main__":
    print("跨语言检索三方案对比实验")
    print(f"语料: {len(ALL_DOCS)} 条（{len(ZH_DOCS)}中 + {len(EN_DOCS)}英）")
    print(f"查询: {len(QUERIES)} 条，A/B/C/D 各2条\n")

    # 加载 bge-large-zh
    print("加载 bge-large-zh-v1.5 ...")
    from langchain_huggingface import HuggingFaceEmbeddings
    emb_zh = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    def encode_zh(texts):
        return np.array(emb_zh.embed_documents(texts))

    # 加载 bge-m3
    print("加载 bge-m3 ...")
    from FlagEmbedding import BGEM3FlagModel
    m3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    def encode_m3(texts):
        return m3.encode(texts, batch_size=8, max_length=512)["dense_vecs"]

    # 加载 LLM（翻译用）
    print("加载 LLM（查询翻译用）...")
    from chains.rag_chain import get_llm
    llm = get_llm()

    # 跑三个方案
    r1 = run_experiment("方案1: bge-large-zh + 单语query (baseline)", encode_zh)
    r2 = run_experiment("方案2: bge-m3 + 单语query",                  encode_m3)
    r3 = run_experiment("方案3: bge-m3 + 双语query (CrossLingual)",   encode_m3,
                        use_cross_lingual=True, llm=llm)

    print_summary(r1, r2, r3)
