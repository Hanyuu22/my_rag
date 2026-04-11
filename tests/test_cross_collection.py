"""
分库路由跨语言检索实验

验证 CrossCollectionRetriever 是否能彻底修复 D 场景（ZH→EN）。

实验设定：
  - 用 bge-m3 分别建两个临时 in-memory 索引（ZH collection / EN collection）
  - 方案3（上一实验）：混合库 + 双语 query  → D 场景 0/2
  - 方案4（本实验）：分库 + 翻译跨库检索   → D 场景预期 2/2

运行：
  conda activate mineru_2.5
  cd ~/rag_project
  python tests/test_cross_collection.py
"""

import sys, os
import numpy as np
sys.path.insert(0, "/home/hanyuu/rag_project")
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7897"

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

# 查询对（A/B 同语言 + C/D 跨语言，各 2 条）
QUERIES = [
    ("反应器床层温度控制和冷氢的作用",                                         "ZH→ZH", 0, "zh", "zh"),
    ("催化剂预硫化的方法和硫化剂",                                             "ZH→ZH", 5, "zh", "zh"),
    ("temperature control in hydrocracking reactor using quench hydrogen",     "EN→EN", 0, "en", "en"),
    ("catalyst presulfiding procedure and sulfiding agent",                    "EN→EN", 5, "en", "en"),
    ("desulfurization reactor removes sulfur compounds to H2S",                "EN→ZH", 1, "en", "zh"),
    ("nitrogen purging before startup oxygen removal",                         "EN→ZH", 4, "en", "zh"),
    ("分馏塔产品分布，石脑油航煤柴油",                                          "ZH→EN", 3, "zh", "en"),
    ("循环氢压缩机出口压力控制",                                                "ZH→EN", 2, "zh", "en"),
]
# 字段: (query, label, gt_doc_idx, query_lang, doc_lang)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_cross_collection(encode_fn, llm):
    """
    方案4：分库路由
    - ZH query → ZH 库（原始）+ EN 库（翻译后）
    - EN query → EN 库（原始）+ ZH 库（翻译后）
    - RRF 跨库融合
    """
    from retrievers.cross_lingual_retriever import detect_language, translate_query, _rrf_merge

    print(f"\n{'='*60}")
    print(f"  方案4: bge-m3 + 分库路由（CrossCollectionRetriever）")
    print(f"{'='*60}")

    zh_vecs = encode_fn(ZH_DOCS)
    en_vecs = encode_fn(EN_DOCS)

    def search_zh(query, top_k=6):
        q = encode_fn([query])[0]
        sims = [(cosine_sim(q, v), ZH_DOCS[i]) for i, v in enumerate(zh_vecs)]
        sims.sort(reverse=True)
        return sims[:top_k]

    def search_en(query, top_k=6):
        q = encode_fn([query])[0]
        sims = [(cosine_sim(q, v), EN_DOCS[i]) for i, v in enumerate(en_vecs)]
        sims.sort(reverse=True)
        return sims[:top_k]

    results = []
    for query, label, gt_idx, q_lang, doc_lang in QUERIES:
        # 翻译 query
        try:
            trans_q = translate_query(query, q_lang, llm)
        except Exception:
            trans_q = query

        # 根据 query 语言决定检索策略
        if q_lang == "zh":
            primary_results = search_zh(query)     # ZH query → ZH 库
            partner_results = search_en(trans_q)   # 翻译→EN 库
        else:
            primary_results = search_en(query)     # EN query → EN 库
            partner_results = search_zh(trans_q)   # 翻译→ZH 库

        # 哪个库里有 ground truth
        if doc_lang == "zh":
            gt_text = ZH_DOCS[gt_idx]
            gt_score_primary = next((s for s, d in primary_results if d == gt_text), 0.0)
            gt_score_partner = next((s for s, d in partner_results if d == gt_text), 0.0)
        else:
            gt_text = EN_DOCS[gt_idx]
            gt_score_primary = next((s for s, d in primary_results if d == gt_text), 0.0)
            gt_score_partner = next((s for s, d in partner_results if d == gt_text), 0.0)

        # RRF 跨库融合
        merged_scores: dict[str, float] = {}
        for rank, (_, text) in enumerate(primary_results, 1):
            merged_scores[text[:80]] = merged_scores.get(text[:80], 0.0) + 1.0 / (60 + rank)
        for rank, (_, text) in enumerate(partner_results, 1):
            merged_scores[text[:80]] = merged_scores.get(text[:80], 0.0) + 1.0 / (60 + rank)

        # 按 RRF 分排序，取 top-k 候选（实际系统会再过 Reranker）
        ranked_keys = sorted(merged_scores, key=merged_scores.get, reverse=True)
        ranked_texts = [
            next((d for d in ZH_DOCS + EN_DOCS if d[:80] == k), "")
            for k in ranked_keys
        ]

        hit_top1 = (ranked_texts[0] == gt_text) if ranked_texts else False
        hit_top3 = gt_text in ranked_texts[:3]
        hit_top5 = gt_text in ranked_texts[:5]
        gt_rank  = next((i+1 for i, t in enumerate(ranked_texts) if t == gt_text), 999)
        rrf_gt   = merged_scores.get(gt_text[:80], 0.0)

        # GT 在 partner search 里的排名（验证翻译是否有效）
        if doc_lang == "zh":
            partner_rank = next(
                (i+1 for i, (_, t) in enumerate(partner_results) if t == gt_text), 999
            )
        else:
            partner_rank = next(
                (i+1 for i, (_, t) in enumerate(partner_results) if t == gt_text), 999
            )

        flag1 = "✅" if hit_top1 else ("🟡" if hit_top3 else "❌")
        print(f"  {flag1} [{label}] top1={'✓' if hit_top1 else '✗'} top3={'✓' if hit_top3 else '✗'} "
              f"top5={'✓' if hit_top5 else '✗'} | gt_rank={gt_rank} | partner_rank={partner_rank}")
        print(f"     query: {query[:50]}")
        print(f"     译文:  {trans_q[:50]}")

        results.append({
            "label": label,
            "hit_top1": hit_top1, "hit_top3": hit_top3, "hit_top5": hit_top5,
            "gt_rank": gt_rank, "partner_rank": partner_rank,
        })

    by_sc: dict = {}
    for r in results:
        sc = r["label"]
        by_sc.setdefault(sc, []).append(r)

    print(f"\n  场景命中率（top-1 / top-3 / top-5）：")
    for sc, sc_res in sorted(by_sc.items()):
        h1 = sum(r["hit_top1"] for r in sc_res)
        h3 = sum(r["hit_top3"] for r in sc_res)
        h5 = sum(r["hit_top5"] for r in sc_res)
        n  = len(sc_res)
        avg_partner = sum(r["partner_rank"] for r in sc_res) / n
        print(f"    {sc}: top1={h1}/{n}  top3={h3}/{n}  top5={h5}/{n}  "
              f"avg_partner_rank={avg_partner:.1f}")

    t1 = sum(r["hit_top1"] for r in results)
    t3 = sum(r["hit_top3"] for r in results)
    t5 = sum(r["hit_top5"] for r in results)
    n  = len(results)
    print(f"\n  总命中率: top1={t1}/{n}={t1/n:.1%}  top3={t3}/{n}={t3/n:.1%}  top5={t5}/{n}={t5/n:.1%}")
    print(f"  （top-3/5 反映进入 Reranker 候选池后的实际覆盖情况）")
    return results


if __name__ == "__main__":
    print("分库路由跨语言检索实验（方案4）\n")

    print("加载 bge-m3 ...")
    from FlagEmbedding import BGEM3FlagModel
    m3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    def encode_m3(texts):
        return m3.encode(texts, batch_size=8, max_length=512)["dense_vecs"]

    print("加载 LLM（查询翻译用）...")
    from chains.rag_chain import get_llm
    llm = get_llm()

    r4 = run_cross_collection(encode_m3, llm)

    print(f"\n{'='*60}")
    print("  与上一实验对比（记忆值）")
    print(f"{'='*60}")
    print("  场景        方案1(zh单语)  方案2(m3单语)  方案3(m3混合双语)  方案4(m3分库)")
    print("  ZH→ZH       2/2           2/2            2/2                ?")
    print("  EN→EN       2/2           2/2            2/2                ?")
    print("  EN→ZH       0/2           0/2            1/2 ↑              ?")
    print("  ZH→EN       0/2           0/2            0/2                ?")
    by_label = {}
    for r in r4:
        by_label.setdefault(r["label"], []).append(r)
    print(f"  {'场景':<10} {'方案3 top1':>12} {'方案4 top1':>12} {'方案4 top3':>12} {'方案4 top5':>12}")
    prev = {"ZH→ZH": "2/2", "EN→EN": "2/2", "EN→ZH": "1/2", "ZH→EN": "0/2"}
    for label in ["ZH→ZH", "EN→EN", "EN→ZH", "ZH→EN"]:
        if label in by_label:
            h1 = sum(r["hit_top1"] for r in by_label[label])
            h3 = sum(r["hit_top3"] for r in by_label[label])
            h5 = sum(r["hit_top5"] for r in by_label[label])
            n  = len(by_label[label])
            p3 = prev.get(label, "?")
            print(f"  {label:<10} {p3:>12} {h1}/{n:>10} {h3}/{n:>10} {h5}/{n:>10}")
