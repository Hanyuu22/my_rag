"""
多语言 Embedding 对比实验：bge-large-zh-v1.5 vs bge-m3

实验设计：
  构造一批 中/英 双语文档片段（加氢裂化工艺领域），
  用相同的查询分别走两个模型，比较 Top-1 相关分和检索准确率。

四类测试场景：
  A. ZH query → ZH doc   (纯中文场景，bge-zh 应领先或持平)
  B. EN query → EN doc   (纯英文场景，bge-zh 预期失效)
  C. EN query → ZH doc   (跨语言：英文查中文库)
  D. ZH query → EN doc   (跨语言：中文查英文库)

运行：
  conda activate mineru_2.5
  cd ~/rag_project
  python tests/test_multilingual_embedding.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, "/home/hanyuu/rag_project")

# ── 测试语料（中英对照，加氢裂化工艺领域） ─────────────────────────────────

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

# ── 查询对（中英对照，涵盖 4 类场景） ─────────────────────────────────────

QUERIES = [
    # (query_text, language, ground_truth_doc_idx, scenario)
    # A: ZH → ZH
    ("反应器床层温度控制和冷氢的作用", "ZH", 0, "A: ZH→ZH"),
    ("催化剂预硫化的方法和硫化剂", "ZH", 5, "A: ZH→ZH"),
    # B: EN → EN
    ("temperature control in hydrocracking reactor using quench hydrogen", "EN", 0, "B: EN→EN"),
    ("catalyst presulfiding procedure and sulfiding agent", "EN", 5, "B: EN→EN"),
    # C: EN query → ZH doc
    ("desulfurization reactor removes sulfur compounds to H2S", "EN", 1, "C: EN→ZH"),
    ("nitrogen purging before startup oxygen removal", "EN", 4, "C: EN→ZH"),
    # D: ZH query → EN doc
    ("分馏塔产品分布，石脑油航煤柴油", "ZH", 3, "D: ZH→EN"),
    ("循环氢压缩机出口压力控制", "ZH", 2, "D: ZH→EN"),
]

ALL_DOCS = ZH_DOCS + EN_DOCS  # index 0~5: ZH, 6~11: EN

# ground_truth_doc_idx 含义：
#   A/B 场景：在同语言 docs 里的索引（ZH=0~5, EN 用 idx+6）
#   C 场景：在 ZH_DOCS 里的索引
#   D 场景：在 EN_DOCS 里的索引（实际在 ALL_DOCS 里是 idx+6）


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def evaluate_model(model_name: str, encode_fn, docs: list[str], queries: list[tuple]) -> dict:
    """
    encode_fn(texts) → np.ndarray (N, dim)
    返回每个 query 的 top1_score 和是否命中 ground_truth
    """
    print(f"\n{'='*60}")
    print(f"  模型: {model_name}")
    print(f"{'='*60}")

    t0 = time.time()
    doc_vecs = encode_fn(docs)
    embed_time = time.time() - t0
    print(f"  文档向量化: {embed_time:.1f}s ({len(docs)} 条)")

    results = []
    for query_text, lang, gt_idx, scenario in queries:
        # 根据场景确定 ground truth 在 ALL_DOCS 中的实际索引
        if scenario.startswith("A"):
            actual_gt = gt_idx              # ZH_DOCS 范围内
        elif scenario.startswith("B"):
            actual_gt = gt_idx + len(ZH_DOCS)  # EN_DOCS 范围
        elif scenario.startswith("C"):
            actual_gt = gt_idx              # EN query → ZH doc
        else:  # D
            actual_gt = gt_idx + len(ZH_DOCS)  # ZH query → EN doc

        q_vec = encode_fn([query_text])[0]
        sims = [cosine_sim(q_vec, d) for d in doc_vecs]
        top1_idx = int(np.argmax(sims))
        top1_score = sims[top1_idx]
        gt_score = sims[actual_gt]
        hit = (top1_idx == actual_gt)

        results.append({
            "scenario": scenario,
            "query": query_text[:35] + "…" if len(query_text) > 35 else query_text,
            "gt_idx": actual_gt,
            "top1_idx": top1_idx,
            "top1_score": top1_score,
            "gt_score": gt_score,
            "hit": hit,
        })

        flag = "✅" if hit else "❌"
        print(f"  {flag} [{scenario}] top1={top1_score:.3f} gt={gt_score:.3f} | {query_text[:40]}")

    # 按场景分组统计
    from collections import defaultdict
    by_scenario: dict = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"][0]].append(r)

    print(f"\n  场景准确率汇总：")
    total_hit = 0
    for sc_key in sorted(by_scenario):
        sc_results = by_scenario[sc_key]
        hit_cnt = sum(r["hit"] for r in sc_results)
        avg_top1 = np.mean([r["top1_score"] for r in sc_results])
        avg_gt   = np.mean([r["gt_score"] for r in sc_results])
        print(f"    场景{sc_key}: {hit_cnt}/{len(sc_results)} 命中 | "
              f"avg_top1={avg_top1:.3f} avg_gt={avg_gt:.3f}")
        total_hit += hit_cnt

    acc = total_hit / len(results)
    print(f"\n  总命中率: {total_hit}/{len(results)} = {acc:.1%}")
    return {"model": model_name, "results": results, "accuracy": acc}


def load_bge_zh():
    """加载 bge-large-zh-v1.5"""
    print("\n[1/2] 加载 bge-large-zh-v1.5 ...")
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    def encode_fn(texts):
        vecs = emb.embed_documents(texts)
        return np.array(vecs)

    return encode_fn


def load_bge_m3():
    """加载 bge-m3（FlagEmbedding，自动下载）"""
    print("\n[2/2] 加载 bge-m3 ...")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        device="cuda",
    )

    def encode_fn(texts):
        out = model.encode(texts, batch_size=8, max_length=512)
        return out["dense_vecs"]  # shape (N, 1024)

    return encode_fn


def print_comparison(r1: dict, r2: dict):
    """输出两模型的对比表格"""
    print(f"\n{'='*70}")
    print(f"  对比总结：bge-large-zh-v1.5  vs  bge-m3")
    print(f"{'='*70}")

    res1 = {(r["scenario"], r["query"]): r for r in r1["results"]}
    res2 = {(r["scenario"], r["query"]): r for r in r2["results"]}

    print(f"  {'场景':<12} {'bge-zh top1':>12} {'bge-m3 top1':>12} {'bge-zh hit':>10} {'bge-m3 hit':>10}")
    print(f"  {'-'*60}")

    from collections import defaultdict
    by_sc1: dict = defaultdict(list)
    by_sc2: dict = defaultdict(list)
    for r in r1["results"]:
        by_sc1[r["scenario"][0]].append(r)
    for r in r2["results"]:
        by_sc2[r["scenario"][0]].append(r)

    for sc in sorted(by_sc1):
        s1 = by_sc1[sc]
        s2 = by_sc2[sc]
        avg1 = np.mean([r["top1_score"] for r in s1])
        avg2 = np.mean([r["top1_score"] for r in s2])
        h1 = sum(r["hit"] for r in s1)
        h2 = sum(r["hit"] for r in s2)
        winner = "bge-m3 ↑" if avg2 > avg1 + 0.02 else ("bge-zh ↑" if avg1 > avg2 + 0.02 else "持平")
        print(f"  场景{sc:<9} {avg1:>12.3f} {avg2:>12.3f} {h1}/{len(s1):>8} {h2}/{len(s2):>8}  {winner}")

    print(f"  {'-'*60}")
    acc1 = r1["accuracy"]
    acc2 = r2["accuracy"]
    print(f"  {'总命中率':<12} {acc1:>12.1%} {acc2:>12.1%}")
    print()
    print("  结论：")
    if acc2 > acc1:
        print(f"  bge-m3 总体命中率更高（+{(acc2-acc1)*100:.0f}pp），适合多语言/混合文档库")
    elif acc1 > acc2:
        print(f"  bge-large-zh 总体命中率更高，纯中文场景可保留现有模型")
    else:
        print(f"  两模型总体命中率相同，但关注场景B/C/D的细项差异")


if __name__ == "__main__":
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
    os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7897"

    print("多语言 Embedding 对比实验")
    print(f"文档数: {len(ALL_DOCS)} ({len(ZH_DOCS)} 中文 + {len(EN_DOCS)} 英文)")
    print(f"查询数: {len(QUERIES)} 条（A/B/C/D 四类场景各 2 条）")

    encode_zh = load_bge_zh()
    encode_m3 = load_bge_m3()

    r_zh = evaluate_model("bge-large-zh-v1.5", encode_zh, ALL_DOCS, QUERIES)
    r_m3 = evaluate_model("bge-m3",            encode_m3, ALL_DOCS, QUERIES)

    print_comparison(r_zh, r_m3)
