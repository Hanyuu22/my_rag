"""
消融实验入库脚本 — chunk_size / embedding 模型对比

用法：
    # chunk_size 消融（固定 bge-large-zh-v1.5）
    python scripts/ingest_ablation.py --collection hydro_crack_256  --chunk_size 256  --overlap 32
    python scripts/ingest_ablation.py --collection hydro_crack      --chunk_size 512  --overlap 64
    python scripts/ingest_ablation.py --collection hydro_crack_1024 --chunk_size 1024 --overlap 128

    # embedding 模型消融（固定最优 chunk_size）
    python scripts/ingest_ablation.py --collection hydro_crack_bge_m3    --chunk_size 512 --overlap 64 --model BAAI/bge-m3
    python scripts/ingest_ablation.py --collection hydro_crack_bge_small --chunk_size 512 --overlap 64 --model BAAI/bge-small-zh-v1.5
    python scripts/ingest_ablation.py --collection hydro_crack_m3e       --chunk_size 512 --overlap 64 --model moka-ai/m3e-base
    python scripts/ingest_ablation.py --collection hydro_crack_text2vec  --chunk_size 512 --overlap 64 --model shibing624/text2vec-base-chinese
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")

CONTENT_LIST = Path(
    "/home/hanyuu/rag_project/data/mineru_output"
    "/【保密】240万吨加氢裂化装置工艺技术规程（试行）-部分章节"
    "/auto/content_list_with_tables_patched.json"
)
DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--overlap",    type=int, required=True)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--force",      action="store_true", help="已存在也强制重建")
    args = parser.parse_args()

    import chromadb
    from config import CHROMA_PERSIST_DIR
    from chains.rag_chain import get_embeddings, save_collection_embedding_model

    # ── 检查是否已存在 ──────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    existing = [c.name for c in client.list_collections()]
    if args.collection in existing:
        col = client.get_collection(args.collection)
        if col.count() > 0 and not args.force:
            print(f"⚠️  {args.collection} 已存在 ({col.count()} chunks)，跳过。加 --force 强制重建。")
            return
        print(f"删除旧 collection: {args.collection}")
        client.delete_collection(args.collection)

    # ── 加载文档 ─────────────────────────────────────────────────────────────
    print(f"\n加载 content_list: {CONTENT_LIST.name}")
    from loaders.mineru_loader import MinerULoader
    docs = list(MinerULoader(str(CONTENT_LIST), source_name="hydro_reg").load())
    print(f"  原始 blocks: {len(docs)}")

    # ── 切分（动态 patch config，避免影响全局） ─────────────────────────────
    import config as _cfg
    _orig_size, _orig_overlap = _cfg.CHUNK_SIZE, _cfg.CHUNK_OVERLAP
    _cfg.CHUNK_SIZE    = args.chunk_size
    _cfg.CHUNK_OVERLAP = args.overlap

    # 重新 import splitter（已被 import 过的模块需要用 importlib 刷新）
    import importlib
    import splitters.markdown_splitter as _spl
    importlib.reload(_spl)
    chunks = _spl.split_documents(docs)

    _cfg.CHUNK_SIZE    = _orig_size
    _cfg.CHUNK_OVERLAP = _orig_overlap

    print(f"  chunk_size={args.chunk_size}/overlap={args.overlap} → {len(chunks)} chunks")

    # ── 向量化入库 ───────────────────────────────────────────────────────────
    print(f"\n加载 Embedding 模型: {args.model}")
    t0 = time.time()
    embeddings = get_embeddings(args.model)
    print(f"  模型加载耗时: {time.time()-t0:.1f}s")

    from langchain_chroma import Chroma
    print(f"\n向量化并写入 Chroma collection: {args.collection}")
    t1 = time.time()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=args.collection,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )
    print(f"  入库耗时: {time.time()-t1:.1f}s，共 {len(chunks)} chunks")

    # ── 注册 embedding 模型 ──────────────────────────────────────────────────
    save_collection_embedding_model(args.collection, args.model)
    print(f"  已注册 embedding 模型: {args.model}")
    print(f"\n✅ 完成：{args.collection}")


if __name__ == "__main__":
    main()
