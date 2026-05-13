"""查看 hydro_crack collection 的元数据结构"""
import sys
sys.path.insert(0, "/home/hanyuu/rag_project")
import chromadb
from config import CHROMA_PERSIST_DIR

client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
col = client.get_collection("hydro_crack")
print(f"总 chunks：{col.count()}\n")

sample = col.get(limit=3, include=["metadatas", "documents"])
for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
    print(f"--- chunk {i} ---")
    print(f"元数据字段：{list(meta.keys())}")
    print(f"chunk_id  ：{meta.get('chunk_id')}")
    print(f"page      ：{meta.get('page')}")
    print(f"文本前60字：{doc[:60]}")
    print()
