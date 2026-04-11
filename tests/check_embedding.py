import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import threading
import itertools
sys.path.insert(0, "/home/hanyuu/rag_project")


def spinner(msg, stop_event):
    for ch in itertools.cycle("|/-\\"):
        if stop_event.is_set():
            break
        print(f"\r{msg} {ch}", end="", flush=True)
        time.sleep(0.1)
    print(f"\r{msg} 完成", flush=True)


print("加载 embedding 模型（bge-large-zh-v1.5）...")
stop = threading.Event()
t = threading.Thread(target=spinner, args=("  模型加载中...", stop))
t.start()

t0 = time.time()
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
finally:
    stop.set()
    t.join()
print(f"  device: {emb._client.device}  加载耗时: {time.time()-t0:.1f}s")

print("测试 embedding 速度（10条）...")
texts = ["这是一段测试文本，用于验证 embedding 速度。"] * 10
t0 = time.time()
emb.embed_documents(texts)
print(f"  10条耗时: {time.time()-t0:.2f}s")
