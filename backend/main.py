"""
Level 5 — FastAPI 后端入口

启动方式：
    conda activate mineru_2.5
    cd ~/rag_project
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import upload, chat, collections


async def _warmup():
    """
    后台预热：在线程池中异步加载所有 collection 的 RAG graph。
    使用 run_in_executor 避免阻塞事件循环，服务启动后即可响应请求。
    知识库按字母顺序依次加载，Reranker 单例只初始化一次。
    """
    import asyncio
    import chromadb
    from config import CHROMA_PERSIST_DIR
    from backend.routers.chat import _build_graph, _graph_cache

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        col_names = [c.name for c in client.list_collections()]
        if not col_names:
            print("[预热] 暂无知识库，跳过预热")
            return

        print(f"[预热] 开始预热 {len(col_names)} 个知识库：{col_names}")
        loop = asyncio.get_event_loop()

        for name in col_names:
            key = (name, 10, 3, 0.1)
            if key not in _graph_cache:
                print(f"[预热] 加载 {name} ...")
                # run_in_executor：在线程池执行，不阻塞事件循环
                # 服务在预热期间仍可正常响应 /api/collections 等轻量请求
                _graph_cache[key] = await loop.run_in_executor(
                    None, _build_graph, name, 10, 3, 0.1
                )
                print(f"[预热] {name} 完成 ✓")

        print("[预热] 全部完成，冷启动已消除")
    except Exception as e:
        print(f"[预热] 失败（不影响服务）：{e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动后在后台异步预热，不阻塞服务就绪
    asyncio.create_task(_warmup())
    yield


app = FastAPI(title="RAG 知识库问答系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(collections.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
