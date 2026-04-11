"""
知识库管理路由

GET    /api/collections          列出所有知识库及其文档数
DELETE /api/collections/{name}   删除指定知识库
"""

import sys
sys.path.insert(0, "/home/hanyuu/rag_project")

from fastapi import APIRouter, HTTPException

router = APIRouter()


def _get_chroma_client():
    import chromadb
    from config import CHROMA_PERSIST_DIR
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


@router.get("/collections")
async def list_collections():
    """列出所有 Chroma collection（知识库）"""
    try:
        from chains.rag_chain import _load_registry
        registry = _load_registry()

        client = _get_chroma_client()
        cols = client.list_collections()
        result = []
        for col in cols:
            c = client.get_collection(col.name)
            model = registry.get(col.name, "BAAI/bge-large-zh-v1.5")
            result.append({
                "name": col.name,
                "count": c.count(),
                "embedding_model": model,
            })
        return {"collections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """删除指定知识库"""
    try:
        client = _get_chroma_client()
        client.delete_collection(name)
        return {"message": f"知识库 '{name}' 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
