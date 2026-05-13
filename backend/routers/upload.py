"""
文件上传路由

POST /api/upload          上传文件，返回 task_id，后台异步处理
GET  /api/tasks/{task_id} 查询处理进度（任务状态持久化在 Redis，重启不丢）
"""

import uuid
import shutil
import re
import threading
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.redis_client import task_set, task_update, task_get

router = APIRouter()

# 临时上传目录
UPLOAD_DIR = Path("/home/hanyuu/rag_project/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 300 * 1024 * 1024  # 300MB


def _run_pipeline(task_id: str, file_path: str, collection_name: str,
                  use_vlm_ocr: bool, embedding_model: str = None, use_parent_child: bool = False):
    """在后台线程中运行处理流水线，进度写入 Redis"""
    task_update(task_id, status="processing", percent=0, step="初始化...")

    def on_progress(step: str, percent: int):
        task_update(task_id, step=step, percent=percent)

    try:
        from backend.services.pipeline import process_file
        result = process_file(
            file_path=file_path,
            collection_name=collection_name,
            use_vlm_ocr=use_vlm_ocr,
            progress_callback=on_progress,
            embedding_model=embedding_model,
            use_parent_child=use_parent_child,
        )
        task_update(task_id, status="done", percent=100, step="处理完成", result=result)
    except Exception as e:
        task_update(task_id, status="error", step="处理失败", error=str(e))
    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    use_vlm_ocr: bool = Form(False),
    embedding_model: str = Form("BAAI/bge-large-zh-v1.5"),
    use_parent_child: bool = Form(False),
):
    """
    上传文件并触发后台处理。

    Returns:
        {"task_id": str}  — 用于轮询进度
    """
    # 格式校验
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".doc", ".docx", ".xlsx", ".xls"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式：{suffix}，支持 PDF / Word (.doc/.docx) / Excel (.xlsx/.xls)"
        )

    # 知识库名校验
    if not re.match(r'^[\w\-\u4e00-\u9fff]+$', collection_name):
        raise HTTPException(status_code=400, detail="知识库名称只能包含字母、数字、下划线、连字符或中文")

    # 保存文件
    task_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{task_id}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 大小 & 空文件校验
    file_size = save_path.stat().st_size
    if file_size == 0:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传文件为空")
    if file_size > MAX_FILE_SIZE:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"文件过大：{file_size // 1024 // 1024}MB，最大支持 300MB"
        )

    # 写入 Redis（24h TTL，重启后仍可查询）
    task_set(task_id, {
        "status": "pending",
        "step": "等待处理...",
        "percent": 0,
        "filename": file.filename,
        "collection": collection_name,
        "error": None,
        "result": None,
    })

    # 启动后台线程
    threading.Thread(
        target=_run_pipeline,
        args=(task_id, str(save_path), collection_name, use_vlm_ocr, embedding_model, use_parent_child),
        daemon=True,
    ).start()

    return {"task_id": task_id}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """查询任务处理进度（从 Redis 读取，重启后依然可用）"""
    task = task_get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在或已过期（24h）")
    return task
