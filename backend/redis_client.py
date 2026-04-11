"""
Redis 客户端单例 + 任务状态封装

任务 key 格式：task:{task_id}  (Redis Hash)
TTL：24小时（任务完成后自动过期，不需要手动清理）
"""

import json
import redis

# 单例连接（lazy init）
_client: redis.Redis | None = None

TASK_TTL = 86400  # 24小时


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    return _client


# ── 任务状态操作 ──────────────────────────────────────────────────────────────

def task_set(task_id: str, data: dict) -> None:
    """创建或覆盖一个任务（hset + expire）"""
    r = get_redis()
    # Redis hash 不支持嵌套，result/error 序列化为 JSON 字符串
    flat = {}
    for k, v in data.items():
        flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else (v if v is not None else "")
    r.hset(f"task:{task_id}", mapping=flat)
    r.expire(f"task:{task_id}", TASK_TTL)


def task_update(task_id: str, **kwargs) -> None:
    """更新任务的部分字段"""
    r = get_redis()
    flat = {}
    for k, v in kwargs.items():
        flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else (v if v is not None else "")
    if flat:
        r.hset(f"task:{task_id}", mapping=flat)


def task_get(task_id: str) -> dict | None:
    """读取任务状态，不存在返回 None"""
    r = get_redis()
    data = r.hgetall(f"task:{task_id}")
    if not data:
        return None
    # 反序列化 JSON 字段
    result = {}
    for k, v in data.items():
        try:
            parsed = json.loads(v)
            result[k] = parsed
        except (json.JSONDecodeError, TypeError):
            result[k] = v if v != "" else None
    # percent 转回 int
    if "percent" in result and result["percent"] is not None:
        try:
            result["percent"] = int(result["percent"])
        except (ValueError, TypeError):
            pass
    return result
