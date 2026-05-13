"""
Redis 缓存三大问题 —— LLM 应用场景单机模拟

前置：
    sudo service redis-server start
    pip install redis

运行：
    python scripts/redis_cache_demo.py
"""

import redis
import time
import threading
import random

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
r.flushdb()  # 清空测试数据

# 模拟昂贵的 LLM 调用（用 sleep 替代）
llm_call_count = 0

def fake_llm_call(question: str) -> str:
    global llm_call_count
    llm_call_count += 1
    time.sleep(0.5)  # 模拟 LLM 耗时
    return f"[LLM答案] 关于'{question}'的回答（第{llm_call_count}次调用）"


# ─────────────────────────────────────────────────
# 场景一：缓存穿透
# 用户反复查询知识库里不存在的内容
# ─────────────────────────────────────────────────
def demo_penetration():
    print("\n" + "="*55)
    print("【缓存穿透】用户反复查询不存在的内容")
    print("="*55)
    global llm_call_count
    llm_call_count = 0

    def query_without_protection(question):
        cached = r.get(f"qa:{question}")
        if cached:
            return f"命中缓存: {cached}"
        # 知识库里根本没有这个内容，但还是触发了 LLM
        answer = fake_llm_call(question)
        # 注意：因为"没有答案"，没有写入缓存，下次还会再调用 LLM
        return answer

    fake_question = "外星人存在吗"  # 知识库里不存在
    for i in range(5):
        result = query_without_protection(fake_question)
        print(f"  第{i+1}次查询: {result[:40]}")

    print(f"\n  ❌ LLM 被调用了 {llm_call_count} 次（应该只调用1次或0次）")

    # ── 解决方案：空值缓存 ──
    print("\n  ✅ 修复方案：对不存在的内容缓存空值（TTL 短一点）")
    llm_call_count = 0

    def query_with_null_cache(question):
        cache_key = f"qa:{question}"
        cached = r.get(cache_key)
        if cached is not None:
            return "命中缓存（空值）" if cached == "__NULL__" else f"命中缓存: {cached}"
        # 判断知识库是否存在（这里用简单判断模拟）
        knowledge_exists = "加氢裂化" in question
        if not knowledge_exists:
            r.setex(cache_key, 30, "__NULL__")  # 缓存空值，TTL 30s
            return "知识库无此内容（已缓存空值）"
        answer = fake_llm_call(question)
        r.setex(cache_key, 300, answer)
        return answer

    for i in range(5):
        result = query_with_null_cache(fake_question)
        print(f"  第{i+1}次查询: {result}")
    print(f"  LLM 被调用了 {llm_call_count} 次（空值缓存后只调用0次）")


# ─────────────────────────────────────────────────
# 场景二：缓存击穿
# 热门问题的缓存恰好过期，50个并发请求同时打进来
# ─────────────────────────────────────────────────
def demo_breakdown():
    print("\n" + "="*55)
    print("【缓存击穿】热门问题缓存过期，并发请求同时涌入")
    print("="*55)
    global llm_call_count
    llm_call_count = 0

    hot_question = "加氢裂化装置的操作压力是多少"
    cache_key = f"qa:{hot_question}"
    r.delete(cache_key)  # 模拟缓存刚好过期

    results = []

    def query_without_lock():
        cached = r.get(cache_key)
        if cached:
            results.append("cache_hit")
            return
        # 缓存不存在，直接调用 LLM
        answer = fake_llm_call(hot_question)
        r.setex(cache_key, 300, answer)
        results.append("llm_called")

    # 50个并发请求同时到达
    threads = [threading.Thread(target=query_without_lock) for _ in range(50)]
    for t in threads: t.start()
    for t in threads: t.join()

    print(f"  ❌ 50个并发请求，LLM 被调用了 {llm_call_count} 次")

    # ── 解决方案：互斥锁 ──
    print("\n  ✅ 修复方案：SETNX 互斥锁，只有一个请求重建缓存")
    llm_call_count = 0
    r.delete(cache_key)
    results.clear()

    def query_with_lock():
        cached = r.get(cache_key)
        if cached:
            results.append("cache_hit")
            return
        # 尝试抢锁（NX=不存在才设置，EX=10秒超时防死锁）
        lock_key = f"lock:{cache_key}"
        got_lock = r.set(lock_key, "1", nx=True, ex=10)
        if got_lock:
            try:
                answer = fake_llm_call(hot_question)
                r.setex(cache_key, 300, answer)
                results.append("llm_called")
            finally:
                r.delete(lock_key)
        else:
            # 没抢到锁，等一下再读缓存
            time.sleep(0.6)
            cached = r.get(cache_key)
            results.append("cache_hit_after_wait" if cached else "miss")

    threads = [threading.Thread(target=query_with_lock) for _ in range(50)]
    for t in threads: t.start()
    for t in threads: t.join()

    print(f"  50个并发请求，LLM 被调用了 {llm_call_count} 次（只应调用1次）")


# ─────────────────────────────────────────────────
# 场景三：缓存雪崩
# 批量缓存的 LLM 答案设置了相同 TTL，同时过期
# ─────────────────────────────────────────────────
def demo_avalanche():
    print("\n" + "="*55)
    print("【缓存雪崩】批量缓存同时过期，瞬间大量 LLM 调用")
    print("="*55)
    global llm_call_count

    questions = [f"问题_{i}" for i in range(20)]

    # ❌ 错误做法：所有 key 用同一个 TTL
    print("\n  ❌ 所有缓存 TTL 相同（3秒后同时过期）:")
    for q in questions:
        r.setex(f"qa_bad:{q}", 3, f"答案_{q}")  # 全部 3 秒后过期

    time.sleep(3.1)  # 等待全部过期
    llm_call_count = 0
    for q in questions:
        if not r.get(f"qa_bad:{q}"):
            fake_llm_call(q)  # 模拟重建
    print(f"  3秒后，{llm_call_count} 个 LLM 调用同时触发")

    # ✅ 正确做法：TTL 加随机抖动
    print("\n  ✅ TTL 加随机抖动（基础300秒 ± 60秒随机）:")
    for q in questions:
        ttl = 5 + random.randint(-2, 2)  # 演示用短TTL
        r.setex(f"qa_good:{q}", ttl, f"答案_{q}")
        print(f"    {q}: TTL={ttl}s", end="  ")
    print("\n  过期时间被分散，不会同时涌入")


# ─────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        r.ping()
    except Exception:
        print("❌ Redis 未启动，请先运行: sudo service redis-server start")
        exit(1)

    demo_penetration()
    demo_breakdown()
    demo_avalanche()

    print("\n" + "="*55)
    print("演示完成")
