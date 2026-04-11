#!/bin/bash
# ─────────────────────────────────────────────
#  RAG Project 一键启动脚本
#  用法：bash ~/rag_project/start.sh
# ─────────────────────────────────────────────

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONDA_ENV="mineru_2.5"
BACKEND_PORT=8000
FRONTEND_DIR="$PROJECT_DIR/frontend"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== RAG Project 启动 ===${NC}"

# ── 1. 检查端口占用 ──────────────────────────
for PORT in $BACKEND_PORT 3000; do
    PID=$(lsof -ti tcp:$PORT 2>/dev/null)
    if [ -n "$PID" ]; then
        echo -e "${YELLOW}[警告] 端口 $PORT 已被占用（PID $PID），正在释放...${NC}"
        kill -9 $PID 2>/dev/null
        sleep 1
    fi
done

# ── 1.5 启动 Redis（WSL 不自动启动） ────────
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}[✓] Redis 已运行${NC}"
else
    sudo service redis-server start > /dev/null 2>&1
    if redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}[✓] Redis 已启动${NC}"
    else
        echo -e "${YELLOW}[警告] Redis 启动失败，任务状态将使用内存存储（重启后丢失）${NC}"
    fi
fi

# ── 加载环境变量 ─────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# ── 2. 激活 conda 环境 ───────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
if [ $? -ne 0 ]; then
    echo -e "${RED}[错误] conda activate $CONDA_ENV 失败${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] conda 环境：$CONDA_ENV${NC}"

# ── 3. 启动后端（后台） ──────────────────────
cd "$PROJECT_DIR"
uvicorn backend.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload \
    > /tmp/rag_backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}[✓] 后端启动中（PID $BACKEND_PID）→ http://localhost:$BACKEND_PORT${NC}"
echo "    日志：tail -f /tmp/rag_backend.log"

# ── 4. 等后端就绪 ────────────────────────────
echo -n "    等待后端就绪"
for i in $(seq 1 30); do
    sleep 1
    if curl -s "http://localhost:$BACKEND_PORT/api/health" > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    if [ $i -eq 30 ]; then
        echo -e " ${RED}超时，请检查 /tmp/rag_backend.log${NC}"
    fi
done

# ── 5. 启动前端（前台，占住终端） ────────────
cd "$FRONTEND_DIR"
echo -e "${GREEN}[✓] 前端启动 → http://localhost:3000${NC}"
echo -e "${YELLOW}    Ctrl+C 可同时停止前后端${NC}\n"

# 捕获 Ctrl+C，退出时一并杀掉后端
trap "echo -e '\n${YELLOW}正在停止...${NC}'; kill $BACKEND_PID 2>/dev/null; exit 0" INT

npm run dev
