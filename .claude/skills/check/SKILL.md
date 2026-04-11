---
name: check
description: 快速检查 rag_project 后端+前端代码健康状态
allowed-tools: Bash, Read, Grep
---

# rag_project 健康检查

请依次完成以下检查并输出一份简洁报告：

## 1. Python 语法检查
```bash
find ~/rag_project/backend ~/rag_project/graphs ~/rag_project/chains ~/rag_project/retrievers -name "*.py" | head -30 | xargs python3 -m py_compile 2>&1
```
输出：有无语法错误

## 2. 关键导入检查
检查以下文件顶部的 import 是否有明显问题：
- graphs/rag_graph.py
- backend/routers/chat.py
- chains/rag_chain.py

## 3. SSE 事件类型一致性
检查 backend/routers/chat.py 里发出的事件类型，与 frontend/src/components/ChatWindow.tsx 里处理的事件类型是否匹配。

## 4. console.log 残留
```bash
grep -r "console\.log" ~/rag_project/frontend/src --include="*.tsx" --include="*.ts" -n
```
输出：有无未清理的调试日志

## 5. TODO 统计
```bash
grep -r "TODO\|FIXME\|HACK" ~/rag_project/backend ~/rag_project/graphs --include="*.py" -n | head -20
```

## 输出格式
最后用一个表格汇总：

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Python 语法 | ✅/❌ | ... |
| 关键导入 | ✅/❌ | ... |
| SSE 事件一致性 | ✅/❌ | ... |
| console.log 残留 | ✅/❌ | N 处 |
| TODO 统计 | ℹ️ | N 处 |
