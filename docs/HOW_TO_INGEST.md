# 文档入库操作指南

> 适用于向本 RAG 项目添加新文档（PDF / Word / Excel）
> 更新日期：2026-04-09

---

## 一、整体流程

```
新文档（PDF/Word/Excel）
    │
    ▼
[MinIO 存储]           原始文件上传至对象存储，永久保留，失败可重处理
    │
    ▼
[MinerU 解析]          PDF → 结构化 block（text/table/equation），携带页码/类型元数据
    │
    ▼
[VLM 补表（可选）]     对 MinerU 未能 OCR 的表格图片，用 qwen-vl-plus 识别补全
    │                  补全结果同步上传 MinIO（content_list_with_tables.json）
    ▼
[文本切分]             text 按 Markdown 标题两阶段切，table/equation 整块保留
    │
    ▼
[三写入库]             ① 向量写入 Chroma  ② 文本写入 Elasticsearch  ③ 元数据写入 MySQL
    │
    ▼
[Redis 任务状态更新]    percent=100，status=done，前端进度条完成
```

**前提条件（启动 start.sh 会自动检查）**：
```bash
conda activate mineru_2.5

# 确认各服务已运行
redis-cli ping          # → PONG
mysql -u root -e "SELECT 1"   # → 1
curl localhost:9000/minio/health/live   # → 200
curl localhost:9200/_cluster/health     # → green/yellow
```

---

## 二、方式一：通过前端上传（推荐日常使用）

启动项目后在浏览器操作：
1. 访问 `http://localhost:3000` → 知识库页面
2. 选择或新建知识库名（命名规范见第五节）
3. 拖拽上传文件，等待进度条完成

**适合**：单个或少量文件，有实时进度反馈。

---

## 三、方式二：批量入库脚本（推荐大批量）

### 3.1 标准批量入库（energy 系列）

```bash
conda activate mineru_2.5
cd ~/rag_project

# 预览（不实际入库）
python scripts/batch_ingest.py --dry-run

# 入库中文库
python scripts/batch_ingest.py --dir chinese --collection energy_zh

# 入库英文库
python scripts/batch_ingest.py --dir english --collection energy_en

# 入库全部
python scripts/batch_ingest.py
```

**文件放置路径**：
```
data/raw/chinese/    ← 中文 PDF 放这里
data/raw/english/    ← 英文 PDF 放这里
```

**特性**：断点续传（已处理的文件自动跳过），每个文件独立子进程处理。

### 3.2 自定义库名批量入库

如果要入库到其他库（如 GB 国标库），修改 `scripts/batch_ingest.py` 的 `TASKS` 配置，或直接用 Python 调用：

```python
from backend.services.pipeline import process_file

result = process_file(
    file_path="data/raw/GB+23971-2025.pdf",
    collection_name="gb_standards",   # 自定义库名
    use_vlm_ocr=True,                 # 开启 VLM 补表
    embedding_model="BAAI/bge-m3",    # 跨语言用 bge-m3，纯中文可用 bge-large-zh-v1.5
)
print(f"入库完成，共 {result['chunks']} 个 chunk")
```

---

## 四、方式三：手动跑 MinerU（用于排查解析质量）

当你需要**先看解析结果再决定是否入库**时（比如验证横向表格识别情况），可以单独跑 MinerU：

```bash
conda activate mineru_2.5
cd ~/rag_project

# 单个文件解析，输出到 data/mineru_output/
python -c "
import sys
sys.path.insert(0, '/home/hanyuu/MinerU-master')
from demo.demo import do_parse, read_fn
from pathlib import Path

pdf = Path('data/raw/GB+23971-2025.pdf')
do_parse(
    output_dir='data/mineru_output',
    pdf_file_names=[pdf.stem],
    pdf_bytes_list=[read_fn(pdf)],
    p_lang_list=['ch'],
    backend='pipeline',
    method='auto',
)
print('解析完成')
"
```

解析完成后，输出在：
```
data/mineru_output/
└── GB+23971-2025/
    └── auto/
        ├── GB+23971-2025_content_list.json   ← 结构化 block 数据
        ├── GB+23971-2025.md                  ← 可读 Markdown
        ├── GB+23971-2025_layout.pdf          ← 带布局标注的 PDF（排查用）
        └── images/                           ← 提取的图片（含表格截图）
```

### 4.1 查看解析质量

```bash
conda activate mineru_2.5
python -c "
import json
path = 'data/mineru_output/GB+23971-2025/auto/GB+23971-2025_content_list.json'
with open(path) as f: blocks = json.load(f)

total    = len(blocks)
tables   = [b for b in blocks if b.get('type') == 'table']
t_empty  = [b for b in tables if not b.get('text','').strip()]
t_no_img = [b for b in t_empty if not b.get('img_path','').strip()]

print(f'总 block 数: {total}')
print(f'表格数: {len(tables)}')
print(f'表格 text 为空: {len(t_empty)}')
print(f'其中连图片也没有: {len(t_no_img)}（无法 VLM 处理）')
print()
# 打印空表格所在页码
for b in t_empty:
    has_img = '有图片' if b.get('img_path') else '无图片❌'
    print(f'  p.{b.get(\"page_idx\",0)+1}  {has_img}')
"
```

### 4.2 VLM 补充空表格（单独跑）

```bash
conda activate mineru_2.5
python -c "
import sys
sys.path.insert(0, '/home/hanyuu/rag_project')
from tools.ocr_empty_tables import run_ocr
from pathlib import Path

base = Path('data/mineru_output/GB+23971-2025/auto')
run_ocr(
    base / 'GB+23971-2025_content_list.json',
    base / 'content_list_with_tables.json'
)
"
```

结果保存为 `content_list_with_tables.json`，后续入库自动优先使用该文件。

---

## 五、知识库命名规范

跨语言检索依赖命名约定，**配对库前缀必须相同**：

| 中文库名 | 对应英文库名 | 跨语言检索 |
|---------|------------|-----------|
| `energy_zh` | `energy_en` | ✅ 自动配对 |
| `gb_standards_zh` | `gb_standards_en` | ✅ 自动配对 |
| `hydro_manual` | `hydro_manual_en` | ✅ 自动配对 |
| `my_docs` | `my_docs_en` | ✅ 自动配对 |

**Embedding 模型选择**：
- 纯中文库：`BAAI/bge-large-zh-v1.5`（效果更好）
- 中英混合或有配对英文库：`BAAI/bge-m3`（必须，向量空间才对齐）
- **同一组配对库必须用相同模型**

模型选择记录在 `data/chroma_db/embedding_registry.json`，自动维护，无需手动修改。

---

## 六、测试集生成与评估

### 6.1 生成 QA 测试集

```bash
conda activate mineru_2.5
cd ~/rag_project

# GB 标准库（支持动态题数，按文档长度分配）
python scripts/generate_qa_v2.py \
    --input_dir data/raw/gb_standards/mineru_output \
    --output    data/qa_dataset/gb_qa_1000.json \
    --target    1000

# 加氢手册（单文档多段采样，目标 200 题）
python scripts/generate_qa_hydro.py --target 200
```

### 6.2 标注 ground truth chunk_id

```bash
python scripts/annotate_chunk_ids.py \
    --input      data/qa_dataset/gb_qa_1000.json \
    --output     data/qa_dataset/gb_qa_annotated_512.json \
    --collection gb_standards_512
```

### 6.3 Precision@K 评估

```bash
python scripts/eval_precision_at_k.py \
    --input      data/qa_dataset/gb_qa_annotated_512.json \
    --collection gb_standards_512 \
    --top_k      5
```

当前结果（41 文档，443 有效题）：

| 实验 | Hit@1 | Hit@3 | Hit@5 | MRR |
|------|-------|-------|-------|-----|
| A1（512/64）  | 0.242 | 0.424 | 0.496 | 0.355 |
| A2（1024/100）| 0.248 | 0.433 | 0.492 | 0.355 |

注：数值受自动标注噪声低估约 15%，人工抽检（spot_check.py）修正后 Hit@5 估算 ≥ 0.60。

### 6.4 RAGAS 端到端评估

```bash
python scripts/eval_ragas_gb.py \
    --collection gb_standards_512 \
    --input      data/qa_dataset/gb_qa_annotated_512.json \
    --limit      50   # 先跑 50 题验证，去掉则全量
```

### 6.5 RAPTOR 摘要节点追加（B1/B2 实验）

```bash
# 追加摘要节点到已有 collection（不需要重新入库）
python splitters/raptor_builder.py --collection gb_standards_512   # → B1
python splitters/raptor_builder.py --collection gb_standards_1024  # → B2
# 完成后用 eval_precision_at_k.py 重跑评估即可
```

---

## 七、当前知识库状态

```bash
# 查看所有库及 chunk 数（Chroma）
conda activate mineru_2.5
python -c "
import chromadb, os
client = chromadb.PersistentClient(path=os.path.expanduser('~/rag_project/data/chroma_db'))
for c in client.list_collections():
    print(f'{c.name}: {c.count()} chunks')
"

# 查看某个库包含哪些文件（MySQL documents 表）
python -c "
import sys; sys.path.insert(0, '/home/hanyuu/rag_project')
from backend.db import list_documents
for doc in list_documents('energy_zh'):
    print(f\"{doc['filename']}  {doc['chunks']}chunks  {doc['ingested_at']}\")
"

# 查看 ES 索引状态
curl -s localhost:9200/_cat/indices?v | grep -E "energy|hydro|investment"
```

当前状态（2026-04-09）：

| 库名 | Chunks | 来源 | Embedding | 说明 |
|------|--------|------|-----------|------|
| `hydro_manual` | 1,632 | 240万吨加氢裂化工艺规程 | bge-large-zh-v1.5 | 含 VLM OCR 补表 |
| `investment_db` | 7,440 | 投资机构&项目数据库 xlsx | bge-large-zh-v1.5 | — |
| `energy_zh` | 12,508 | CNPC/NEA 等中文能源报告（12份）| bge-m3 | — |
| `energy_en` | 5,961 | BP/EIA/Shell 等英文能源报告（17份）| bge-m3 | — |
| `gb_standards_512` | ~6,000 | GB 国标 41 份 | bge-large-zh-v1.5 | chunk 512/64，含 chunk_id 元数据，A1 实验库 |
| `gb_standards_1024` | ~3,200 | GB 国标 41 份 | bge-large-zh-v1.5 | chunk 1024/100，含 chunk_id 元数据，A2 实验库 |

---

## 八、常见问题

**Q: MinerU 解析时卡住不动？**
先检查 GPU 显存：
```bash
nvidia-smi
# 如有僵尸进程：kill -9 <PID>
```

**Q: 提示 ModuleNotFoundError？**
确认已激活正确环境：终端左侧应显示 `(mineru_2.5)`

**Q: 表格内容识别为乱码/空白？**
说明是横向/复杂表格，MinerU 未能 OCR。跑 VLM 补表步骤（第4.2节），
qwen-vl-plus 看图片识别，不受方向影响。

**Q: 新入库的文档前端没显示？**
刷新知识库列表页，或重启后端（图缓存问题）：
```bash
bash ~/rag_project/start.sh
```

**Q: 后端重启后任务进度不见了？**
任务状态存在 Redis（TTL 24h），重启后应仍可查到。如果 Redis 没启动：
```bash
sudo service redis-server start
redis-cli ping   # 应返回 PONG
```

**Q: ES 索引和 Chroma 数据不一致怎么办（某库 ES 有但 Chroma 没有，或反之）？**
重新入库该知识库（force_rebuild=True），pipeline 会同时写两个：
```python
from backend.services.pipeline import process_file
process_file("data/raw/xxx.pdf", "collection_name", force_rebuild=True)
```

**Q: 想从 MinIO 重新处理某个文件怎么做？**
```python
from backend.storage import download_file
from backend.services.pipeline import process_file

# 1. 从 MinIO 下载到临时路径
download_file("collection_name/task_id/filename.pdf", "/tmp/reprocess.pdf")
# 2. 重新入库
process_file("/tmp/reprocess.pdf", "collection_name", force_rebuild=True)
```

---

## 九、预留扩展点

以下功能计划中，入库流程会相应更新：

- [ ] **Level 7c GraphRAG**：入库时同步抽取实体关系，写入图数据库
- [x] **RAPTOR**：`splitters/raptor_builder.py` 已实现，追加摘要节点无需重建库
- [x] **切分优化**：chunk_size 已升级到 1024/100，chunk_id 元数据已注入
- [x] **GB 测试集**：`gb_qa_annotated_512/1024.json` 已建成，Precision@K 流水线跑通
- [ ] **切分优化 V2**：overlap 句子边界对齐，列表前导句合并
- [ ] **ES 集成**：BM25 持久化替代内存 rank_bm25，消除每次重建开销
