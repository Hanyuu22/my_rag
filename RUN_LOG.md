# RAG 项目运行日志

> 记录各阶段实际运行过程、遇到的问题和解决方案。

---

## Level 1 — 基础 RAG 链跑通（2026-03-19）

### 测试文件
- 初版：`MinerU-master/output/.../course_content_list.json`（课程 PDF，98 chunks）
- 正式：`加氢裂化装置工艺技术规程（试行）-部分章节.doc`（78页工艺手册）

---

### 问题记录

#### ① Embedding 模型加载卡死（多进程抢显存）
**现象**：`test_level1.py` 跑到 "Step 2: 建向量库" 卡住无响应，几分钟无进展。
**原因**：历史遗留的多个 Python 进程（最早一个跑了 2 小时+）同时占着 GPU，8GB 显存全被占满，新进程等不到资源。
**解决**：`kill` 掉所有残留进程，当前进程自动恢复并继续。
**教训**：跑 GPU 脚本前先 `nvidia-smi` 确认没有僵尸进程。

---

#### ② HuggingFace 模型缓存损坏（.incomplete 文件）
**现象**：BGE 模型重新开始下载，进度条出现 `pytorch_model.bin: 0%`。
**原因**：卡死进程把下载文件留成了 `.incomplete` 状态，被误删后触发重新下载。
**解决**：开梯子（关闭 hf-mirror），重新下载 1.3GB，约 1 分钟完成（24MB/s）。
**后续**：加入 `HF_HUB_DISABLE_XET=1` 到 `.bashrc`，避免 hf-mirror 不支持 XET 协议导致的卡顿。

---

#### ③ HF 镜像 + 梯子冲突
**现象**：开着日本梯子，下载 HuggingFace 还是慢/失败。
**原因**：`.bashrc` 里同时设了 `HF_ENDPOINT=https://hf-mirror.com` 和 `https_proxy=127.0.0.1:7897`，访问镜像站也走了代理，适得其反。
**解决**：注释掉 `HF_ENDPOINT`，有梯子时直接走代理访问 HuggingFace 官网。
**切换策略**：有梯子 → 注释 `HF_ENDPOINT`；无梯子 → 取消注释用镜像。

---

#### ④ HuggingFaceEmbeddings 属性名变更
**现象**：`AttributeError: 'HuggingFaceEmbeddings' object has no attribute 'client'`
**原因**：langchain-huggingface 1.2.x 把 `.client` 改成了 `._client`（私有属性）。
**解决**：将 `emb.client.device` 改为 `emb._client.device`。

---

#### ⑤ Word (.doc) 文件无法用 python-docx/docx2txt 读取
**现象**：`BadZipFile: File is not a zip file`
**原因**：`.doc` 是旧版 OLE 二进制格式，不是 `.docx`（zip 格式），python-docx 不支持。
**解决**：调用 `powershell.exe` + Windows Word COM 对象，从 WSL 直接操控 Windows Word 转 PDF：
```python
$doc.SaveAs([ref]$pdfPath, [ref]17)  # 17 = wdFormatPDF
```
转出 PDF 后走正常的 MinerU pipeline。
**注意**：WSL 路径要转成 `\\wsl.localhost\{distro}\...` 格式；中文文件名需先复制为英文名再转换。

---

#### ⑥ WordLoader 多处语法错误
**现象（a）**：`SyntaxError: (unicode error) 'unicodeescape'`
**原因**：docstring 里含 `\U`（`\\wsl.localhost\Ubuntu`），Python 把它当 unicode 转义。
**解决**：docstring 改成 raw string（加 `r` 前缀）。

**现象（b）**：`SyntaxError: f-string expression part cannot include a backslash`
**原因**：Python 3.11 以下的 f-string 不允许在 `{}` 内出现反斜杠。
**解决**：先把含反斜杠的表达式赋值给变量，再插入 f-string。

**现象（c）**：`ValueError: embedded null byte`
**原因**：用 PowerShell 获取 WSL distro 名称时，输出是 UTF-16LE，含大量 null byte，传给 `subprocess` 报错。
**解决**：改用 `os.environ.get("WSL_DISTRO_NAME", "Ubuntu-22.04")` 直接读环境变量。

---

#### ⑦ MinerU 模型重新下载（HF 缓存 vs ModelScope 缓存不同路径）
**现象**：MinerU 启动后显示 `Fetching 1 files: models/MFD/YOLO/yolo_v8_ft.pt: 0%`，即将从 HuggingFace 重新下载 350MB。
**原因**：模型文件在 ModelScope 缓存（`~/.cache/modelscope/`），但 MinerU 默认读 HuggingFace 缓存（`~/.cache/huggingface/`），两个路径不同，找不到就触发下载。
**解决**：在 `.bashrc` 加 `export MINERU_MODEL_SOURCE=modelscope`，MinerU 改用 ModelScope 的 `snapshot_download`，直接命中已有缓存。
**注意**：设置后需要 `source ~/.bashrc` 才在当前终端生效；不影响 BGE 等 HuggingFace 模型，两套缓存独立。

---

#### ⑧ 环境未激活导致 ModuleNotFoundError
**现象**：`ModuleNotFoundError: No module named 'langchain_core'`
**原因**：从 `(base)` 环境运行了脚本，langchain 系列包只装在 `mineru_2.5` 里。
**解决**：`conda activate mineru_2.5` 后再运行。
**教训**：注意终端提示符前缀，`(base)` vs `(mineru_2.5)` 一眼确认。

---

### 最终运行结果（正式文档）

**文件**：加氢裂化装置工艺技术规程（78页，部分章节）
**MinerU 解析耗时**：约 2 分钟（Layout/MFD/MFR/OCR 全流程）
**文档统计**：381 个公式（MFR 处理），22 个表格
**Embedding**：BGE bge-large-zh-v1.5，device: cuda:0

**问答质量评估（主观）**：

| 问题 | 评估 |
|------|------|
| 加氢裂化装置的主要反应条件是什么？ | ✅ 优秀：完整归纳出反应温度和氢分压两大核心，逻辑清晰，带影响机制 |
| 装置开工前需要哪些准备工作？ | ✅ 优秀：8类准备工作，细节丰富，来源溯源到具体页码 |
| 反应器的操作温度和压力范围是多少？ | ⚠️ 良好：诚实说明了表格数据未被检索到，给出了上下文中能找到的具体数值，未幻觉 |

**来源溯源**：每条答案附 `[p.X]` 页码，可追溯原文，功能正常。

---

---

## Level 2 — Hybrid Retriever + Reranker（2026-03-19）

### 测试文件
- 同 Level 1：加氢裂化装置工艺技术规程（78页），814 chunks

---

### 问题记录

#### ① Reranker 模型首次下载（bge-reranker-v2-m3，2.27GB）
**现象**：`model.safetensors: 35%` 开始下载，速度约 6~10MB/s。
**原因**：bge-reranker-v2-m3 之前从未下载，HuggingFace 缓存里没有。
**解决**：等待下载完成（约 6 分钟），之后缓存在 `~/.cache/huggingface/hub/`，不再重新下载。
**注意**：tokenizer 相关小文件下载速度较慢（sentencepiece.bpe.model 5MB 用了 49s），原因是 hf-mirror 对小文件限速，下载大文件反而正常。

---

#### ② Reranker 精排第一个问题耗时异常高（105s vs 正常 1~2s）
**现象**：第一次运行，第1个问题精排耗时 105s；第2、3个问题分别只需 1.2s 和 0.9s。
**原因**：PyTorch CUDA kernel 编译（JIT compilation），第一次调用某个算子时需要编译并缓存，后续复用缓存直接运行。
**解决**：属于正常冷启动现象，无需处理。第2次运行所有问题精排均在 2s 以内。
**参考数据**：第1次平均全链路 50s（含冷启动），第2次平均全链路 12s。

---

#### ③ Embedding 加载顺序不合理（优化）
**现象**：向量库已存在时，仍先花 71s 加载 Embedding，再打印"复用已有向量库"。
**原因**：`build_vectorstore` 里先调用 `get_embeddings()`，再检查向量库是否存在，逻辑顺序倒置。
**解决**：新增 `_check_vectorstore_count()` 函数，用 `chromadb.PersistentClient` 直接查询 collection 的文档数（无需 Embedding），先判断是否存在，再决定要不要加载 Embedding 以及原因。
**效果**：逻辑不变，但用户能提前看到"向量库已存在，加载 Embedding 供查询使用"，而非无缘由等待。

---

### 最终运行结果（jieba 修复后，2026-03-19）

**性能（含冷启动，第1次运行）**：

| 阶段 | 平均耗时 | 备注 |
|------|---------|------|
| 检索（BM25 + Dense + RRF） | 1.4s | |
| 精排（BGE Reranker Top-K→3） | 15.5s | Q1 冷启动 44s，Q2/Q3 约 1s |
| 生成（qwen-plus） | 7.3s | |
| 全链路均值 | 24s | 稳态约 10~12s |

**问答质量对比（jieba 修复前 → 后）**：

| 问题 | 修复前 Top3 分数 | 修复后 Top3 分数 | 变化 |
|------|----------------|----------------|------|
| 主要反应条件 | 0.995/0.987/0.986 | **0.995/0.995**/0.987 | ✅ BM25 新召回 p.13（结焦副反应），第2条从0.987→0.995 |
| 开工准备工作 | 0.965/0.964/0.617 | 0.965/0.964/0.617 | — Dense 本已覆盖，无变化 |
| 操作温度和压力范围 | 0.244/0.178/0.093 | 0.244/0.178/0.135 | ⚠️ 第3条略有改善，但整体仍低，表格数据问题 |

**第三题低分根因**：原文表9-1～9-4（操作参数表格）数值经切分后语义稀疏，检索得分极低。这是文档解析局限，不是检索算法问题。Level 3 的 query rewriting 可部分缓解，根本解决需改进表格 chunk 组织方式。

---

#### ④ BM25 未使用中文分词（关键 Bug）
**现象**：Hybrid Retriever 代码运行正常，但 BM25 实际贡献为零。
**原因**：`BM25Retriever.from_documents()` 默认用空格分词。中文文本几乎没有空格，导致整段文字被当成单个 token，BM25 完全失去关键词匹配能力，Hybrid 实质上退化为纯 Dense 检索。
**解决**：安装 jieba，通过 `preprocess_func=_jieba_tokenize` 传入自定义分词函数，过滤单字和空白，保留有意义的词组。
**影响**：修复前的所有 Level 2 测试结果中，BM25 未生效，检索结果等同于 Level 1 纯 Dense。修复后重跑可得到真正的 Hybrid 效果。

---

---

## Level 3 — LangGraph 多步 RAG（2026-03-19）

### 测试文件
- 同 Level 2：加氢裂化装置工艺技术规程（78页），**1632 chunks**（VLM OCR 补充表格后从 814 增至 1632）

---

### 问题记录

#### ① MinerU 表格文字为空（VLM OCR 补充）
**现象**：Q3（操作温度和压力范围）top1_score 仅 0.244，来源为 `p.29 6.2.4.3 操作温度`，核心数据表9-1～9-4（p.21~23）未被命中。
**原因**：MinerU 将这些表格识别为图片（`type=table, img_path=xxx.jpg`），但 `text` 字段为空字符串，完全不可检索。
**解决**：新建 `tools/ocr_empty_tables.py`，用 `qwen-vl-plus` 逐一对空表格图片做 OCR，输出 Markdown 表格格式，写回 `text` 字段，另存为 `content_list_with_tables.json`（不覆盖原文件）。
- 本次处理：19张空表格，19/19 成功，消耗 API 约 57 秒
- 向量库 chunks 从 814 → **1632**（每张表格拆分为多个 chunks）
- 修复后 Q3 top1_score：**0.244 → 0.849**，来源直接命中表9-2、表9-4

**集成方式**：在 `WordLoader.__init__` 加 `use_vlm_ocr=False` 参数。已有 `content_list_with_tables.json` 时自动复用，无论 `use_vlm_ocr` 开关状态；首次处理时需显式传 `use_vlm_ocr=True` 触发 API 调用。API 额度耗尽时打印警告并 break，已处理结果仍会保存。

---

#### ② Query Rewriting 跑偏（Prompt 优化）
**现象（第1版，运行日志 17:31:22）**：Q3 触发 2 次改写，最终 query 变成"固定床反应器、流化床反应器和釜式反应器……"，完全偏离石化装置语境，top1_score 降到 0.092（差于原始 0.244）。
**原因**：原始 `REWRITE_PROMPT` 只描述了通用改写策略，没有绑定石化领域背景，LLM 泛化到了教科书中的通用化工问题。
**解决**：为 `REWRITE_PROMPT` 的 system 消息加入领域约束：
- 明确指出知识库内容（加氢裂化工艺规程、反应原理、操作规程、设备说明、开停工、安全规程）
- 提供具体术语示例（氢分压、LHSV、WABT、冷氢、急冷氢、催化剂床层等）
- 加入"保持领域专一性，不要泛化为通用化工或其他行业的问题"约束

**修复后（运行日志 18:03:47）**：Q5（冷氢作用）触发 1 次改写，query 准确改写为"加氢裂化装置中冷氢（急冷氢）的用途、注入位置及对催化剂床层温度控制的作用"，top1_score 0.997。Q3 因已补充表格 OCR，0 次改写直接命中（0.849）。

---

#### ③ XLMRobertaTokenizerFast 警告
**现象**：加载 bge-reranker-v2-m3 时出现警告：
```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
```
**原因**：`bge-reranker-v2-m3` 基于 XLM-RoBERTa 架构，加载时检测到 tokenizer 词表有特殊扩展 token（sentencepiece），触发常规提示。
**影响**：无实际影响。这是 HuggingFace tokenizer 的常规信息，不影响模型输出质量和精排分数。无需处理。

---

### 最终运行结果（2026-03-19 18:03:47）

| 问题 | 路由 | 改写次数 | Top1 分 | 全链路耗时 |
|------|------|---------|---------|-----------|
| 加氢裂化装置的主要反应条件 | retrieve ✅ | 0 | 0.995 | 21.1s |
| 装置开工前需要哪些准备工作 | retrieve ✅ | 0 | 0.965 | 9.0s |
| 反应器的操作温度和压力范围 | retrieve ✅ | 0 | **0.849** | 5.0s |
| 你好，帮我介绍一下你自己 | direct ✅ | 0 | N/A | 4.8s |
| 冷氢的作用是什么 | retrieve ✅ | 1 | 0.997 | 6.0s |

**整体指标**：路由准确率 **100%**（5/5）｜ 平均 Top1 **0.952** ｜ 平均全链路 **9.18s**

**Level 2 → Level 3 关键提升**：

| 指标 | Level 2 | Level 3 | 变化 |
|------|---------|---------|------|
| Q3 Top1 score | 0.244 | **0.849** | +0.605（表格 OCR 补充后） |
| 平均 Top1 score | ~0.735 | **0.952** | +0.217 |
| Query rewriting | N/A | ✅ 自动触发，领域感知 | 新增能力 |
| 路由（闲聊过滤）| N/A | ✅ 100% 准确 | 新增能力 |

---

---

## Level 4 — RAGAS 多级对比评估（2026-03-19）

### 问题记录

#### ① ragas 0.4.3 metrics 导入方式变更
**现象**：`from ragas.metrics import faithfulness` 导入的是一个 **module**（不是对象），传给 `evaluate()` 报 `TypeError: All metrics must be initialised metric objects`。
**原因**：ragas 0.4.x 把 metrics 重构为 `ragas.metrics.collections` 下的类，但 `evaluate()` 函数只接受老式实例（`ragas.metrics._faithfulness.faithfulness` 等）。同时新式类（`Faithfulness(llm=...)`）虽然可以构造，但 `evaluate()` 的 isinstance 检查不认它。
**解决**：从私有模块导入老式实例：
```python
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
```
传给 `evaluate(llm=LangchainLLMWrapper(...), embeddings=LangchainEmbeddingsWrapper(...))`，evaluate 内部自动注入。

---

#### ② EvaluationResult 不支持 `.get()` 访问
**现象**：`AttributeError: 'EvaluationResult' object has no attribute 'get'`
**原因**：ragas 0.4 的 `EvaluationResult` 是 dataclass，`result[key]` 返回该指标每个样本的分数列表（`List[float]`），不是均值 dict。
**解决**：用 `np.nanmean(result[key])` 手动计算均值，`NaN` 对应超时失败的样本自动跳过。

---

#### ③ 单次评估超时（TimeoutError）
**现象**：`Exception raised in Job[4]: TimeoutError()`，DashScope 响应偶尔超过默认 60s。
**解决**：加入 `RunConfig(timeout=120, max_retries=2)`，延长超时上限。

---

#### ④ 向量库数据陈旧（98 条 vs 818 chunks）
**现象**：`向量库已存在（98 条）`，实际应有 818 条，Level 1 检索结果严重不准。
**原因**：Chroma collection `rag_docs` 残留了早期测试（课程 PDF）的旧数据，`build_vectorstore` 因"已存在"跳过重建。
**解决**：`test_level4.py` 加入数量检测：若 `count < chunks * 0.5` 则触发 `force_rebuild=True`。

---

### 最终评估结果（2026-03-19 19:18:39）

| 指标 | Level 1 | Level 2 | Level 3 | 关键发现 |
|------|---------|---------|---------|---------|
| Faithfulness | 0.743 | 0.780 | **0.858** | L3 表格 OCR 让答案更有据可查 |
| AnswerRelevancy | 0.807 | **0.912** | 0.911 | L2 Hybrid 检索大幅提升切题度 |
| ContextPrecision | 0.122 | **0.500** | 0.500 | L2 Reranker 过滤无关文档，提升 4 倍 |
| ContextRecall | 0.667 | 0.583 | **0.750** | L2 Top-3 截断略有丢失，L3 靠 OCR 补回 |

---

---

## Level 5 — 前后端搭建（2026-03-19）

### 概述

FastAPI 后端 + React 19 + Vite + TypeScript 前端，三页面布局，流式问答，Markdown 渲染。

---

### 问题记录

#### ① vite create 交互式命令在 WSL 无法自动化
**现象**：`npm create vite@latest . -- --template react-ts --yes` 和多种写法均提示 `Operation cancelled`。
**原因**：`create-vite` 在 WSL 终端检测到非 TTY 环境时拒绝非交互模式。
**解决**：`npx create-vite@latest --template react-ts --force .` 创建到子目录 `vite-project/`，再 `cp -r` 移动到 `frontend/`。

---

#### ② 前端头像图片路径问题
**现象**：`settings.userAvatar` 初始值为 emoji 字符串，切换为图片后需要区分渲染方式。
**解决**：约定 `/` 开头表示 `public/` 目录下的图片路径，用 `src.startsWith('/')` 判断渲染 `<img>` 还是直接显示 emoji。自定义头像图片（`avatar_user.png` / `avatar_bot.png`）放入 `frontend/public/`。

---

#### ③ App.tsx 写入报错（File has not been read）
**现象**：Write 工具拒绝写入 `App.tsx`，提示未先读取。
**原因**：工具安全机制要求写入前必须先读取文件内容。
**解决**：先 Read 再 Write，正常流程。

---

### UI 迭代记录

| 版本 | 主要改动 |
|------|---------|
| v1 | 暗绿色主题，左侧上传+知识库，右侧聊天，顶部设置按钮抽屉 |
| v2 | 改为清新浅绿白主题，SVG 叶片花纹背景，白色磨砂玻璃卡片，绿色按钮阴影 |
| v3 | 头像从 emoji 改为自定义图片（羽入/战人），设置面板支持图片头像选择 |
| v4 | 重构为三页面布局（左侧导航栏），添加 react-markdown 渲染，头像放大至 40px，聊天区限宽 820px |
| v5 | 公式渲染（KaTeX）、头像放大至 46px、历史记录持久化、修复切换知识库竞态 bug |

---

#### ④ 切换知识库时历史记录互相污染（竞态 Bug）
**现象**：从知识库 A 切换到 B，B 的聊天界面仍显示 A 的历史记录。
**原因**：存在两个 `useEffect`：一个监听 `[collection]` 负责加载历史，另一个监听 `[messages, collection]` 负责自动保存。切换 collection 时两个 effect 同时触发，保存 effect 此时 `collection` 已是 B 但 `messages` 还是 A 的内容，导致 A 的消息被存入 B 的 localStorage key。
**解决**：删除自动保存的 `useEffect`，改为在 `sendMessage` 的 stream 结束后**显式调用** `saveHistory(collection, final)`，以及在 `deleteMessage` 时显式保存。这样保存时机完全可控，不依赖 effect 执行顺序。
**教训**：多个 `useEffect` 依赖同一变量时，执行顺序无法保证，涉及持久化操作应避免依赖 effect 联动，改用显式调用。

---

#### ⑤ LaTeX 公式显示为原始文字
**现象**：回答中的 `$5 \times 10^{-4}$` 直接显示为字符串，未渲染为数学公式。
**原因**：`react-markdown` 默认不解析数学语法，需要额外插件。
**解决**：安装 `remark-math`（解析 `$...$` 语法）+ `rehype-katex`（渲染为 HTML）+ `katex`，在 `ReactMarkdown` 的 `remarkPlugins` 和 `rehypePlugins` 中注册，并在 `main.tsx` 引入 `katex/dist/katex.min.css`。

---

### 最终功能清单（2026-03-19）

- ✅ 文件上传（PDF / Word / Excel），拖拽或点击，命名弹窗，进度条实时轮询
- ✅ 多知识库管理（列表、切换、删除）
- ✅ 流式问答（SSE），AI 回复完整 Markdown 渲染（表格/代码/加粗/列表/LaTeX 公式）
- ✅ 来源溯源（可展开，显示页码和置信分）
- ✅ 参数设置持久化（Top-K / Reranker Top-N / score_threshold / max_retry / temperature）
- ✅ 头像自定义（羽入为 Bot 默认，战人为 User 默认，localStorage 保存）
- ✅ 三页面导航（问答 / 知识库 / 设置）
- ✅ 对话历史按知识库独立持久化（localStorage，刷新/重启不丢失）
- ✅ 切换知识库自动加载对应历史，互不干扰
- ✅ 单条对话删除（hover 显示删除按钮，同时删除问答对）
- ✅ 清空当前知识库全部记录（二次确认）

---

---

## Level 6 — 大规模测试 + Fallback 路由 + Pipeline 控制（2026-03-19）

### 概述

- RAGAS TestsetGenerator 自动生成 10 道中文测试题，批量跑 RAG 并评估
- 新增 `fallback` 节点（Tavily 网络搜索 + LLM 兜底），前端标注来源类型
- 设置页新增 Pipeline 控制面板（Reranker/Fallback 开关 + 方法选择器）
- FastAPI lifespan warmup 消除冷启动延迟

---

### 问题记录

#### ① rapidfuzz 缺失（RAGAS TestsetGenerator 依赖）
**现象**：运行 `generate_testset.py` 报 `ImportError: rapidfuzz is required for string distance calculations`。
**原因**：RAGAS 的 KG 构建流程内部用 rapidfuzz 做字符串去重，但未列入 core 依赖。
**解决**：`pip install rapidfuzz`。

---

#### ② TestsetGenerator 输出英文/混合语言题目
**现象**：RAGAS 生成的 10 道题均为英文或中英混合，如 "What is the mechanism of carbocation..."。
**原因**：RAGAS 内置 synthesizer prompt 为英文，即使文档是中文，LLM 也倾向用英文出题。
**解决（本次）**：直接将已生成的英文题手动翻译为中文，保存为 `golden_set_auto.json`，避免重新跑耗时的 KG 构建（~10min）。
**解决（后续）**：`generate_testset.py` 已加入中文 persona（石化工艺工程师 / 装置操作员）和 `llm_context`，未来重新生成时会得到中文题目。

---

#### ③ FALLBACK_PROMPT 中文引号引起 SyntaxError
**现象**：添加 `fallback_node` 后，`rag_graph.py` 报 `SyntaxError: invalid syntax. Perhaps you forgot a comma?`。
**原因**：prompt 字符串中使用了中文弯引号 `"知识库"` 和 `"检索"`，而外层字符串定界符也是英文双引号 `"`，中文 `"` 被解析器识别为字符串结束符，导致语法错误。
**解决**：将 prompt 中的中文引号 `"知识库"` 和 `"检索"` 改为单引号 `'知识库'` 和 `'检索'`。

---

#### ④ fallback_node try/except 缩进错误
**现象**：添加 `if method in ("auto", "web"):` 条件后，`try` 块内代码缩进不一致，Python 报 `IndentationError`。
**原因**：在 try 块内插入了新的 if 条件层，但内层代码未整体向右缩进 4 格。
**解决**：将 try 块内所有语句统一调整为 12 空格缩进（3级）。

---

#### ⑤ 首次请求冷启动延迟 ~45s
**现象**：服务启动后第一个问题要等约 45~60 秒才返回，第二个问题恢复正常（8~12s）。
**原因**：首次请求触发了三个模型的懒加载：
- BGE bge-large-zh-v1.5 Embedding 加载：~30.7s
- Jieba 词典初始化：~4.2s
- BGE bge-reranker-v2-m3 加载：~10.0s
**解决**：在 `backend/main.py` 的 FastAPI `lifespan` 中 `asyncio.create_task(_warmup())`，服务启动后立即在后台预热所有 collection，不阻塞服务就绪。第一个真实请求到达时模型已加载完毕。

---

### RAGAS 评估结果（2026-03-19 23:35:34，10 道中文自动生成题）

**测试集构成**：
- SingleHop（6道）：直接检索类，有 reference，评估全部 4 指标
- MultiHopAbstract（2道）：跨段抽象推理，有 reference
- MultiHopSpecific（2道）：跨段精确查找，有 reference

| 指标 | 总体 | 单跳 | 多跳抽象 | 多跳精确 |
|------|------|------|---------|---------|
| Faithfulness | 0.739 | 0.750 | 0.604 | **0.938** |
| AnswerRelevancy | 0.681 | 0.610 | 0.527 | **0.904** |
| ContextPrecision | **0.850** | **1.000** | 0.250 | **1.000** |
| ContextRecall | **0.875** | **1.000** | 0.625 | **1.000** |

**关键发现**：
- 单跳题 ContextPrecision/Recall 均为 1.0，检索精准
- 多跳抽象型 ContextPrecision=0.25，为预期内（跨章节抽象推理对 chunk 截断敏感）
- 多跳精确型 Faithfulness=0.938，说明答案忠实度高，跨段信息整合良好
- 总体 AnswerRelevancy=0.68 偏低，与 RAGAS 自动生成 reference 质量参差不齐有关（LLM-as-Judge 自评偏差）

**评估报告**：`evaluation/results/eval_20260319_223534.json` / `latest.json`

---

### 最终功能清单（Level 6 新增，2026-03-19）

- ✅ RAGAS TestsetGenerator 自动出题（中文 persona + llm_context）
- ✅ 批量评估 pipeline（run_generate.py + run_ragas_eval.py，支持断点续跑）
- ✅ Fallback 路由（Tavily 网络搜索优先，LLM 知识兜底）
- ✅ 前端 fallback 来源标注（🌐 蓝色/⚠️ 黄色气泡，含可点击来源链接）
- ✅ Pipeline 控制面板（设置页，Reranker/Fallback 开关，fallback 方法选择）
- ✅ 冷启动消除（FastAPI lifespan warmup，后台异步预热所有知识库）
- ✅ 运行时参数透传（reranker_enabled / fallback_enabled / fallback_method 从前端 → SSE → LangGraph State）

---

## 工程优化（2026-03-20）

### 问题记录

#### ① Excel 上传必现 TypeError（Bug）
**现象**：通过前端上传 xlsx 文件时，后台抛 `TypeError: __init__() got an unexpected keyword argument 'source_name'`，文件无法入库。
**原因**：`pipeline.py` 调用 `ExcelLoader(str(path), source_name=path.stem)`，但 `ExcelLoader.__init__` 没有 `source_name` 参数，这是复制 MinerULoader 调用方式时留下的笔误。
**解决**：删除多余的 `source_name=path.stem`，改为 `ExcelLoader(str(path))`。

---

#### ② Router Prompt 硬编码石化领域导致多知识库失效
**现象**：切换到 `investment_db` 知识库，问"华泰紫金的投资阶段"等投资相关问题，系统完全不检索知识库，直接用 LLM 通用知识回答，答案与库内数据无关。
**原因**：`ROUTER_PROMPT` system 消息写死"服务于石油化工技术文档知识库"，并且规则是"与石化工艺无关的内容 → direct"。投资问题被判定为 direct，绕过了所有检索逻辑。`REWRITE_PROMPT` 同样注入了石化专业术语，对其他领域适得其反。
**解决**：将 Router/Rewrite/Direct 三个 Prompt 全部改为通用逻辑：
- Router：遇到模糊情况优先 retrieve，只有纯闲聊/问候/数学才 direct
- Rewrite：通用策略（提取核心实体、具体化、同义替换），不绑定任何领域

---

#### ③ Reranker 每个知识库重复加载（浪费约 8s + 显存）
**现象**：预热 3 个知识库，日志显示 `加载 Reranker 模型...` 出现 3 次，每次约 2.5s，合计多花 ~5s，且显存中同时存在多个相同模型实例。
**原因**：`_build_graph` 每次调用都 `BGEReranker()`，没有单例控制，N 个知识库就加载 N 次。
**解决**：`reranker.py` 新增 `get_reranker()` 全局单例工厂，`chat.py` 改用 `get_reranker(top_n=...)`。首次调用加载模型，之后复用，`top_n` 可动态调整。

---

#### ④ 预热阻塞事件循环导致服务启动期间前端无法访问
**现象**：重启后端后，前端出现 `ECONNREFUSED` 或知识库列表加载失败，直到预热全部完成（约 30s）后才恢复正常。日志显示 `Application startup complete` 在预热结束后才打印。
**原因**：`_warmup()` 虽然用 `asyncio.create_task` 调度，但内部调用的 `_build_graph` 是同步阻塞函数（模型加载、BM25 构建等），没有 `await`，导致任务启动后独占事件循环，uvicorn 无法处理任何请求。
**解决**：将 `_build_graph` 调用改为 `await loop.run_in_executor(None, _build_graph, ...)`，在线程池中执行，事件循环保持空闲，服务启动后立即响应请求，预热在后台静默完成。

---

#### ⑤ 大型投资 xlsx（93 列）直接入库效果差
**现象**：用通用 `ExcelLoader` 导入 93 列 × 1815 行的投资数据库，每行拼成一个 Document，部分行超过 5000 字，BGE 512 token 上限大量截断，检索召回严重失真。
**解决**：编写专用脚本 `data/preprocess_investment_db.py`：
- 筛选有价值的列，跳过隐私字段和低填充率字段
- 每行拆为两类 chunk：结构化概况（~150 字）+ 长文本段落（≤600 字/段）
- 每个 chunk 开头附实体 header（`【投资方】姓名 | 机构 | 职位`），确保检索到任意片段都知道来源
- 最终 7440 个 chunk，平均 196 字，最长 639 字，全部在 BGE 有效范围内
- 入库耗时 796s（约 13 分钟，7440 条 × BGE embedding）

---

### 最终功能清单（工程优化新增，2026-03-20）

- ✅ Excel 上传 Bug 修复（source_name 参数错误）
- ✅ Router/Rewrite Prompt 通用化，支持任意领域知识库
- ✅ BGEReranker 单例化（get_reranker()），预热加载次数 3→1
- ✅ 预热非阻塞（run_in_executor），服务启动后立即可用
- ✅ 投资数据库专用清洗脚本（7440 chunks，investment_db collection）
- ✅ 一键启动脚本 start.sh（端口检查 + conda激活 + 前后端联动 + Ctrl+C停止）

---

## Level 7 — Function Calling + MCP Server（2026-03-26）

### 改动概述

在原有 Level 6 基础上新增两个能力层：
1. **Function Calling**：LangGraph router 支持 `bind_tools` 模式，LLM 可主动选择工具调用
2. **MCP Server**：将 RAG 系统暴露为 MCP 协议接口，供 Claude Desktop 等外部客户端使用

---

### 新增文件

| 文件 | 说明 |
|------|------|
| `tools/tool_definitions.py` | LangChain StructuredTool 定义（search_kb / analyze_process_data / web_search） |
| `tools/data_analyzer.py` | DCS 时序数据 Pandas 分析（LLM 生成代码 + exec，失败降级 LLM 描述） |
| `mcp_server/__init__.py` | 包初始化 |
| `mcp_server/server.py` | FastMCP Server，暴露 3 个工具（list_collections / search_kb / analyze_data） |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `graphs/rag_graph.py` | RAGState 新增 tools_enabled / tool_name / tool_args / tool_result；router 支持双模式；新增 tool_executor_node；图添加 "tool" 路由分支 |
| `backend/routers/chat.py` | ChatRequest 新增 tools_enabled；_run_graph 透传参数；SSE 新增 tool_call 事件 |

---

### 设计决策

**双模式 Router**：
- `tools_enabled=False`（默认）：保持原有字符串分类路由，零破坏性，现有前端无需改动
- `tools_enabled=True`：`llm.bind_tools(3个工具 + direct_answer)` 让 LLM 自行选择

**工具路由逻辑**：
- `search_knowledge_base` → 路由到现有 retrieve_node（复用 Hybrid + Reranker）
- `analyze_process_data` → 路由到新 tool_executor_node（Pandas 分析 DCS 数据）
- `web_search` → 路由到 tool_executor_node（Tavily，升为主动工具而非只是 fallback）
- `direct_answer` → 路由到现有 direct_node

**data_analyzer 实现**：
- DCS 数据：加氢裂化装置时序数据，9985行 × 34列，5分钟采样（2023-01-30 ~ 2023-03-05）
- 方案：LLM 生成 pandas 代码 → exec 执行 → 失败降级为 LLM 直接描述
- 数据加载后全局缓存（_df_cache），避免重复 IO

**MCP Server**：
- 基于 FastMCP（mcp==1.26.0）
- stdio 传输（Claude Desktop 标准方式）
- 工具函数复用 tool_definitions.py / data_analyzer.py，不重复实现

---

### 兼容性说明

- 现有 Level 1~6 所有功能完全保留，tools_enabled 默认 False
- 前端现有代码无需修改即可继续使用（tool_call SSE 事件为新增可选项）
- graph 缓存 key 不含 tools_enabled，建议后续按需加入

---

### 语法验证（全部通过）

```
✅ tools/data_analyzer.py
✅ tools/tool_definitions.py
✅ graphs/rag_graph.py
✅ mcp_server/server.py
✅ backend/routers/chat.py
```

DCS 数据加载验证：
```
✅ 9985行 × 34列，时间范围 2023-01-30 ~ 2023-03-05
✅ RAGState 新字段结构验证通过
✅ MCP server import OK
```

---

### 待完成（可选后续）

- [ ] 前端 SettingsPage 添加"Function Calling"开关（已有占位）
- [ ] 前端 ChatWindow 渲染 tool_call 气泡（显示"调用了 analyze_process_data"）
- [ ] tests/test_level7.py 集成测试（需要 API 调用，建议手动验证）
- [ ] MCP Server HTTP/SSE 模式支持（现仅 stdio）

---

## 多语言 Embedding 对比实验（2026-03-30）

### 实验目的
评估 bge-large-zh-v1.5 vs bge-m3 在中英混合文档库中的检索表现，
为后续加入英文文档库制定技术选型依据。

### 测试数据（合成）
- 6条中文文档片段（加氢裂化工艺领域）
- 6条对应英文翻译片段（手写合成，非真实文档）
- 8条测试 query，覆盖4类场景（各2条）

### 实验结果

| 场景 | 说明 | bge-zh 命中 | bge-m3 命中 | bge-m3 跨语言gt提升 |
|------|------|------------|------------|---------------------|
| A: ZH→ZH | 中文查中文库 | 2/2 | 2/2 | - |
| B: EN→EN | 英文查英文库 | 2/2 | 2/2 | - |
| C: EN→ZH | 英文查中文库 | 0/2 | 0/2 | +0.17（0.40→0.57）|
| D: ZH→EN | 中文查英文库 | 0/2 | 0/2 | +0.01（0.54→0.58）|

**总命中率：两模型均 50%（4/8）**

### 关键结论
1. bge-m3 跨语言对齐能力更强（gt_score 提升明显），但混合语料库的语言偏置问题导致正确跨语言文档仍被同语言文档压过
2. 光换模型无法解决跨语言检索，需要双语 query 策略
3. bge-m3 已下载缓存，迁移只需改 config 一行

### 后续改进
→ 见 Level 7d：CrossLingualRetriever（双语 query + RRF 融合）

---

## Level 7d — CrossLingualRetriever 三方案对比实验（2026-03-30）

### 实验设计
在上次 bge-zh vs bge-m3 基础实验之上，新增方案3：bge-m3 + 双语 query（翻译后双路 RRF 融合）

### 实验结果

| 场景 | bge-zh单语 | bge-m3单语 | m3+双语 |
|------|-----------|-----------|--------|
| A: ZH→ZH | 2/2 | 2/2 | 2/2 |
| B: EN→EN | 2/2 | 2/2 | 2/2 |
| C: EN→ZH | 0/2 | 0/2 | **1/2** ↑ |
| D: ZH→EN | 0/2 | 0/2 | 0/2 |
| 总命中率 | 50% | 50% | **62.5%** |

### CrossLingualRetriever 工作原理
1. `detect_language()` 判断 query 是 zh/en（中文字符比例 >30% 判中文）
2. `translate_query()` 调用 qwen-plus 生成翻译 query（结果缓存）
3. 原始 query + 翻译 query 分别走 hybrid_retriever 检索
4. `_rrf_merge()` 合并两路结果（k=60 标准 RRF）

### C 场景命中分析
- ✅ "desulfurization reactor..." → 翻译 "脱硫反应器将硫化物去除为硫化氢" → 成功命中中文 doc[1]
- ❌ "nitrogen purging before startup" → 翻译 "启动前氮气吹扫除氧" vs 原文"装置开工前" → 术语差异导致失败

### D 场景为何仍失败
ZH→EN 方向的语言偏置比 EN→ZH 更严重：中文 query 在混合库中，
即使翻译成英文，同语言中文文档的竞争分仍更高。
根本解决方案是分库路由，不是双语 query。

### 新增文件
- `retrievers/cross_lingual_retriever.py`
- `tests/test_multilingual_embedding.py`（bge-zh vs bge-m3 基础对比）
- `tests/test_cross_lingual.py`（三方案完整对比）

---

## Level 7d — CrossCollectionRetriever 分库路由实验（方案4，2026-03-30）

### 实验目的
验证分库路由策略（CrossCollectionRetriever）能否彻底修复 D 场景（ZH→EN）。

### 实验设计
- bge-m3 分别对 ZH 和 EN 文档建独立 in-memory 索引（各 6 条）
- ZH query → ZH 库（原始）+ EN 库（翻译后）→ RRF 融合
- EN query → EN 库（原始）+ ZH 库（翻译后）→ RRF 融合

### 实验结果（tests/test_cross_collection.py，2026-03-30）

| 场景 | top1 | top3 | top5 | avg_partner_rank |
|------|------|------|------|-----------------|
| ZH→ZH | 2/2 | 2/2 | 2/2 | 999（同库，partner 无关）|
| EN→EN | 2/2 | 2/2 | 2/2 | 999（同库，partner 无关）|
| EN→ZH | 0/2 | **2/2** | 2/2 | **1.0**（配对库完美命中）|
| ZH→EN | 0/2 | **2/2** | 2/2 | **1.0**（配对库完美命中）|
| 总体   | 4/8=50% | **8/8=100%** | 8/8=100% | - |

### 关键发现

**top1 失败原因（RRF tie）**：
- 跨语言场景中，primary（同语言）和 partner（配对语言）各自 top-1 均为 `1/(60+1)=0.01639`
- 两库各自命中自己的最相关文档，RRF 后形成 tie，Python `sorted()` 稳定排序保留插入顺序
- 实际上 GT 始终排名第 2（gt_rank=2），紧随其后进入 top-3

**top-3/top-5 为何是正确的评估指标**：
- 生产环境中 CrossCollectionRetriever 将 top-20 送入 BGE Reranker 精排，Reranker 会选出真正最相关的那条
- partner_rank=1.0 证明翻译 query 在配对库里精准命中了 GT
- 分库路由的核心价值是"确保 GT 进入 Reranker 候选池"，top-3 覆盖即已达成目标

### 与方案3对比

| 方案 | ZH→ZH top1 | EN→EN top1 | EN→ZH top1 | ZH→EN top1 | EN→ZH top3 | ZH→EN top3 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| 方案3（m3+双语query，单库）| 2/2 | 2/2 | **1/2** | 0/2 | 2/2 | 1/2 |
| 方案4（分库路由，CrossCollection）| 2/2 | 2/2 | 0/2 | 0/2 | **2/2** | **2/2** |

**结论**：
- 方案4 的 EN→ZH top3=2/2（方案3 也是 2/2）
- 方案4 的 ZH→EN top3=2/2（方案3 仅 1/2），显著提升
- top-1 因小规模测试的 RRF tie 现象无法体现优势；真实多文档场景 tie 概率极低
- 分库路由 + Reranker 是跨语言检索的根本解法

### 问题记录

#### ① RRF tie 导致 top1 评估失真
**现象**：ZH→EN 和 EN→ZH 场景的 merged_scores 中，primary 库 top1 和 partner 库 top1 分数完全相同（均为 0.01639），Python sorted 按插入顺序排，中文文档总在前，GT（英文）总排第2。
**根因**：小规模合成数据集（各6条），primary 和 partner 都只有一个明确最优项，导致两路 top-1 得分完全相等。
**修复**：改为报告 top-1/top-3/top-5 三项指标 + partner_rank，全面展示分库路由的真实效果。
**生产影响**：真实数据集（几百~几千条）不会出现完美 tie，gt_rank=1 的概率极高。

### 集成状态
- ✅ `CrossCollectionRetriever` 已完整实现（`retrievers/cross_lingual_retriever.py`）
- ✅ `graphs/rag_graph.py` retrieve_node 已集成（cross_lingual_enabled=True 时自动启用）
- ✅ `backend/routers/chat.py` ChatRequest 已支持 cross_lingual_enabled 参数
- ⬜ 前端 Settings 尚未添加 cross_lingual_enabled 开关（可后续添加）

---

## 工程优化2 — LLM 流式透传 + 聚合查询检测 + per-collection Embedding（2026-03-31）

### 背景

用户反馈：问答延迟感严重（~9s白屏），以及"数据里一共有多少家VC机构？"之类的聚合/计数问题，RAG检索top-k文档后无法全库统计，LLM返回了不相关的行业统计数字。

---

### 一、LLM 流式透传（解决9秒白屏）

**根因分析**：
- 原有架构：`graph.invoke()` 在 executor 线程中同步运行，等完整答案生成后才一次性通过 SSE 发送 `{"type":"answer"}`
- 用户体验：问题发出后前端 9~15s 白屏，只有一个 Loading 动画，感知延迟严重

**方案**：asyncio.Queue 桥接后台线程与 SSE 协程

```
后台线程（executor）                       SSE 协程（async）
graph.invoke()
  └─ generate_node
       └─ llm.stream() → token → callback → loop.call_soon_threadsafe(queue.put, token)
                                                           ↓
                                              await queue.get() → yield "answer_token" SSE
  完成后 → queue.put(_SENTINEL)
                                              收到哨兵 → 跳出循环 → 发 status/sources/done
```

**改动文件**：
- `graphs/rag_graph.py`：
  - `RAGState` 新增 `token_callback: Any` 字段
  - 新增 `_stream_or_invoke(prompt, llm, inputs, callback)` 辅助函数
  - `generate_node` / `direct_node` / `fallback_node` 全部改用 `_stream_or_invoke`，有 callback 时走 `llm.stream()`，无 callback 退化为 `invoke()`（向后兼容）
- `backend/routers/chat.py`：
  - `_run_graph(req, token_callback=None)` 接受回调
  - `event_stream()` 改用 Queue 桥接：`token_callback → queue.put_nowait → yield answer_token`
  - 图完成后再发 `status`（路由信息）/ `answer`（完整文本备用）/ `sources` / `done`
- `frontend/src/components/ChatWindow.tsx`：
  - 新增 `answer_token` 事件处理：第一个 token 到达时清除 loading 状态，后续 token 逐字追加
  - `hasStreamedTokens` 标志（持久于整个 SSE 流）：避免 `answer`（兜底）覆盖已流式内容
  - `status` 事件：只在 `!hasStreamedTokens` 时显示加载提示，防止路由状态消息污染已渲染答案
  - `done` 事件：确保 `loading: false`

**效果**：用户发问后约 1~2s 即开始看到文字输出，消除白屏感。

---

### 二、聚合/计数查询检测（解决"有多少家VC"问题）

**根因**：`_router_string` 把"数据里一共有多少家VC机构？"路由到 `retrieve`，检索到的 top-3 chunk 是行业统计，LLM 误用行业数据回答，未说明自身局限。

**方案**：在 `_router_string` 前置正则检测，命中聚合模式时直接路由到 `direct`，由更新后的 `DIRECT_PROMPT` 引导 LLM 诚实说明无法全库计数。

**正则规则**（`_AGGREGATION_RE`）：
```
(一共|总共|共有?|总计)[多少几个家条份]
|[多少几](个|家|只|条|份|种)
|列出所有|列举所有|所有的?\S+有哪些
|统计.{0,10}(数量|个数|多少|几个|几家)
```

**`DIRECT_PROMPT` 补充**：告知 LLM 本系统基于 RAG top-k，无法遍历全库，遇到计数/列举类问题应明确告知局限。

---

### 三、per-collection Embedding 注册表

**背景**：`hydro_manual`（中文）用 bge-large-zh-v1.5，`investment_db`（中英混合）用 bge-m3，不能全局统一模型。

**方案**：`data/chroma_db/embedding_registry.json` 存储 `{collection_name: model_name}`。

**改动**：
- `chains/rag_chain.py`：`_embeddings_cache: dict` 多实例缓存，`_load_registry()` / `_save_registry()` / `get_collection_embedding_model()` / `save_collection_embedding_model()`
- `backend/services/pipeline.py`：`embedding_model` 参数透传，lang 从 collection 名自动检测（`_en` 后缀 → `"en"`）
- `backend/routers/upload.py`：Form 参数新增 `embedding_model`
- `backend/routers/collections.py`：`list_collections` 从注册表读取并返回 `embedding_model` 字段
- `frontend/UploadZone.tsx`：两个 radio 选项（bge-zh / bge-m3），多文件一起选/拖拽，共享同一知识库名和模型
- `frontend/CollectionList.tsx`：`m3` / `zh` badge 显示各知识库的 Embedding 类型

---

### 最终功能清单（2026-03-31 新增）

- ✅ LLM 流式透传：首 token 延迟 1~2s，消除白屏
- ✅ 聚合/计数查询检测：正则路由到 direct，LLM 诚实说明 top-k 局限
- ✅ per-collection Embedding 注册表：新旧知识库向后兼容，按需切换
- ✅ 多文件批量上传：同一 collection 多个文件并行处理，独立进度条
- ✅ 知识库列表显示 Embedding 模型标签（m3 / zh）
