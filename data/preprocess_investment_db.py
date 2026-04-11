"""
投资数据库 xlsx 专用预处理脚本

输入：总表_更新版_20250729_155718.xlsx（两个 Sheet）
输出：直接写入 Chroma 向量库（collection: investment_db）

设计思路：
  每行拆成两类 chunk：
  ① 结构化概况 chunk（身份 + 关键字段）→ 短、精准
  ② 长文本 chunk（简介/纪要/备注等）→ 每段 ≤ 600 字，开头附实体标识

  两类 chunk 开头都带实体 header，确保检索到任意 chunk 都能知道是谁。

用法：
  conda activate mineru_2.5
  cd ~/rag_project
  python data/preprocess_investment_db.py --xlsx /mnt/d/Download/work/总表_更新版_20250729_155718.xlsx
"""

import sys
import argparse
sys.path.insert(0, "/home/hanyuu/rag_project")

import pandas as pd
from langchain_core.documents import Document
from chains.rag_chain import build_vectorstore

# ── 参数 ────────────────────────────────────────────────────────────────────

COLLECTION = "investment_db"
CHUNK_SIZE  = 600   # 长文本每段最大字符数（BGE 约 512 token，中文约 600 字）

# ── Sheet 1：投资方（投资人&机构）列配置 ────────────────────────────────────

# 短字段：直接拼进结构化 chunk
INVESTOR_STRUCT_COLS = [
    "投资人姓名", "所在机构", "职位", "在职/离职",
    "关注细分赛道", "FAwork行业", "机构类型", "投资阶段",
    "单笔金额", "币种", "总部", "机构规模",
    "机构别名", "关系状态", "投资压力", "负责人",
    "地区", "返投地区", "能领投么？", "机构_备注",
    "决策流程（组成人员、机制、流程）",
]

# 长文本字段：单独拆 chunk，开头带实体标识
INVESTOR_LONG_COLS = [
    "简介",                              # 66% 填充，机构详细介绍
    "机构简介",                          # 37%
    "投资项目方向/偏好（看好赛道、项目类型、收入、估值标准）",  # 5%，核心偏好
    "基金老大介绍",                      # 4%，主要负责人介绍
    "近期基金情况",                      # 2%，最新基金状态
    "具体压力说明",                      # 5%，投资/返投压力细节
    "备注/返投城市",                     # 39%
    "备注",                              # 19%
    "有哪些 LP？ LP 对你们有什么要求？",
    "有哪些产业资源？产业方对你们有什么要求？",
    "医疗组情况介绍（负责人、团队话语权、覆盖领域）",
    "投资人个人情况",
    "更新",                              # 0% 但极长，有时含完整访谈记录
]

# 跳过：隐私（联系方式）/ 无意义（时间戳、状态标记、序号）
INVESTOR_SKIP_COLS = {
    "联系方式", "邮箱", "手机号", "微信号",
    "更新时间", "创建时间 1", "建联时间", "创建人", "更新人",
    "纪要更新",   # 只是"已处理/已更新"等操作标记
    "纪要", "投资人",  # 0% 填充
    "历史推荐",
}

# ── Sheet 2：项目列配置 ──────────────────────────────────────────────────────

PROJECT_STRUCT_COLS = [
    "项目名称", "公司名称", "一句话介绍", "FAwork行业", "FAwork标签",
    "地区", "优先级", "开发状态", "项目来源", "项目状态",
    "综合推荐指数", "成立时间", "币种", "团队人数",
    "上一年年收入、利润情况", "今年收入、利润情况",
    "投前估值/亿元", "融资金额",
    "能否落地", "落地城市、或要求",
    "下一步计划", "能否排会",
]

PROJECT_LONG_COLS = [
    "简介",
    "纪要更新",   # 17% 填充，项目访谈记录
    "反馈",
    "备注",
]

PROJECT_SKIP_COLS = {
    "文件", "文件（文件名+文件地址）", "文件数",
    "联系方式", "手机号", "邮箱", "微信号",
    "创建时间", "创建人", "更新人", "纪要更新时间",
    "开发时间", "项目状态更新时间",
    "联系人-开发状态", "联系人-开发时间", "联系人-创建人",
    "联系人-创建时间", "联系人-更新时间", "联系人-更新人", "联系人-备注",
    "不用看这个/项目阶段",
    "序列号", "名单", "工作小组",
}


# ── 通用工具 ────────────────────────────────────────────────────────────────

def split_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """把长文本按段落 / 句子边界分段，每段 ≤ size 字符。"""
    if not text or len(text) <= size:
        return [text] if text else []

    # 先尝试在自然段落边界切
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + "\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            # 段落本身就超长，按字符暴力切
            while len(p) > size:
                chunks.append(p[:size])
                p = p[size:]
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def clean(val) -> str:
    """pandas 值 → 干净字符串，过滤空/NaN。"""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    s = str(val).strip()
    # 去掉 RAGFlow 导出时的特殊包装，如 [""] / ['']
    if s in ('[""]', "['']", '[]', '""', "''"):
        return ""
    return s


def make_struct_chunk(header: str, row, col_list: list, all_cols: set) -> str:
    """把结构化字段拼成单个文本块。"""
    lines = [header]
    for col in col_list:
        if col not in all_cols:
            continue
        val = clean(row.get(col, ""))
        if val:
            lines.append(f"{col}：{val}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Sheet 1 处理 ────────────────────────────────────────────────────────────

def process_investors(df: pd.DataFrame) -> list[Document]:
    docs = []
    all_cols = set(df.columns)

    for idx, row in df.iterrows():
        name  = clean(row.get("投资人姓名", "")) or f"Row{idx}"
        org   = clean(row.get("所在机构", "")) or clean(row.get("机构", ""))
        role  = clean(row.get("职位", ""))
        header = f"【投资方】{name}"
        if org:
            header += f" | {org}"
        if role:
            header += f" | {role}"

        # ① 结构化 chunk
        struct_text = make_struct_chunk(header, row, INVESTOR_STRUCT_COLS, all_cols)
        if struct_text:
            docs.append(Document(
                page_content=struct_text,
                metadata={
                    "source": "investment_db",
                    "sheet": "投资方",
                    "entity": name,
                    "org": org,
                    "chunk_type": "structured",
                    "row_index": idx,
                    "block_type": "table_row",
                }
            ))

        # ② 长文本 chunk（每个字段单独处理）
        for col in INVESTOR_LONG_COLS:
            if col not in all_cols:
                continue
            val = clean(row.get(col, ""))
            if not val or len(val) < 20:  # 太短的跳过
                continue
            for i, segment in enumerate(split_text(val)):
                label = col if len(split_text(val)) == 1 else f"{col}（第{i+1}段）"
                docs.append(Document(
                    page_content=f"{header}\n【{label}】\n{segment}",
                    metadata={
                        "source": "investment_db",
                        "sheet": "投资方",
                        "entity": name,
                        "org": org,
                        "chunk_type": "notes",
                        "field": col,
                        "row_index": idx,
                        "block_type": "table_row",
                    }
                ))

    return docs


# ── Sheet 2 处理 ────────────────────────────────────────────────────────────

def process_projects(df: pd.DataFrame) -> list[Document]:
    docs = []
    all_cols = set(df.columns)

    for idx, row in df.iterrows():
        name   = clean(row.get("项目名称", "")) or f"Row{idx}"
        intro  = clean(row.get("一句话介绍", ""))
        header = f"【项目】{name}"
        if intro:
            header += f" — {intro}"

        # ① 结构化 chunk
        struct_text = make_struct_chunk(header, row, PROJECT_STRUCT_COLS, all_cols)
        if struct_text:
            docs.append(Document(
                page_content=struct_text,
                metadata={
                    "source": "investment_db",
                    "sheet": "项目",
                    "entity": name,
                    "chunk_type": "structured",
                    "row_index": idx,
                    "block_type": "table_row",
                }
            ))

        # ② 长文本 chunk
        for col in PROJECT_LONG_COLS:
            if col not in all_cols:
                continue
            val = clean(row.get(col, ""))
            if not val or len(val) < 20:
                continue
            for i, segment in enumerate(split_text(val)):
                label = col if len(split_text(val)) == 1 else f"{col}（第{i+1}段）"
                docs.append(Document(
                    page_content=f"{header}\n【{label}】\n{segment}",
                    metadata={
                        "source": "investment_db",
                        "sheet": "项目",
                        "entity": name,
                        "chunk_type": "notes",
                        "field": col,
                        "row_index": idx,
                        "block_type": "table_row",
                    }
                ))

    return docs


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="xlsx 文件路径")
    parser.add_argument("--collection", default=COLLECTION, help="Chroma collection 名称")
    parser.add_argument("--dry-run", action="store_true", help="只统计 chunk 数，不入库")
    args = parser.parse_args()

    print(f"读取：{args.xlsx}")
    xl = pd.ExcelFile(args.xlsx)

    all_docs = []

    if "投资方（投资人&机构）" in xl.sheet_names:
        df = xl.parse("投资方（投资人&机构）").fillna("")
        print(f"Sheet 投资方：{len(df)} 行")
        investor_docs = process_investors(df)
        print(f"  → 生成 {len(investor_docs)} 个 chunk")
        all_docs.extend(investor_docs)

    if "项目" in xl.sheet_names:
        df = xl.parse("项目").fillna("")
        print(f"Sheet 项目：{len(df)} 行")
        project_docs = process_projects(df)
        print(f"  → 生成 {len(project_docs)} 个 chunk")
        all_docs.extend(project_docs)

    # 统计
    struct_n = sum(1 for d in all_docs if d.metadata.get("chunk_type") == "structured")
    notes_n  = sum(1 for d in all_docs if d.metadata.get("chunk_type") == "notes")
    avg_len  = sum(len(d.page_content) for d in all_docs) / len(all_docs) if all_docs else 0
    max_len  = max(len(d.page_content) for d in all_docs) if all_docs else 0

    print(f"\n汇总：")
    print(f"  总 chunk 数：{len(all_docs)}")
    print(f"  结构化 chunk：{struct_n}")
    print(f"  长文本 chunk：{notes_n}")
    print(f"  平均长度：{avg_len:.0f} 字")
    print(f"  最长 chunk：{max_len} 字")

    # 打印几个样例
    print("\n── 样例 chunk（前3个）──")
    for d in all_docs[:3]:
        print(f"\n[{d.metadata['chunk_type']}] {d.metadata.get('entity','')}")
        print(d.page_content[:300])
        print("...")

    if args.dry_run:
        print("\n[dry-run] 未入库，退出。")
        return

    # 入库
    print(f"\n向量化并写入 Chroma collection='{args.collection}'...")
    build_vectorstore(all_docs, collection_name=args.collection, force_rebuild=True)
    print("✓ 完成")


if __name__ == "__main__":
    main()
