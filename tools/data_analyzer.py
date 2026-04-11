"""
工艺数据 Pandas 分析工具

分析 data/raw/ 下的 DCS 时序 xlsx 文件（加氢裂化工艺数据集）。
适用于统计均值/极值、时段筛选、趋势判断等向量检索无法处理的数值查询。
"""

import os
import glob
import pandas as pd
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")

_df_cache: Optional[pd.DataFrame] = None


def _load_dcs_data() -> Optional[pd.DataFrame]:
    """
    加载工艺 DCS 数据，结构：
      行0（header）：中文列名（总进料泵P101A出口总流量 等）
      行1（数据第0行）：仪表 tag 代码（X2XJQLH... 等），需跳过
      行2+：时间戳 + 数值
    """
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    files = glob.glob(os.path.join(_DATA_RAW_DIR, "*.xlsx"))
    if not files:
        return None

    try:
        # header=0 取中文列名，skiprows=[1] 跳过仪表 tag 行
        df = pd.read_excel(files[0], header=0, skiprows=[1])
        df = df.rename(columns={df.columns[0]: "时间"})
        df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
        df = df.dropna(subset=["时间"]).reset_index(drop=True)
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        _df_cache = df
        return df
    except Exception as e:
        return None


def _build_context(df: pd.DataFrame) -> str:
    """为 LLM 生成数据概况描述"""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    stats_lines = []
    for col in num_cols[:15]:
        s = df[col]
        stats_lines.append(
            f"  {col}: 均值={s.mean():.3f}, 最小={s.min():.3f}, 最大={s.max():.3f}"
        )
    return (
        f"DataFrame `df` 已加载\n"
        f"行数: {len(df)}，时间范围: {df['时间'].min()} ~ {df['时间'].max()}\n"
        f"全部列名: {list(df.columns)}\n"
        f"数值列统计（前15列）:\n" + "\n".join(stats_lines)
    )


def analyze_process_data(query: str) -> str:
    """
    对工艺 DCS 数据进行自然语言查询，返回分析结果。

    实现步骤：
      1. LLM 根据数据概况生成 pandas 代码
      2. 安全执行代码，取 result 变量
      3. 执行失败则回退到 LLM 直接依据概况回答
    """
    df = _load_dcs_data()
    if df is None:
        return "未找到 DCS 工艺数据文件（data/raw/*.xlsx）"

    context = _build_context(df)

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from chains.rag_chain import get_llm

    CODE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "你是工业数据分析专家。DataFrame `df` 已加载（pandas），请为用户问题生成 pandas 分析代码。\n\n"
         "数据概况：\n{context}\n\n"
         "要求：\n"
         "- 只输出纯 Python 代码，不要 markdown 代码块\n"
         "- 最后一行必须是 `result = <某个值>` 赋值\n"
         "- result 必须可用 str() 转换\n"
         "- 不超过 12 行，使用 df 和 pd 变量\n"
         "- 列名必须与数据概况中一致"),
        ("human", "{query}"),
    ])

    llm = get_llm()
    code = (CODE_PROMPT | llm | StrOutputParser()).invoke({
        "context": context, "query": query
    }).strip()

    # 移除 markdown 代码块
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(
            lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        )

    try:
        local_ns: dict = {"df": df, "pd": pd}
        exec(compile(code, "<data_analyzer>", "exec"), {"pd": pd, "df": df}, local_ns)
        result = local_ns.get("result", "执行完成，但未设置 result 变量")
        return str(result)
    except Exception as exec_err:
        # 降级：LLM 基于概况直接回答
        FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
            ("system",
             "你是工业数据分析专家。基于以下数据概况，直接回答用户问题。\n"
             "如无法精确计算，给出基于均值/范围的估计并说明。\n\n"
             "数据概况：\n{context}"),
            ("human", "{query}"),
        ])
        answer = (FALLBACK_PROMPT | llm | StrOutputParser()).invoke({
            "context": context, "query": query
        })
        return f"{answer}\n\n（注：自动代码执行失败：{exec_err}）"
