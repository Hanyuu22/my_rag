"""
Excel Document Loader

xlsx 和 PDF 的本质区别：xlsx 是结构化表格，不需要 OCR 解析，
也不按字符切分——每行（或每 N 行）就是一个 Document。

支持两种模式：
- row_mode：每行一个 Document（适合产品目录、知识条目、QA 对）
- sheet_mode：每个 sheet 一个 Document（适合少量大表）

输入：.xlsx 文件路径
输出：List[Document]，metadata 含 source / sheet / row_index
"""

from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class ExcelLoader(BaseLoader):
    """
    加载 xlsx 文件，按行转成 LangChain Document。

    Args:
        file_path: xlsx 文件路径
        sheet_name: 指定 sheet，None 表示加载所有 sheet
        text_columns: 要拼成正文的列名列表，None 表示所有列
        metadata_columns: 要放进 metadata 的列名列表
        row_mode: True=每行一个Doc，False=整个sheet一个Doc
    """

    def __init__(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        text_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        row_mode: bool = True,
    ):
        self.path = Path(file_path)
        self.sheet_name = sheet_name
        self.text_columns = text_columns
        self.metadata_columns = metadata_columns or []
        self.row_mode = row_mode

    def lazy_load(self) -> Iterator[Document]:
        xl = pd.ExcelFile(self.path)
        sheets = [self.sheet_name] if self.sheet_name else xl.sheet_names

        for sheet in sheets:
            df = xl.parse(sheet)
            df = df.fillna("")  # NaN 转空字符串

            text_cols = self.text_columns or list(df.columns)

            if self.row_mode:
                for idx, row in df.iterrows():
                    # 正文：选定列拼成 "列名：值" 格式
                    text = "\n".join(
                        f"{col}：{row[col]}"
                        for col in text_cols
                        if str(row[col]).strip()
                    )
                    if not text.strip():
                        continue

                    metadata = {
                        "source": self.path.name,
                        "sheet": sheet,
                        "row_index": idx,
                        "block_type": "table_row",
                    }
                    for col in self.metadata_columns:
                        if col in row:
                            metadata[col] = str(row[col])

                    yield Document(page_content=text, metadata=metadata)
            else:
                # 整个 sheet 转成 markdown 表格
                text = df[text_cols].to_markdown(index=False)
                metadata = {
                    "source": self.path.name,
                    "sheet": sheet,
                    "block_type": "table",
                }
                yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        return list(self.lazy_load())
