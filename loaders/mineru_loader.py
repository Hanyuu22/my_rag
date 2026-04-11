"""
MinerU Document Loader
读取 MinerU 输出的 _content_list.json，转成 LangChain Document 列表。

MinerU content_list 每个元素结构：
{
  "type": "text" | "table" | "image" | "equation",
  "text": "...",         # text/equation 有
  "img_path": "...",     # image 有
  "table_caption": [],   # table 有
  "page_idx": 0
}
"""

import json
from pathlib import Path
from typing import Iterator, List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class MinerULoader(BaseLoader):
    """
    加载 MinerU 解析输出的 _content_list.json。

    Args:
        content_list_path: _content_list.json 文件路径
        source_name: 来源标识（通常是原始 PDF 文件名）
        include_types: 要加载的 block 类型，默认只要 text 和 table
    """

    def __init__(
        self,
        content_list_path: str,
        source_name: str = "",
        include_types: List[str] = None,
    ):
        self.path = Path(content_list_path)
        self.source_name = source_name or self.path.stem.replace("_content_list", "")
        self.include_types = include_types or ["text", "table", "equation"]

    def lazy_load(self) -> Iterator[Document]:
        with open(self.path, encoding="utf-8") as f:
            content_list = json.load(f)

        for block in content_list:
            block_type = block.get("type", "")
            if block_type not in self.include_types:
                continue

            # 提取文本内容
            if block_type == "text":
                text = block.get("text", "").strip()
            elif block_type == "table":
                # 表格：caption + body（MinerU 输出的 table 有 text 字段含 markdown 表格）
                caption = " ".join(block.get("table_caption", []))
                body = block.get("text", "")
                text = f"{caption}\n{body}".strip() if caption else body.strip()
            elif block_type == "equation":
                text = block.get("text", "").strip()
            else:
                continue

            if not text:
                continue

            metadata = {
                "source": self.source_name,
                "page": block.get("page_idx", -1),
                "block_type": block_type,
            }

            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        return list(self.lazy_load())
