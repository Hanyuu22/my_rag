"""
MinerU Document Loader
读取 MinerU 输出的 _content_list.json，转成 LangChain Document 列表。

MinerU content_list 每个元素结构：
{
  "type": "text" | "table" | "image" | "equation",
  "text": "...",         # text/equation 有；table 的 text 字段可能被截断
  "table_body": "...",   # table 专有，完整 HTML，优先于 text 字段
  "img_path": "...",     # image 有
  "table_caption": [],   # table 有
  "page_idx": 0
}
"""

import json
import re
from pathlib import Path
from typing import Iterator, List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


def _html_has_more_rows(html: str, text: str) -> bool:
    """HTML 中的 <tr> 数量比 text 里的数据行数多 → text 被截断，需用 HTML。"""
    html_rows = html.count("<tr")
    text_rows = sum(
        1 for l in text.split("\n")
        if l.strip().startswith("|") and not l.strip().startswith("|---")
    )
    return html_rows > text_rows + 1


def _html_table_to_markdown(html: str) -> str:
    """将 HTML 表格转为 Markdown（忽略 rowspan/colspan，保留全部行文本）。"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return ""
        lines = []
        sep_added = False
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells:
                continue
            lines.append("| " + " | ".join(cells) + " |")
            if not sep_added:
                lines.append("|" + " --- |" * len(cells))
                sep_added = True
        return "\n".join(lines)
    except Exception:
        # 兜底：用正则提取单元格文本
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
        lines = []
        for i, row in enumerate(rows):
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            if not cells:
                continue
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("|" + " --- |" * len(cells))
        return "\n".join(lines)


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
                caption = " ".join(block.get("table_caption", []))
                raw_text = block.get("text", "")
                html_body = block.get("table_body", "")
                # 仅当 HTML 行数明显多于 text 行数时才用 HTML（text 被截断的情况）
                # 否则保留原始 text（避免 HTML→MD 引入多余分隔行破坏续块识别）
                if html_body and _html_has_more_rows(html_body, raw_text):
                    body = _html_table_to_markdown(html_body)
                else:
                    body = raw_text
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
