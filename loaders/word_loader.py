"""
Word Document Loader

将 .doc/.docx → PDF（Windows Word COM） → MinerU 解析 → LangChain Documents。
仅在 WSL 环境下有效（需要 Windows 侧安装了 Microsoft Word）。

用法：
    from loaders.word_loader import WordLoader
    docs = WordLoader("path/to/file.doc").load()
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# MinerU 在独立目录，加入搜索路径
_MINERU_ROOT = "/home/hanyuu/MinerU-master"
if _MINERU_ROOT not in sys.path:
    sys.path.insert(0, _MINERU_ROOT)


def _wsl_to_win_unc(path: str) -> str:
    r"""将 WSL 绝对路径转为 Windows UNC 路径（\\wsl.localhost\Ubuntu-22.04\...）"""
    p = Path(path).resolve()
    distro = os.environ.get("WSL_DISTRO_NAME", "Ubuntu-22.04")
    win_path = str(p).replace('/', '\\')
    return f"\\\\wsl.localhost\\{distro}{win_path}"


def doc_to_pdf(doc_path: str, output_pdf: str = None) -> str:
    """
    用 Windows Word COM 将 .doc/.docx 转为 PDF。

    Returns:
        生成的 PDF 文件路径（字符串）
    """
    doc_path = Path(doc_path).resolve()
    if output_pdf is None:
        output_pdf = doc_path.with_suffix(".pdf")
    else:
        output_pdf = Path(output_pdf).resolve()

    win_doc = _wsl_to_win_unc(str(doc_path))
    win_pdf = _wsl_to_win_unc(str(output_pdf))

    ps_script = f"""
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open('{win_doc}')
$doc.SaveAs([ref]'{win_pdf}', [ref]17)
$doc.Close()
$word.Quit()
Write-Host 'ok'
"""
    result = subprocess.run(
        ["powershell.exe", "-Command", ps_script],
        capture_output=True, text=True, errors="replace"
    )
    if "ok" not in result.stdout:
        raise RuntimeError(f"Word COM 转换失败:\n{result.stdout}\n{result.stderr}")

    return str(output_pdf)


def run_mineru(pdf_path: str, output_dir: str = None) -> str:
    """
    用 MinerU pipeline 解析 PDF，返回生成的 content_list.json 路径。
    """
    from demo.demo import parse_doc

    pdf_path = Path(pdf_path).resolve()
    if output_dir is None:
        output_dir = pdf_path.parent / "mineru_output"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parse_doc(
        path_list=[pdf_path],
        output_dir=str(output_dir),
        lang="ch",
        backend="pipeline",
        method="auto",
    )

    # MinerU 输出路径：output_dir/{stem}/auto/{stem}_content_list.json
    stem = pdf_path.stem
    content_list_path = output_dir / stem / "auto" / f"{stem}_content_list.json"
    if not content_list_path.exists():
        raise FileNotFoundError(f"MinerU 未生成 content_list: {content_list_path}")

    return str(content_list_path)


class WordLoader:
    """
    加载 .doc/.docx：Word COM → PDF → MinerU 解析 → LangChain Documents。

    Args:
        file_path: Word 文件路径
        source_name: 来源名称（用于 metadata），默认取文件名
        mineru_output_dir: MinerU 输出目录，默认 ~/rag_project/data/mineru_output
        reuse_pdf: 若 PDF 已存在则跳过转换（默认 True）
        reuse_mineru: 若 content_list.json 已存在则跳过 MinerU（默认 True）
        use_vlm_ocr: 自动对 MinerU 未提取到文字的表格调用 qwen-vl-plus 补充 OCR。
                     首次处理会调用 API（每张表 ~1~10s），结果缓存为
                     content_list_with_tables.json，后续加载直接复用不重复计费。
                     默认 False，处理大量文档时建议显式传 True。
    """

    def __init__(
        self,
        file_path: str,
        source_name: str = None,
        mineru_output_dir: str = None,
        reuse_pdf: bool = True,
        reuse_mineru: bool = True,
        use_vlm_ocr: bool = False,
    ):
        self.file_path = Path(file_path).resolve()
        self.source_name = source_name or self.file_path.stem
        self.mineru_output_dir = (
            Path(mineru_output_dir) if mineru_output_dir
            else Path("/home/hanyuu/rag_project/data/mineru_output")
        )
        self.reuse_pdf = reuse_pdf
        self.reuse_mineru = reuse_mineru
        self.use_vlm_ocr = use_vlm_ocr

    def load(self) -> List[Document]:
        from loaders.mineru_loader import MinerULoader

        # Step 1: .doc → PDF
        pdf_path = self.file_path.with_suffix(".pdf")
        if pdf_path.exists() and self.reuse_pdf:
            print(f"复用已有 PDF: {pdf_path.name}")
        else:
            print(f"转换 Word → PDF: {self.file_path.name}")
            doc_to_pdf(str(self.file_path), str(pdf_path))
            print(f"  PDF 生成完成: {pdf_path.name} ({pdf_path.stat().st_size // 1024}KB)")

        # Step 2: PDF → MinerU content_list.json
        stem = pdf_path.stem
        content_list_path = self.mineru_output_dir / stem / "auto" / f"{stem}_content_list.json"
        if content_list_path.exists() and self.reuse_mineru:
            print(f"复用已有 MinerU 解析: {content_list_path.name}")
        else:
            print(f"MinerU 解析 PDF（首次约需 1~3 分钟）...")
            content_list_path_str = run_mineru(str(pdf_path), str(self.mineru_output_dir))
            content_list_path = Path(content_list_path_str)
            print(f"  解析完成: {content_list_path.name}")

        # Step 3: VLM 补充表格 OCR（可选，结果永久缓存）
        enhanced_path = content_list_path.parent / "content_list_with_tables.json"
        if enhanced_path.exists():
            # 已有缓存，直接复用（无论 use_vlm_ocr 是否开启）
            print(f"使用 VLM 增强版解析（含表格 OCR）: {enhanced_path.name}")
            content_list_path = enhanced_path
        elif self.use_vlm_ocr:
            # 首次处理：调用 VLM API 补充空表格
            print("检测到空表格，启动 VLM OCR 补充...")
            from tools.ocr_empty_tables import run_ocr
            run_ocr(content_list_path, enhanced_path)
            if enhanced_path.exists():
                content_list_path = enhanced_path

        # Step 4: content_list.json → Documents
        return MinerULoader(str(content_list_path), source_name=self.source_name).load()
