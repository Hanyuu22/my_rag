"""
统一文件处理流水线

支持 PDF / Word (.doc/.docx) / Excel (.xlsx/.xls)
通过 progress_callback 实时汇报每个步骤的进度百分比。
"""

import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, "/home/hanyuu/rag_project")


# 进度步骤定义（每种格式的步骤权重）
PDF_STEPS = [
    ("文件接收完成", 5),
    ("MinerU 文档解析", 60),
    ("文本切分", 75),
    ("向量化入库", 100),
]

WORD_STEPS = [
    ("文件接收完成", 5),
    ("Word → PDF 转换", 20),
    ("MinerU 文档解析", 60),
    ("表格 OCR 识别", 78),
    ("文本切分", 85),
    ("向量化入库", 100),
]

EXCEL_STEPS = [
    ("文件接收完成", 5),
    ("Excel 数据解析", 60),
    ("向量化入库", 100),
]


def process_file(
    file_path: str,
    collection_name: str,
    use_vlm_ocr: bool = True,
    progress_callback: Callable[[str, int], None] = None,
    embedding_model: str = None,
) -> dict:
    """
    统一处理入口，根据文件类型自动分发。

    Args:
        file_path: 上传文件的本地路径
        collection_name: 存入 Chroma 的 collection 名称
        use_vlm_ocr: 是否对空表格启用 VLM OCR（仅 PDF/Word 有效）
        progress_callback: (step_name, percent) 回调

    Returns:
        {"chunks": int, "collection": str}
    """
    def report(step: str, pct: int):
        if progress_callback:
            progress_callback(step, pct)

    path = Path(file_path)
    suffix = path.suffix.lower()
    # collection 名以 _en 结尾时自动设英文 OCR，否则中文
    lang = "en" if collection_name.endswith("_en") else "ch"

    if suffix in (".pdf",):
        return _process_pdf(path, collection_name, use_vlm_ocr, report, lang, embedding_model)
    elif suffix in (".doc", ".docx"):
        return _process_word(path, collection_name, use_vlm_ocr, report, lang, embedding_model)
    elif suffix in (".xlsx", ".xls"):
        return _process_excel(path, collection_name, report, embedding_model)
    else:
        raise ValueError(f"不支持的文件格式：{suffix}，支持 PDF / Word / Excel")


def _process_pdf(path: Path, collection_name: str, use_vlm_ocr: bool, report, lang: str = "ch", embedding_model: str = None):
    from loaders.mineru_loader import MinerULoader
    from splitters.markdown_splitter import split_documents
    from chains.rag_chain import build_vectorstore
    from config import CHROMA_COLLECTION_NAME

    # Step 1: MinerU 解析
    report("MinerU 文档解析中...", 10)
    import sys as _sys
    _MINERU_ROOT = "/home/hanyuu/MinerU-master"
    if _MINERU_ROOT not in _sys.path:
        _sys.path.insert(0, _MINERU_ROOT)
    from demo.demo import do_parse, read_fn
    mineru_out = path.parent / "mineru_output"
    mineru_out.mkdir(parents=True, exist_ok=True)
    # energy 类知识库（energy_en/energy_zh）无数学公式，禁用 MFD+MFR 避免图表误判
    has_formula = not collection_name.startswith("energy")
    do_parse(
        output_dir=str(mineru_out),
        pdf_file_names=[path.stem],
        pdf_bytes_list=[read_fn(path)],
        p_lang_list=[lang],
        backend="pipeline",
        parse_method="auto",
        formula_enable=has_formula,
    )
    stem = path.stem
    content_list_path = mineru_out / stem / "auto" / f"{stem}_content_list.json"
    if not content_list_path.exists():
        raise FileNotFoundError(f"MinerU 未生成 content_list: {content_list_path}")
    report("MinerU 解析完成", 60)

    # Step 2: VLM OCR（可选）
    enhanced_path = content_list_path.parent / "content_list_with_tables.json"
    if enhanced_path.exists():
        content_list_path = enhanced_path
        report("复用已有表格 OCR 缓存", 75)
    elif use_vlm_ocr:
        report("表格 OCR 识别中...", 62)
        from tools.ocr_empty_tables import run_ocr
        run_ocr(content_list_path, enhanced_path)
        if enhanced_path.exists():
            content_list_path = enhanced_path
        report("表格 OCR 完成", 75)

    # Step 3: 切分
    report("文本切分中...", 78)
    docs = MinerULoader(str(content_list_path), source_name=path.stem).load()
    chunks = split_documents(docs)
    report("文本切分完成", 82)

    # Step 4: 入库（追加模式，多文件安全）
    report("向量化入库中...", 85)
    build_vectorstore(chunks, collection_name=collection_name, force_rebuild=True, embedding_model=embedding_model)

    # Step 4b: 同步写 ES（ES 不可用时静默跳过）
    try:
        from retrievers.es_retriever import index_chunks_to_es, es_available
        if es_available():
            index_chunks_to_es(chunks, collection_name)
    except Exception:
        pass

    report("向量化入库完成", 100)

    return {"chunks": len(chunks), "collection": collection_name}


def _process_word(path: Path, collection_name: str, use_vlm_ocr: bool, report, lang: str = "ch", embedding_model: str = None):
    from loaders.word_loader import doc_to_pdf, run_mineru
    from loaders.mineru_loader import MinerULoader
    from splitters.markdown_splitter import split_documents
    from chains.rag_chain import build_vectorstore
    from config import CHROMA_COLLECTION_NAME

    # Step 1: Word → PDF
    report("Word → PDF 转换中...", 8)
    pdf_path = path.with_suffix(".pdf")
    if not pdf_path.exists():
        doc_to_pdf(str(path), str(pdf_path))
    report("PDF 转换完成", 20)

    # Step 2: MinerU 解析
    report("MinerU 文档解析中...", 22)
    mineru_out = path.parent / "mineru_output"
    mineru_out.mkdir(parents=True, exist_ok=True)
    content_list_path_str = run_mineru(str(pdf_path), str(mineru_out))
    from pathlib import Path as P
    content_list_path = P(content_list_path_str)
    report("MinerU 解析完成", 60)

    # Step 3: VLM OCR（可选）
    enhanced_path = content_list_path.parent / "content_list_with_tables.json"
    if enhanced_path.exists():
        content_list_path = enhanced_path
        report("复用已有表格 OCR 缓存", 75)
    elif use_vlm_ocr:
        report("表格 OCR 识别中...", 62)
        from tools.ocr_empty_tables import run_ocr
        run_ocr(content_list_path, enhanced_path)
        if enhanced_path.exists():
            content_list_path = enhanced_path
        report("表格 OCR 完成", 78)

    # Step 4: 切分
    report("文本切分中...", 80)
    docs = MinerULoader(str(content_list_path), source_name=path.stem).load()
    chunks = split_documents(docs)
    report("文本切分完成", 85)

    # Step 5: 入库（追加模式，多文件安全）
    report("向量化入库中...", 88)
    build_vectorstore(chunks, collection_name=collection_name, force_rebuild=True, embedding_model=embedding_model)

    try:
        from retrievers.es_retriever import index_chunks_to_es, es_available
        if es_available():
            index_chunks_to_es(chunks, collection_name)
    except Exception:
        pass

    report("向量化入库完成", 100)

    return {"chunks": len(chunks), "collection": collection_name}


def _process_excel(path: Path, collection_name: str, report, embedding_model: str = None):
    from loaders.excel_loader import ExcelLoader
    from chains.rag_chain import build_vectorstore

    # Step 1: Excel 解析
    report("Excel 数据解析中...", 10)
    docs = ExcelLoader(str(path)).load()
    report("Excel 解析完成", 60)

    # Step 2: 入库（Excel 按行，不二次切分，追加模式）
    report("向量化入库中...", 65)
    build_vectorstore(docs, collection_name=collection_name, force_rebuild=True, embedding_model=embedding_model)
    report("向量化入库完成", 100)

    return {"chunks": len(docs), "collection": collection_name}
