"""
批量入库脚本

用法：
    python scripts/batch_ingest.py              # 入库全部文件
    python scripts/batch_ingest.py --dry-run    # 只预览，不实际入库
    python scripts/batch_ingest.py --dir chinese --collection energy_zh  # 只跑中文
    python scripts/batch_ingest.py --dir english --collection energy_en  # 只跑英文

特性：
    - 断点续传：跳过已有 mineru_output 的文件
    - 两个库都用 bge-m3（支持跨语言检索）
    - 进度实时打印
"""

import sys
import argparse
import time
import subprocess
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")

RAW_DIR = Path("/home/hanyuu/rag_project/data/raw")

TASKS = [
    {
        "dir":        RAW_DIR / "chinese",
        "collection": "energy_zh",
        "model":      "BAAI/bge-m3",
        "use_vlm":    False,
    },
    {
        "dir":        RAW_DIR / "english",
        "collection": "energy_en",
        "model":      "BAAI/bge-m3",
        "use_vlm":    False,
    },
]


def already_processed(pdf_path: Path) -> bool:
    """检查 MinerU 是否已处理过（有 content_list.json 则跳过）"""
    stem = pdf_path.stem
    mineru_out = pdf_path.parent / "mineru_output" / stem / "auto"
    content_list = mineru_out / f"{stem}_content_list.json"
    enhanced = mineru_out / "content_list_with_tables.json"
    return content_list.exists() or enhanced.exists()


def collect_pdfs(base_dir: Path):
    return sorted(p for p in base_dir.rglob("*.pdf") if "mineru_output" not in p.parts)


def run_task(task: dict, dry_run: bool):
    pdfs = collect_pdfs(task["dir"])
    collection = task["collection"]
    model = task["model"]
    use_vlm = task["use_vlm"]

    print(f"\n{'='*60}")
    print(f"目标库：{collection}  模型：{model}")
    print(f"文件数：{len(pdfs)}  来源目录：{task['dir']}")
    print(f"{'='*60}")

    skipped, done, failed = [], [], []

    for i, pdf in enumerate(pdfs, 1):
        rel = pdf.relative_to(RAW_DIR)
        prefix = f"[{i}/{len(pdfs)}]"

        if already_processed(pdf):
            print(f"{prefix} ⏭  跳过（已处理）：{rel}")
            skipped.append(pdf)
            continue

        if dry_run:
            print(f"{prefix} 🔍 预览：{rel}  →  {collection}")
            continue

        print(f"{prefix} ▶  开始：{rel}")
        t0 = time.time()

        # 每个文件用独立子进程处理，处理完自动释放 MinerU + Embedding 模型内存
        worker_script = f"""
import sys
sys.path.insert(0, "/home/hanyuu/rag_project")
from backend.services.pipeline import process_file

def cb(step, pct):
    print(f"      {{pct:3d}}%  {{step}}", flush=True)

result = process_file(
    file_path={str(pdf)!r},
    collection_name={collection!r},
    use_vlm_ocr={use_vlm!r},
    progress_callback=cb,
    embedding_model={model!r},
)
print(f"CHUNKS:{{result['chunks']}}")
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", worker_script],
                capture_output=False,
                text=True,
            )
            elapsed = time.time() - t0
            if proc.returncode == 0:
                print(f"      ✅ 完成  耗时 {elapsed:.0f}s")
                done.append(pdf)
            else:
                print(f"      ❌ 失败：子进程退出码 {proc.returncode}")
                failed.append((pdf, f"exit code {proc.returncode}"))
        except Exception as e:
            print(f"      ❌ 失败：{e}")
            failed.append((pdf, str(e)))

    return skipped, done, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只预览，不实际入库")
    parser.add_argument("--dir", choices=["chinese", "english"], help="只跑指定语言")
    parser.add_argument("--collection", help="指定 collection 名（配合 --dir 使用）")
    args = parser.parse_args()

    tasks = TASKS
    if args.dir:
        tasks = [t for t in TASKS if t["dir"].name == args.dir]
        if args.collection:
            tasks[0] = {**tasks[0], "collection": args.collection}

    total_skipped, total_done, total_failed = [], [], []

    for task in tasks:
        s, d, f = run_task(task, dry_run=args.dry_run)
        total_skipped += s
        total_done += d
        total_failed += f

    print(f"\n{'='*60}")
    print(f"汇总：完成 {len(total_done)} | 跳过 {len(total_skipped)} | 失败 {len(total_failed)}")
    if total_failed:
        print("失败文件：")
        for p, err in total_failed:
            print(f"  {p.name}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
