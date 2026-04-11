"""
批量入库脚本：扫描 data/raw/ 下所有 PDF，入库到指定 collection

特性：
  - 断点续传：读取 Chroma 已有文件记录，跳过已处理文件
  - 子目录递归扫描
  - 每个文件在独立子进程处理（CUDA context 隔离，单个文件崩溃不影响后续）
  - 实时进度显示

用法：
  conda activate mineru_2.5
  cd ~/rag_project

  # 预览（不实际入库）
  python scripts/batch_ingest_raw.py --dry-run

  # 入库到 gb_standards（默认）
  python scripts/batch_ingest_raw.py

  # 指定 collection 和目录
  python scripts/batch_ingest_raw.py --dir data/raw --collection my_collection
"""

import sys
import json
import argparse
import time
import subprocess
from pathlib import Path

sys.path.insert(0, "/home/hanyuu/rag_project")

CHECKPOINT_FILE = "data/ingest_checkpoint.json"


def load_checkpoint(path: str = CHECKPOINT_FILE) -> set[str]:
    """读取已处理文件列表（文件名集合）"""
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text())
        return set(data.get("done", []))
    return set()


def save_checkpoint(done: set[str], path: str = CHECKPOINT_FILE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"done": sorted(done)}, ensure_ascii=False, indent=2)
    )


def find_pdfs(root: Path, prefix: str = None) -> list[Path]:
    """递归查找所有 PDF，排除 mineru_output 目录。prefix 指定文件名前缀过滤。"""
    pdfs = []
    for p in sorted(root.rglob("*.pdf")):
        if "mineru_output" in str(p):
            continue
        if prefix and not p.name.startswith(prefix):
            continue
        pdfs.append(p)
    return pdfs


def run(scan_dir: str, collection: str, dry_run: bool, use_vlm: bool, prefix: str = None, checkpoint: str = CHECKPOINT_FILE, limit: int = None):
    root = Path(scan_dir)
    if not root.exists():
        print(f"目录不存在：{root}")
        return

    all_pdfs = find_pdfs(root, prefix=prefix)
    done = load_checkpoint(checkpoint)

    todo = [p for p in all_pdfs if p.name not in done]
    if limit:
        todo = todo[:limit]

    print(f"扫描目录：{root.resolve()}")
    print(f"发现 PDF：{len(all_pdfs)} 个")
    print(f"已处理：{len(done)} 个（跳过）")
    print(f"待处理：{len(todo)} 个")
    print(f"目标库：{collection}")
    print(f"VLM 补表：{'开启' if use_vlm else '关闭'}")

    if dry_run:
        print("\n[dry-run] 以下文件将被处理：")
        for p in todo:
            print(f"  {p.name}")
        return

    if not todo:
        print("\n全部已处理，无需入库。")
        return

    failed = []
    for idx, pdf_path in enumerate(todo):
        print(f"\n[{idx+1}/{len(todo)}] {pdf_path.name}")
        t0 = time.time()
        # 每个文件用独立子进程，防止 CUDA context 污染后续文件
        cmd = [
            sys.executable, __file__,
            "--_single-file", str(pdf_path.resolve()),
            "--collection", collection,
        ]
        if not use_vlm:
            cmd.append("--no-vlm")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=False,   # 子进程日志直接输出到终端
                timeout=600,            # 单个文件最多 10 分钟
                cwd="/home/hanyuu/rag_project",
            )
            elapsed = time.time() - t0
            if proc.returncode == 0:
                print(f"  ✅ 完成  {elapsed:.0f}s")
                done.add(pdf_path.name)
                save_checkpoint(done, checkpoint)
            else:
                print(f"  ❌ 子进程退出码 {proc.returncode}  {elapsed:.0f}s")
                failed.append((pdf_path.name, f"exit code {proc.returncode}"))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  ❌ 超时（{elapsed:.0f}s > 600s），跳过")
            failed.append((pdf_path.name, "timeout"))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌ 失败 ({elapsed:.0f}s)：{e}")
            failed.append((pdf_path.name, str(e)))

    print(f"\n{'='*50}")
    print(f"完成：{len(todo) - len(failed)}/{len(todo)} 个文件")
    if failed:
        print(f"失败 {len(failed)} 个：")
        for name, err in failed:
            print(f"  {name}: {err}")

    # 最终 chunk 统计
    try:
        import chromadb, os
        client = chromadb.PersistentClient(
            path=os.path.expanduser("~/rag_project/data/chroma_db")
        )
        col = client.get_collection(collection)
        print(f"\n{collection} 当前总 chunks：{col.count()}")
    except Exception:
        pass


def run_single_file(pdf_path: str, collection: str, use_vlm: bool):
    """子进程入口：处理单个 PDF 文件并退出（exit 0 成功，exit 1 失败）"""
    from backend.services.pipeline import process_file
    import traceback
    try:
        result = process_file(
            file_path=pdf_path,
            collection_name=collection,
            use_vlm_ocr=use_vlm,
        )
        chunks = result.get("chunks", "?")
        print(f"  → {chunks} chunks 入库完成")
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量入库 data/raw/ 下所有 PDF")
    parser.add_argument("--dir",          default="data/raw",     help="扫描目录（默认 data/raw）")
    parser.add_argument("--collection",   default="gb_standards", help="目标知识库（默认 gb_standards）")
    parser.add_argument("--dry-run",      action="store_true",    help="仅预览，不实际入库")
    parser.add_argument("--no-vlm",       action="store_true",    help="关闭 VLM 补表（加快速度）")
    parser.add_argument("--prefix",       default=None,           help="只处理文件名以此开头的 PDF（如 GB）")
    parser.add_argument("--checkpoint",   default=CHECKPOINT_FILE, help="断点续传文件路径（默认 data/ingest_checkpoint.json）")
    parser.add_argument("--limit",        type=int, default=None,  help="最多处理几个文件（默认不限制）")
    # 内部参数：由批量模式的子进程调用，不对用户暴露
    parser.add_argument("--_single-file", dest="single_file", default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_file:
        # 子进程模式：只处理一个文件
        run_single_file(args.single_file, args.collection, not args.no_vlm)
    else:
        run(args.dir, args.collection, args.dry_run, not args.no_vlm, args.prefix, args.checkpoint, args.limit)
