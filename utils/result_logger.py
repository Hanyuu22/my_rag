"""
结果日志工具

同时输出到终端和文件，测试结束后自动保存到 data/results/。

用法：
    from utils.result_logger import ResultLogger

    with ResultLogger("test_level2") as log:
        log("问题：xxx")
        log("答案：yyy")
        log.metrics({"检索耗时": 0.3, "top1_score": 0.95})
"""

import os
import sys
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path("/home/hanyuu/rag_project/data/results")


class ResultLogger:
    """
    上下文管理器，自动把所有 log() 输出写到文件 + 打印到终端。

    文件名格式：{name}_{YYYYMMDD_HHMMSS}.log
    """

    def __init__(self, name: str):
        self.name = name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = RESULTS_DIR / f"{name}_{timestamp}.log"
        self._file = None

    def __enter__(self):
        self._file = open(self.log_path, "w", encoding="utf-8")
        header = (
            f"{'='*60}\n"
            f"测试：{self.name}\n"
            f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*60}\n"
        )
        self._write(header)
        return self

    def __exit__(self, *_):
        footer = f"\n{'='*60}\n日志已保存到：{self.log_path}\n"
        self._write(footer)
        if self._file:
            self._file.close()

    def _write(self, text: str):
        print(text, end="")
        if self._file:
            self._file.write(text)
            self._file.flush()

    def __call__(self, text: str = ""):
        """log("任意文本")"""
        self._write(text + "\n")

    def metrics(self, data: dict, title: str = "📊 指标"):
        """格式化输出指标字典"""
        lines = [f"\n{title}"]
        for k, v in data.items():
            if isinstance(v, float):
                lines.append(f"  {k:<18} {v:.3f}")
            elif isinstance(v, list):
                lines.append(f"  {k:<18} {v}")
            else:
                lines.append(f"  {k:<18} {v}")
        self._write("\n".join(lines) + "\n")

    def divider(self, char: str = "─", width: int = 60):
        self._write(char * width + "\n")

    def section(self, title: str):
        self._write(f"\n{'─'*60}\n{title}\n{'─'*60}\n")
