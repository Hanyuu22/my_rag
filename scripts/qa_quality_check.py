"""
QA 数据集质量检查脚本

检查项目：
1. 字段完整性（缺失字段）
2. LaTeX 残留（source_text / ground_truth 含 $ \\ \\mathrm 等）
3. PDF 解析异常（"错误!未定义书签"等）
4. 重复或高度相似的问题
5. 过短的 ground_truth（可能无意义）
6. negative 类型但答案不像 是/否 的条目
7. source_text 和 ground_truth 不对应的嫌疑（答案在 source 中找不到）

用法：
    python scripts/qa_quality_check.py \
        --input data/qa_dataset/hydro_reg_qa.json \
        --output data/qa_dataset/hydro_reg_qa_report.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ── 检查规则 ────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = ["question", "ground_truth", "source_text",
                   "question_type", "difficulty", "page"]

VALID_TYPES       = {"factual", "reasoning", "negative", "comparison"}
VALID_DIFFICULTIES = {"single_hop", "multi_hop"}

LATEX_PATTERN   = re.compile(r'(\$|\\mathrm|\\times|\\sim|\\circ|\\leq|\\geq|\\frac|\\sqrt|\\\w{2,})')
ARTIFACT_PATTERN = re.compile(r'错误!未定义书签|\.{5,}|\.{3,}\s+\d+')  # TOC 残留

NEGATIVE_STARTS = re.compile(r'^(否|不|不可以|不允许|不适用|不符合|不能|不满足|非|没有|无|是)')


def check_fields(item, idx):
    issues = []
    for f in REQUIRED_FIELDS:
        if not item.get(f) and item.get(f) != 0:
            issues.append(f"缺失字段: {f}")
    qt = item.get("question_type", "")
    if qt and qt not in VALID_TYPES:
        issues.append(f"无效 question_type: {qt}")
    diff = item.get("difficulty", "")
    if diff and diff not in VALID_DIFFICULTIES:
        issues.append(f"无效 difficulty: {diff}")
    return issues


def check_latex(item):
    issues = []
    for field in ("source_text", "ground_truth", "question"):
        text = item.get(field, "")
        if LATEX_PATTERN.search(text):
            snippet = text[:80].replace("\n", " ")
            issues.append(f"LaTeX残留({field}): {snippet}")
    return issues


def check_artifacts(item):
    issues = []
    for field in ("source_text", "question", "ground_truth"):
        text = item.get(field, "")
        if ARTIFACT_PATTERN.search(text):
            snippet = text[:80].replace("\n", " ")
            issues.append(f"PDF解析异常({field}): {snippet}")
    return issues


def check_length(item):
    issues = []
    gt = item.get("ground_truth", "")
    q  = item.get("question", "")
    # negative 类型允许 "是/否" 这类单字答案
    is_negative = item.get("question_type") == "negative"
    if not is_negative and len(gt.strip()) < 2:
        issues.append(f"ground_truth 过短: '{gt}'")
    elif is_negative and len(gt.strip()) == 0:
        issues.append(f"ground_truth 为空")
    if len(q.strip()) < 8:
        issues.append(f"question 过短: '{q}'")
    if len(q) > 200:
        issues.append(f"question 过长({len(q)}字): {q[:60]}...")
    return issues


def check_negative_consistency(item):
    issues = []
    if item.get("question_type") == "negative":
        gt = item.get("ground_truth", "").strip()
        if not NEGATIVE_STARTS.search(gt):
            issues.append(f"negative题型但答案不像否定: '{gt[:60]}'")
    return issues


def check_answer_in_source(item):
    """
    检查 ground_truth 的核心词汇是否与 source_text 有重叠。
    生成式 QA 的答案是对原文的归纳，不会逐字出现，
    因此只做宽松的词汇重叠检查（任意连续4字），
    避免大量误报。
    """
    issues = []
    gt  = item.get("ground_truth", "").strip()
    src = item.get("source_text", "").strip()
    if not gt or not src:
        return issues
    # 跳过很短的答案（"否"/"是"/"不适用"等）
    if len(gt) <= 6:
        return issues
    # 跳过 reasoning 类型（答案是推断结论，不在原文中是正常的）
    if item.get("question_type") == "reasoning":
        return issues
    # 宽松检查：gt 中任意连续4字是否出现在 src 中
    found = any(gt[i:i+4] in src for i in range(0, len(gt) - 3, 4))
    if not found and len(src) > 20:
        issues.append(f"答案与source_text无重叠词汇: GT='{gt[:40]}'")
    return issues


def find_duplicates(items):
    """找出问题文本高度相似（完全一样或前30字相同）的条目"""
    seen = defaultdict(list)
    for i, item in enumerate(items):
        q = item.get("question", "").strip()
        key = q[:30]
        seen[key].append(i)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    return dups


# ── 主流程 ──────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str):
    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)

    print(f"加载 {len(items)} 条 QA\n")

    report = {
        "total": len(items),
        "flagged_count": 0,
        "issues_by_type": defaultdict(int),
        "flagged_items": [],
        "duplicate_groups": [],
        "summary": {},
    }

    # 逐条检查
    for idx, item in enumerate(items):
        all_issues = []
        all_issues += check_fields(item, idx)
        all_issues += check_latex(item)
        all_issues += check_artifacts(item)
        all_issues += check_length(item)
        all_issues += check_negative_consistency(item)
        all_issues += check_answer_in_source(item)

        if all_issues:
            report["flagged_count"] += 1
            for iss in all_issues:
                category = iss.split(":")[0].split("(")[0]
                report["issues_by_type"][category] += 1
            report["flagged_items"].append({
                "index": idx,
                "page": item.get("page"),
                "question": item.get("question", "")[:80],
                "question_type": item.get("question_type"),
                "issues": all_issues,
            })

    # 重复检查
    dups = find_duplicates(items)
    for key, indices in dups.items():
        group = {
            "prefix": key,
            "indices": indices,
            "questions": [items[i].get("question", "")[:100] for i in indices],
        }
        report["duplicate_groups"].append(group)
        report["issues_by_type"]["重复问题"] += len(indices) - 1

    # 统计摘要
    from collections import Counter
    report["summary"] = {
        "题型分布":     dict(Counter(i.get("question_type") for i in items)),
        "难度分布":     dict(Counter(i.get("difficulty")    for i in items)),
        "主题分布":     dict(Counter(i.get("topic_tag")     for i in items)),
        "问题标记率":   f"{report['flagged_count']}/{len(items)} ({100*report['flagged_count']/len(items):.1f}%)",
        "重复组数":     len(report["duplicate_groups"]),
    }
    report["issues_by_type"] = dict(report["issues_by_type"])

    # 输出
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 终端摘要
    print("=" * 60)
    print(f"总条数：{report['total']}")
    print(f"有问题条数：{report['flagged_count']}")
    print(f"重复问题组：{len(report['duplicate_groups'])}")
    print("\n问题类型分布：")
    for k, v in sorted(report["issues_by_type"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} 条")
    print(f"\n详细报告已保存：{output_path}")

    # 打印前5个有问题的条目
    print("\n── 前5个标记条目 ──")
    for entry in report["flagged_items"][:5]:
        print(f"\n[{entry['index']}] p.{entry['page']} [{entry['question_type']}]")
        print(f"  问题：{entry['question']}")
        for iss in entry["issues"]:
            print(f"  ⚠ {iss}")

    if report["duplicate_groups"]:
        print("\n── 重复问题示例 ──")
        for grp in report["duplicate_groups"][:3]:
            print(f"\n前缀：'{grp['prefix']}'")
            for i, q in zip(grp["indices"], grp["questions"]):
                print(f"  [{i}] {q}")


# ── 入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/qa_dataset/hydro_reg_qa.json")
    parser.add_argument("--output", default="data/qa_dataset/hydro_reg_qa_report.json")
    args = parser.parse_args()
    run(args.input, args.output)
