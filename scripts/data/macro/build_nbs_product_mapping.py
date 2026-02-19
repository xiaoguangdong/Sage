#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
三次映射：工业品产量目录 → GB/T 4754-2017 行业 → 申万一级

依赖：
1) 主要工业产品产量目录（含说明/行业字段，建议放在 refs_docs/）
2) 2017年国民经济行业分类注释（网络版）.xlsx
3) 申万行业分类标准 / 上市公司行业统计分类（用于口径校对）

输出：
config/nbs_product_sw_mapping.yaml
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def find_first(patterns: List[str], root: Path) -> Optional[Path]:
    for pattern in patterns:
        for path in root.glob(pattern):
            return path
    return None


def load_gbt_notes(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df["__sheet__"] = sheet
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_gbt_catalog(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    code_pattern = re.compile(r"^[A-Z]$|^\\d{2,4}$")
    for _, row in df.iterrows():
        values = [v for v in row.tolist() if isinstance(v, str) and v.strip()]
        if not values:
            continue
        code = None
        for value in values:
            text = value.strip()
            if code_pattern.match(text):
                code = text
                break
        if not code:
            continue
        name = None
        for value in reversed(values):
            text = value.strip()
            if text in {"◇", "—"}:
                continue
            if code_pattern.match(text):
                continue
            name = text
            break
        if not name:
            continue
        rows.append({"gb_code": code, "gb_name": name})
    return pd.DataFrame(rows).drop_duplicates()


def load_sw_mapping(path: Path) -> Dict[str, str]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sw_to_nbs = data.get("sw_to_nbs") or {}
    mapping: Dict[str, str] = {}
    weights: Dict[str, float] = {}
    for sw_name, items in sw_to_nbs.items():
        if not items:
            continue
        for item in items:
            nbs_name = item.get("nbs_industry")
            weight = float(item.get("weight", 0) or 0)
            if not nbs_name:
                continue
            if nbs_name not in mapping or weight > weights.get(nbs_name, -1):
                mapping[nbs_name] = sw_name
                weights[nbs_name] = weight
    return mapping


def load_gb_keyword_rules(path: Path) -> List[Tuple[List[str], str]]:
    if not path.exists():
        return []
    try:
        import yaml  # type: ignore
    except Exception:
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = []
    for item in data.get("rules", []) or []:
        industry = (item or {}).get("gb_industry")
        keywords = (item or {}).get("keywords") or []
        keywords = [str(k).strip() for k in keywords if str(k).strip()]
        if industry and keywords:
            rules.append((keywords, str(industry).strip()))
    return rules


def normalize_text(text: str) -> str:
    return text.replace("（", "(").replace("）", ")").replace(" ", "").replace("\n", "").strip()


def build_gb_name_index(gb_df: pd.DataFrame) -> List[str]:
    names = []
    for name in gb_df["gb_name"].dropna().unique():
        names.append(str(name))
    names = sorted(set(names), key=len, reverse=True)
    return names


def match_gb_industry(text: str, gb_names: List[str]) -> Optional[str]:
    if not text:
        return None
    raw = normalize_text(text)
    for name in gb_names:
        if normalize_text(name) in raw:
            return name
    return None


def match_gb_by_keywords(text: str, rules: List[Tuple[List[str], str]]) -> Optional[str]:
    if not text:
        return None
    raw = normalize_text(text)
    for keywords, industry in rules:
        if any(normalize_text(k) in raw for k in keywords):
            return industry
    return None


def has_chinese(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def convert_doc_to_txt(path: Path) -> Path:
    if path.suffix.lower() == ".txt":
        return path
    out_dir = Path(tempfile.mkdtemp(prefix="nbs_product_"))
    out_path = out_dir / f"{path.stem}.txt"
    subprocess.run(
        ["textutil", "-convert", "txt", str(path), "-output", str(out_path)],
        check=True,
    )
    return out_path


def parse_product_txt(path: Path) -> pd.DataFrame:
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    raw_lines = [line.strip() for line in content if line.strip()]
    lines: List[str] = []
    for line in raw_lines:
        tokens = [t for t in re.split(r"\s+", line) if t]
        if len(tokens) >= 2 and all(t.isdigit() and len(t) >= 5 for t in tokens):
            lines.extend(tokens)
        else:
            lines.append(line)

    code_pattern = re.compile(r"^\d{5,}$")
    records = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not code_pattern.match(line):
            i += 1
            continue
        code = line
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        name = lines[i].strip()
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        unit = lines[i].strip()
        i += 1

        desc_lines = []
        while i < len(lines) and not code_pattern.match(lines[i].strip()):
            text = lines[i].strip()
            if text and text != "　":
                desc_lines.append(text)
            i += 1
        desc = " ".join(desc_lines).strip()

        records.append(
            {
                "product_code": code,
                "product_name": name,
                "unit": unit,
                "description": desc,
            }
        )
    return pd.DataFrame(records)


def load_product_catalog(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".doc", ".docx", ".txt"}:
        txt_path = convert_doc_to_txt(path)
        return parse_product_txt(txt_path)
    raise ValueError(f"Unsupported product catalog file: {path}")


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    cols = list(df.columns)
    name_candidates = ["产品", "产品名称", "product", "product_name", "指标名称", "名称"]
    code_candidates = ["产品代码", "product_code", "指标代码", "代码"]
    industry_candidates = ["行业", "所属行业", "行业名称", "行业大类"]
    desc_candidates = ["说明", "补充说明", "指标解释", "主要生产活动", "产品说明", "description"]

    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    return pick(name_candidates), pick(code_candidates), pick(industry_candidates), pick(desc_candidates)


def main():
    parser = argparse.ArgumentParser(description="构建工业品产量目录 → 行业 → 申万映射")
    parser.add_argument("--product-file", default=None, help="工业品产量目录文件路径")
    parser.add_argument("--gbt-notes", default=None, help="GB/T 4754-2017 注释（xlsx）路径")
    parser.add_argument("--sw-map", default="config/sw_nbs_mapping.yaml", help="申万-国统局映射")
    parser.add_argument("--output", default="config/nbs_product_sw_mapping.yaml")
    parser.add_argument("--unmatched-output", default="data/processed/nbs_product_sw_unmatched.csv")
    parser.add_argument("--gb-keywords", default="config/nbs_product_gb_keywords.yaml")
    args = parser.parse_args()

    refs_dir = PROJECT_ROOT / "refs_docs"

    product_file = (
        Path(args.product_file)
        if args.product_file
        else find_first(["*工业产品产量目录*.xlsx", "*工业产品产量目录*.xls", "*工业产品产量目录*.csv"], refs_dir)
    )
    if not product_file or not product_file.exists():
        raise SystemExit("未找到工业产品产量目录文件，请放到 refs_docs/ 或使用 --product-file 指定。")

    gbt_notes = (
        Path(args.gbt_notes)
        if args.gbt_notes
        else find_first(["*国民经济行业分类注释*xlsx", "*行业分类注释*xlsx"], refs_dir)
    )
    if not gbt_notes or not gbt_notes.exists():
        raise SystemExit("未找到 GB/T 4754-2017 行业分类注释（xlsx）。")

    gb_raw = load_gbt_notes(gbt_notes)
    gb_catalog = extract_gbt_catalog(gb_raw)
    gb_names = build_gb_name_index(gb_catalog)

    product_df = load_product_catalog(product_file)
    name_col, code_col, industry_col, desc_col = detect_columns(product_df)
    if not name_col:
        raise SystemExit(f"产品目录缺少产品名称列，当前列={list(product_df.columns)}")

    sw_map = load_sw_mapping(PROJECT_ROOT / args.sw_map)
    gb_keyword_rules = load_gb_keyword_rules(PROJECT_ROOT / args.gb_keywords)

    results = []
    for _, row in product_df.iterrows():
        product_name = row.get(name_col)
        if pd.isna(product_name):
            continue
        product_name = str(product_name)
        product_code = row.get(code_col) if code_col else None
        industry_text = row.get(industry_col) if industry_col else None
        desc_text = row.get(desc_col) if desc_col else None

        gb_name = None
        source = None
        if industry_text and isinstance(industry_text, str):
            gb_name = match_gb_industry(industry_text, gb_names)
            source = "industry_field" if gb_name else None
        if not gb_name and desc_text and isinstance(desc_text, str):
            gb_name = match_gb_industry(desc_text, gb_names)
            source = "desc_match" if gb_name else None
        if not gb_name:
            gb_name = match_gb_industry(product_name, gb_names)
            source = "name_match" if gb_name else None
        if not gb_name:
            combined = " ".join([s for s in [product_name, desc_text] if isinstance(s, str)])
            gb_name = match_gb_by_keywords(combined, gb_keyword_rules)
            source = "keyword_rule" if gb_name else None

        gb_code = None
        if gb_name:
            match = gb_catalog[gb_catalog["gb_name"] == gb_name]
            if not match.empty:
                gb_code = match.iloc[0]["gb_code"]

        sw_industry = None
        if gb_name and gb_name in sw_map:
            sw_industry = sw_map[gb_name]

        results.append(
            {
                "product_name": product_name,
                "product_code": None if pd.isna(product_code) else str(product_code),
                "gb_name": gb_name,
                "gb_code": gb_code,
                "sw_industry": sw_industry,
                "map_source": source,
            }
        )

    result_df = pd.DataFrame(results)

    # 仅保留包含中文的产品名称
    result_df = result_df[result_df["product_name"].apply(has_chinese)].copy()

    # 按 product_name 去重，优先保留映射更完整的记录
    source_rank = {
        "industry_field": 4,
        "desc_match": 3,
        "name_match": 2,
        "keyword_rule": 1,
    }
    result_df["has_sw"] = result_df["sw_industry"].notna().astype(int)
    result_df["has_gb"] = result_df["gb_name"].notna().astype(int)
    result_df["source_rank"] = result_df["map_source"].map(source_rank).fillna(0).astype(int)
    result_df = (
        result_df.sort_values(["has_sw", "has_gb", "source_rank"], ascending=False)
        .drop_duplicates(subset=["product_name"], keep="first")
        .drop(columns=["has_sw", "has_gb", "source_rank"])
    )

    try:
        import yaml  # type: ignore
    except Exception:
        raise SystemExit("缺少 PyYAML，请先安装后再运行。")

    products = []

    def _clean(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return str(value)

    for _, row in result_df.iterrows():
        products.append(
            {
                "product_code": _clean(row.get("product_code")),
                "product_name": _clean(row.get("product_name")),
                "gb_industry": _clean(row.get("gb_name")),
                "gb_code": _clean(row.get("gb_code")),
                "sw_industry": _clean(row.get("sw_industry")),
                "map_source": _clean(row.get("map_source")),
            }
        )

    output = {
        "version": 1,
        "source": "工业品产量目录 + GB/T 4754-2017 注释 + 申万映射",
        "products": products,
        "keywords": [],
    }

    out_path = PROJECT_ROOT / args.output
    out_path.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False), encoding="utf-8")

    total = len(result_df)
    mapped = result_df["sw_industry"].notna().sum()
    print(f"映射完成: {mapped}/{total} 有申万行业映射")
    print(f"输出: {out_path}")

    unmatched = result_df[result_df["sw_industry"].isna()]
    if not unmatched.empty and args.unmatched_output:
        unmatched_path = PROJECT_ROOT / args.unmatched_output
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
        print(f"未匹配清单: {unmatched_path}")


if __name__ == "__main__":
    main()
