from __future__ import annotations

import pandas as pd

from scripts.data.concepts.build_concept_industry_mapping import build_mapping


def test_build_mapping_with_quality_tiers():
    concept_df = pd.DataFrame(
        [
            {"concept_code": "A.TI", "concept_name": "概念A", "ts_code": "000001.SZ"},
            {"concept_code": "A.TI", "concept_name": "概念A", "ts_code": "000002.SZ"},
            {"concept_code": "A.TI", "concept_name": "概念A", "ts_code": "000003.SZ"},
            {"concept_code": "B.TI", "concept_name": "概念B", "ts_code": "000004.SZ"},
            {"concept_code": "B.TI", "concept_name": "概念B", "ts_code": "000005.SZ"},
            {"concept_code": "C.TI", "concept_name": "概念C", "ts_code": "000006.SZ"},
        ]
    )
    stock_map = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry_name": "电子"},
            {"ts_code": "000002.SZ", "industry_name": "电子"},
            {"ts_code": "000003.SZ", "industry_name": "电子"},
            {"ts_code": "000004.SZ", "industry_name": "汽车"},
            {"ts_code": "000005.SZ", "industry_name": "电子"},
            {"ts_code": "000006.SZ", "industry_name": "煤炭"},
        ]
    )

    _, primary, unmapped, report = build_mapping(
        concept_df=concept_df,
        stock_map=stock_map,
        min_ratio=0.05,
        strict_ratio=0.6,
    )

    assert len(primary) == 3
    assert set(primary["mapping_quality"]) == {"high", "low"}
    assert len(unmapped) == 0
    assert report["coverage_rate"] == 1.0
    assert "mapping_quality_distribution" in report
