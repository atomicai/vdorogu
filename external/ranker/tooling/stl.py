from typing import Dict

import polars as pl


def _pl_map(mapping: Dict):
    cursor = pl.element()
    for pre, nex in mapping.items():
        cursor = cursor.str.replace_all(pre, str(nex), literal=True)
    return cursor


def _pl_contains(_df, col, words, cased: bool = False):
    # regex pattern to count for word(s)
    if not cased:
        words = ["(?i)" + w for w in words]
    pattern = "|".join(words)
    _df = _df.with_columns([pl.col(col).str.contains(pattern).alias("silo")])
    _cnt = int(str(list(_df.select(pl.sum("silo")).to_arrow()[0])[0]))
    return _df, _cnt


def _pl_project(db, plane, on: str, how: str = "inner"):
    projected = db.join(plane, on=on, how=how)
    return projected


def _pl_count(db, column_name: str = "description"):
    projected = (
        db.with_row_count()
        .with_columns(
            [
                pl.count("row_nr").over(column_name).alias(f"counts_per_{column_name}"),
                pl.first("row_nr").over(column_name).alias("mask"),
            ]
        )
        .filter(pl.col("mask") == pl.col("row_nr"))
        .sort(f"counts_per_{column_name}", descending=True)
    )
    return projected


def _pl_match(db, col_x, col_y):
    projected = db.select((pl.col(col_x) == pl.col(col_y)).alias("match"))
    return projected.select(pl.sum("match")).to_dict()["match"][0]


def _pl_unique(db, column_name):
    projected = (
        db.with_row_count()
        .with_columns([pl.first("row_nr").over(column_name).alias("mask")])
        .filter(pl.col("row_nr") == pl.col("mask"))
        .drop(["mask", "row_nr"])
    )
    return projected


def _pl_normalize(db, column_name: str):
    de = (
        db.with_columns([pl.col(column_name).str.to_lowercase().str.strip().alias(f"_{column_name}")])
        .drop(column_name)
        .rename({f"_{column_name}": f"{column_name}"})
    )
    return de


__all__ = ["_pl_map", "_pl_contains", "_pl_project", "_pl_count", "_pl_unique"]
