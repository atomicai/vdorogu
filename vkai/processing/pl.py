from typing import List

import polars as pl


def _pl_project(db: pl.DataFrame, plane: pl.DataFrame, on: str, how: str = "inner"):
    projected = db.join(plane, on=on, how=how)
    return projected


def _pl_count(db: pl.DataFrame, column_name: str = "description"):
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
    ).drop(["row_nr"])
    return projected


def _pl_as_dict(db: pl.DataFrame, col_key, col_value):
    keys = list(db[col_key])
    assert len(set(keys)) == len(keys), f"The columns {col_key} contains non unique keys."
    values = list(db[col_value])
    return {k: v for k, v in zip(keys, values)}


def _pl_normalize(db: pl.DataFrame, column_name: str):
    de = (
        db.with_columns([pl.col(column_name).str.to_lowercase().str.strip().alias(f"_{column_name}")])
        .drop(column_name)
        .rename({f"_{column_name}": f"{column_name}"})
    )
    return de


def _pl_match(db: pl.DataFrame, col_x, col_y):
    projected = db.select((pl.col(col_x) == pl.col(col_y)).alias("match"))
    return projected.select(pl.sum("match")).to_dict()["match"][0]


def _pl_unique(db: pl.DataFrame, column_name):
    projected = (
        db.with_row_count()
        .with_columns([pl.first("row_nr").over(column_name).alias("mask")])
        .filter(pl.col("row_nr") == pl.col("mask"))
        .drop("mask")
    )
    return projected


def _pl_contains(_df, col, words: List[str]):
    pattern = "|".join(words)
    _df = _df.with_columns([pl.col(col).str.contains(pattern).alias(f"silo_{col}")])
    return _df


__all__ = ["_pl_project", "_pl_count", "_pl_as_dict", "_pl_normalize", "_pl_match", "_pl_unique", "_pl_contains"]
