from typing import Iterable, List

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


def _pl_stopwords(
    db: pl.DataFrame,
    column_name: str,
    stopwords: Iterable[str],
    min_chars_per_word: int = 2,
    min_chars_per_sentence: int = 5,
):
    stopwords = set(stopwords)
    projected = (
        db.with_row_count()
        .with_columns(
            [
                pl.col(column_name)
                .str.split(" ")
                .arr.eval(
                    pl.when((~pl.element().is_in(stopwords)) & (pl.element().str.n_chars() > min_chars_per_word))
                    .then(pl.element())
                    .otherwise(pl.lit(""))
                )
                .alias(f"silo_{column_name}")
            ]
        )
        .with_columns(
            [
                (pl.col(f"silo_{column_name}").arr.join(" ").str.n_chars() > min_chars_per_sentence).alias(
                    f"_is_silo_{column_name}"
                )
            ]
        )
    )
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


def _pl_contains(_df, col, words: List[str], cased: bool = False):
    # regex pattern to count for word(s)
    if not cased:
        words = ["(?i)" + w for w in words]
    pattern = "|".join(words)
    _df = _df.with_columns([pl.col(col).str.contains(pattern).alias(f"silo_{col}")])
    _cnt = int(str(list(_df.select(pl.sum(f"silo_{col}")).to_arrow()[0])[0]))
    return _df, _cnt


__all__ = [
    "_pl_project",
    "_pl_count",
    "_pl_as_dict",
    "_pl_stopwords",
    "_pl_normalize",
    "_pl_match",
    "_pl_unique",
    "_pl_contains",
]
