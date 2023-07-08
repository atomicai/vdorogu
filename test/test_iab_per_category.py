import unittest
from pathlib import Path

import networkx as nx
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pygraphviz as pgv
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph


def _pl_project(db, plane, on: str, how: str = "inner"):
    projected = db.join(plane, on=on, how=how)
    return projected


def _pl_count(db, column_name: str = "description", title="Y"):
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


def _pl_as_dict(db, col_key, col_value):
    keys = list(db[col_key])
    assert len(set(keys)) == len(keys), f"The columns {col_key} contains non unique keys."
    values = list(db[col_value])
    return {k: v for k, v in zip(keys, values)}


def _pl_normalize(db, column_name: str):
    de = (
        db.with_columns([pl.col(column_name).str.to_lowercase().str.strip().alias(f"_{column_name}")])
        .drop(column_name)
        .rename({f"_{column_name}": f"{column_name}"})
    )
    return de


def _pl_match(db, col_x, col_y):
    projected = db.select((pl.col(col_x) == pl.col(col_y)).alias("match"))
    return projected.select(pl.sum("match")).to_dict()["match"][0]


def _pl_unique(db, column_name):
    projected = (
        db.with_row_count()
        .with_columns([pl.first("row_nr").over(column_name).alias("mask")])
        .filter(pl.col("row_nr") == pl.col("mask"))
        .drop("mask")
    )
    return projected


def view(xs, ys, using: str = "bar"):
    import plotly.express as px

    if using == "bar":
        fig = px.bar(x=xs, y=ys)
    else:
        fig = px.line(x=xs, y=ys)
    fig.update_layout(title=dict(text=title, font=dict(size=14), automargin=False, yref='paper'), xaxis={'type': 'category'})
    fig.update_layout(yaxis_title=None)
    fig.update_layout(xaxis_title=None)
    fig.update_xaxes(tickfont_size=9, ticks="outside", ticklen=0.5, tickwidth=1)
    fig.update_xaxes(tick)
    return fig


class ITest(unittest.TestCase):
    def setUp(self):
        self.where = Path.home() / "IDataset" / "mobapp" / "mobapp.parquet"
        self.where_ios = Path.home() / "IDataset" / "mobapp" / "ios_iab_category.csv"
        self.where_droid = Path.home() / "IDataset" / "mobapp" / "droid_iab_category.csv"
        self.where_iab = Path.home() / "IDataset" / "mobapp" / "iab_category.csv"

    def test_main(self):
        dp = pq.read_table(self.where)
        df = pl.from_arrow(dp)

        df_ios = pl.read_csv(self.where_ios)
        df_droid = pl.read_csv(self.where_droid)
        df_meta_iab = pl.read_csv(self.where_iab)
        # Проецируем на подпр-во `iab_category` мобилок ios/android - урезается вдвое.
        # ios - 24
        # android - 26
        df_meta_iab_ios = (
            df_meta_iab.rename({"id": "iab_category_id"})
            .select(["iab_category_id", "parent_id", "name", "description"])
            .join(df_ios, on="iab_category_id")
            .select(["iab_category_id", "category_id", "parent_id", "name", "description"])
            .rename({"name": "iab_category_name"})
        )
        df_meta_iab_droid = (
            df_meta_iab.rename({"id": "iab_category_id"})
            .select(["iab_category_id", "parent_id", "name", "description"])
            .join(df_droid, on="iab_category_id")
            .select(["iab_category_id", "category_id", "parent_id", "name", "description"])
            .rename({"name": "iab_category_name"})
        )
        assert (
            len(set(df_meta_iab_ios["category_id"]) & set(df_meta_iab_droid["category_id"])) == 0
        ), "None empty intersection can cause problem(s)"
        # <->|<->|<->
        db_droid = _pl_project(df, df_meta_iab_droid, on="category_id")
        db_ios = _pl_project(df, df_meta_iab_ios, on="category_id")
        # Препроцессинг
        de = _pl_normalize(df, column_name="title")
        # Отдельно по источнику / магазину
        de_droid = (
            de.filter(pl.col("source") == "google").select(["title", "category_id"]).join(df_meta_iab_droid, on="category_id")
        )
        de_ios = de.filter(pl.col("source") == "apple").select(["title", "category_id"]).join(df_meta_iab_ios, on="category_id")
        print("IO is ok")
        # Посмотрим на кол-во общих приложений,
        # Кол-во совпадений категорий среди них и насколько все плохо
        db_proj = de_droid.join(de_ios, how="outer", on="title").with_columns(
            [
                (pl.col("iab_category_name").is_not_null() & pl.col("iab_category_name_right").is_not_null()).alias("common"),
                (pl.col("iab_category_name").is_not_null() & pl.col("iab_category_name_right").is_null()).alias("droid"),
                (pl.col("iab_category_name").is_null() & pl.col("iab_category_name_right").is_not_null()).alias("ios"),
            ]
        )
        # Сужаем на "приложения, которые одни и те же"
        db_proj_common = db_proj.filter(pl.col("common"))
        metrica = {}
        # сколько общих.
        metrica["all"] = len(de)
        metrica["common_title"] = db_proj_common.shape[0]
        metrica["common_title_iab"] = _pl_match(db_proj_common, col_x="iab_category_name", col_y="iab_category_name_right")
        metrica["variance"] = None
        # Считаем разброс по категориям (среди общих)
        print()


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
