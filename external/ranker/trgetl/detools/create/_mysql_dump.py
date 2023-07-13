import io
from pathlib import Path
from typing import Optional

import pandas as pd

from ...database import Clickhouse
from ...filesystem import DDL_PATH


def mysql_dump(
    origin_table_name: str,
    new_table_name: Optional[str] = None,
    dsn: str = "mysql_target_main_dev",
) -> None:
    origin_schema_name, origin_table_name = origin_table_name.split(".")
    if new_table_name is None:
        new_table_name = origin_table_name
    assert "." not in new_table_name

    mysql_engine = dict(
        dsn=dsn,
        schema_name=origin_schema_name,
        table_name=origin_table_name,
    )

    column_data = _get_column_data(mysql_engine)

    odbc_ddl = _get_odbc_ddl(column_data, mysql_engine)
    dump_ddl = _get_dump_ddl(column_data)

    odbc_path = DDL_PATH / "ch" / "foreign_table" / "mysql_odbc" / new_table_name
    dump_path = DDL_PATH / "ch" / "table" / "dump" / new_table_name

    _create_table(path=odbc_path, ddl=odbc_ddl)
    _create_table(path=dump_path, ddl=dump_ddl)


def _get_column_data(mysql_engine: dict) -> pd.DataFrame:
    column_data = _read_column_data(mysql_engine)
    comments_data = _read_table_comments(
        schema_name=mysql_engine["schema_name"],
        table_name=mysql_engine["table_name"],
    )
    column_data = column_data.merge(
        comments_data,
        how="left",
        on="colname",
    )
    return column_data


def _read_column_data(mysql_engine: dict) -> pd.DataFrame:
    olap = Clickhouse("olap")

    column_data = olap.read(
        """
        select
            COLUMN_NAME as colname,
            DATA_TYPE as datatype,
            COLUMN_TYPE as coltype,
            COLUMN_KEY as colkey
        from odbc('DSN={dsn}', 'information_schema', 'columns')
        where TABLE_SCHEMA = '{schema_name}' and TABLE_NAME = '{table_name}'
        order by ORDINAL_POSITION
    """.format(
            **mysql_engine
        )
    )

    if column_data.shape[0] == 0:
        raise ValueError("Table {schema_name}.{table_name} not found".format(**mysql_engine))

    column_data = _apply_clickhouse_datatypes(column_data)
    return column_data


def _get_primary_keys(column_data: pd.DataFrame) -> str:
    primary_keys = column_data.query("colkey == 'PRI'").colname
    primary_keys = ", ".join(primary_keys)
    return primary_keys


def _apply_clickhouse_datatypes(column_data: pd.DataFrame) -> pd.Series:
    column_data = column_data.copy()
    column_data.datatype = column_data.datatype.apply(lambda datatype: Clickhouse.standartize_dtype(datatype))
    column_data.datatype = column_data.apply(
        lambda row: "U" + row.datatype if "Int" in row.datatype and "unsigned" in row.coltype else row.datatype, axis=1
    )
    return column_data


def _create_table(
    path: Path,
    ddl: str,
) -> None:
    if not path.exists():
        path.mkdir(parents=True)
    create_path = path / "0001.sql"
    if create_path.exists():
        print(f"Already exists: {create_path}")
    else:
        create_path.write_text(ddl)
        print(f"Created: {create_path}")


def _get_odbc_ddl(column_data: pd.DataFrame, mysql_engine: dict) -> str:
    column_ddl = _get_column_ddl(column_data)
    return (
        "CREATE TABLE ${dbname}.${tablename}\n"
        + f"(\n    {column_ddl}\n)\n"
        + "ENGINE = ODBC('DSN={dsn}', '{schema_name}', '{table_name}')\n".format(**mysql_engine)
    )


def _get_dump_ddl(column_data: pd.DataFrame) -> str:
    primary_keys = _get_primary_keys(column_data)
    column_ddl = _get_column_ddl(column_data)
    return (
        "CREATE TABLE ${dbname}.${tablename}\n"
        "(\n    load_dttm DateTime COMMENT 'Время загрузки (техническое поле)',"
        f"\n    {column_ddl}\n)\n"
        "ENGINE = ReplicatedMergeTree\n"
        f"ORDER BY ({primary_keys})\n"
        "SETTINGS index_granularity = 8192\n"
    )


def _get_column_ddl(column_data: pd.DataFrame) -> str:
    column_list = [
        f"{row.colname} {row.datatype}" + (f" COMMENT '{row.comment}'" if str(row.comment) not in ("", "nan") else "")
        for _, row in column_data.iterrows()
    ]
    column_ddl = ",\n    ".join(column_list)
    return column_ddl


def _read_table_comments(schema_name: str, table_name: str) -> pd.DataFrame:
    comment_directory_paths = [
        Path.home() / "target-web" / "db" / "doc",
        Path.home() / "orauthd" / "doc",
    ]
    md_possible_paths = [
        comment_directory_path / schema_name / f"{table_name}.md" for comment_directory_path in comment_directory_paths
    ]
    md_paths = [md_possible_path for md_possible_path in md_possible_paths if md_possible_path.exists()]

    if len(md_paths) > 0:
        md_path = md_paths[0]
        comments_md = md_path.read_text()
        comments_md = comments_md.split("Описание полей:\n\n")[-1]
        comments_md = comments_md.split("\n\n")[0]
        comments_md = comments_md.strip()

        comment_data = pd.read_csv(io.StringIO(comments_md), sep="|")
        comment_data = (
            comment_data.rename(columns=lambda x: x.strip())
            .rename(
                columns={
                    "Поле": "colname",
                    "Колонка": "colname",
                    "Описание": "comment",
                    "Назначение": "comment",
                }
            )[["colname", "comment"]]
            .query('colname != "---"')
            .fillna("")
            .applymap(lambda cell: cell.strip())
        )
        comment_data.comment = comment_data.comment.str.replace("'", "\\'")

        return comment_data

    else:
        print(f"No comment path exists: {md_possible_paths}")
        return pd.DataFrame(columns=("colname", "comment"))


def _get_alter_comment_ddl(column_data: pd.DataFrame) -> str:
    comment_list = [
        f"ALTER TABLE ${{dbname}}.${{tablename}} COMMENT COLUMN {row.colname} '{row.comment}';"
        for _, row in column_data.iterrows()
        if row.comment
    ]
    return "\n".join(comment_list)
