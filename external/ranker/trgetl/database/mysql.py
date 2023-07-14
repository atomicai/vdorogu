from contextlib import contextmanager
from typing import List, Union

import MySQLdb
import pandas as pd

from .. import tokens
from .base_database import BaseDatabase


class Mysql(BaseDatabase):
    DBS = [
        "billing",
        "configs",
        "mysql_target_dev_replica",
        "mysql_target_main_dev",
        "orauthddb",
        "rbmedia",
        "reklama",
        "target_auth",
        "target_stat",
        "target_ui",
        "target_web",
        "target",
    ]
    PROXY = "rbhp-ui2"
    PORT = 33306
    ENCODING = "utf8"

    def __init__(self, db: str = None):
        if db is None:
            db = "mysql_target_main_dev"
        assert db in self.DBS, f"Unknown db {db}"
        self.db = db

        self.db_parameters = dict(
            host=self.PROXY,
            port=self.PORT,
            user=getattr(tokens, "MYSQL_TRGDB_DEV_USER", None),
            passwd=getattr(tokens, "MYSQL_TRGDB_DEV_PASSWORD", None),
            db=self.db,
            charset=self.ENCODING,
        )

    @contextmanager
    def _connection(self) -> MySQLdb.connect:
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _get_connection(self) -> MySQLdb.connect:
        conn = None
        try:
            conn = MySQLdb.connect(**self.db_parameters)
            return conn
        except Exception as e:
            if conn is not None:
                conn.close()
            raise MySQLError(e)

    def read(self, query: Union[str, tuple]) -> pd.DataFrame:
        if isinstance(query, str):
            with self._connection() as conn:
                return pd.read_sql(sql=query, con=conn, parse_dates=True)
        else:
            raise NotImplementedError

    def execute(self, query: Union[str, tuple], return_rownum: bool = False) -> str:
        raise NotImplementedError

    def columns(self, table_name: str, return_dtypes: bool = False) -> List[str]:
        parts = table_name.split(".")
        if len(parts) == 1:
            schema, table = self.db, parts[0]
        elif len(parts) == 2:
            schema, table = parts
        else:
            raise MySQLError("Wrong table name")
        columns = self.read(
            f"""
            select COLUMN_NAME as name,  DATA_TYPE as type
            from INFORMATION_SCHEMA.COLUMNS
            where TABLE_SCHEMA = '{schema}' and TABLE_NAME = '{table}'
        """
        )
        if return_dtypes:
            return columns
        else:
            return list(columns.name)

    def tables(self) -> List[str]:
        tables = self.read(
            """
            select concat(TABLE_SCHEMA, '.', TABLE_NAME) as name
            from INFORMATION_SCHEMA.TABLES
        """
        )
        return list(tables.name)

    def search_tables(self, search: str) -> List[str]:
        tables = self.read(
            f"""
            select distinct concat(TABLE_SCHEMA, '.', TABLE_NAME) as table_name
            from INFORMATION_SCHEMA.COLUMNS
            where TABLE_SCHEMA like '%{search}%'
                or TABLE_NAME like '%{search}%'
        """
        )
        return list(tables.table_name)


class MySQLError(Exception):
    pass
