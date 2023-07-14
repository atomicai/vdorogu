import csv
import getpass
import io
from contextlib import contextmanager

import pandas as pd
import psycopg2

from .. import tokens
from .base_database import BaseDatabase, retry


class Postgres(BaseDatabase):
    DEVELOPER_USERS = ["e.shelavina", "d.kulemin", "airflow-trgetl", "jenkins-trgan"]
    DB_PARAMETERS = {
        "opex": dict(dbname="opex", host="10.146.230.1", port=5432),
        "trg": dict(dbname="target_dev", host="rbhp-trg1.rbdev.mail.ru", port=5432),
    }
    RETRY_ERRORS = {"MySQL server has gone away": "MySQL server has gone away"}
    DB_AUTH_METHODS = {
        "opex": "password",
        "trg": "role",
    }

    def __init__(
        self,
        db: str = None,
        role: str = None,
        retries: int = 10,
        retry_sleep: int = 10,
        user: str = None,
        password: str = None,
    ):
        if db is None:
            db = "trg"

        assert db in self.DB_PARAMETERS, f"Unknown db: {db}"
        self.db = db
        self.db_parameters = self.DB_PARAMETERS[db].copy()
        self.retries = retries
        self.retry_sleep = retry_sleep

        auth_method = self.DB_AUTH_METHODS[db]
        if auth_method == "role":
            self.role = self._get_role(role)
        else:
            self.role = "no role"

        if auth_method == "password":
            if user is None:
                user = getattr(tokens, f"PG_{self.db.upper()}_USER", None)
            self.db_parameters["user"] = user
            if password is None:
                password = getattr(tokens, f"PG_{self.db.upper()}_PASSWORD", None)
            self.db_parameters["password"] = password

    @retry
    def execute(self, query, vars=None, return_rownum=True):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, vars)
            if return_rownum:
                return cursor.rowcount
            else:
                return cursor.fetchall()

    @retry
    def read(self, query):
        with self._connection() as conn:
            return pd.read_sql(sql=query, con=conn, parse_dates=True)

    @retry
    def insert(self, table_name: str, df: pd.DataFrame):
        df = df.replace({"\\N": None})
        buffer = io.StringIO()
        df.to_csv(
            buffer,
            index=False,
            header=False,
            sep="\t",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        buffer.seek(0)
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.copy_from(buffer, table_name, sep="\t")
            return cursor.rowcount

    def delete(self, table_name, date, date_column="date"):
        initial_rownum = self.get_table_rownum(table_name, date, date_column)
        if initial_rownum == 0:
            print(f"Nothing to delete: {table_name}, {date}")
            return 0
        delete_query = f"delete from {table_name} where {date_column} = '{date}'"
        self.execute(delete_query)
        return initial_rownum

    def truncate(self, table_name):
        initial_rownum = self.get_table_rownum(table_name)
        if initial_rownum == 0:
            print(f"Nothing to truncate: {table_name}")
            return 0
        truncate_query = f"truncate table {table_name}"
        self.execute(truncate_query)
        return initial_rownum

    def columns(self, table_name, return_dtypes=False) -> list:
        schema, table = table_name.split(".")
        columns = self.read(
            f"""
            select column_name as name, data_type as type
            from information_schema.columns
            where table_schema = '{schema}' and table_name = '{table}'
            order by ordinal_position
        """
        )
        if return_dtypes:
            return columns
        else:
            return list(columns.name)

    def tables(self):
        tables = self.read(
            """
            select concat(table_schema, '.', table_name) as name
            from information_schema.tables
        """
        )
        return sorted(list(tables.name))

    def search_tables(self, search):
        tables = self.read(
            f"""
            select distinct concat(table_schema, '.', table_name) as name
            from information_schema.columns
            where concat(table_schema, '.', table_name) like '%{search}%'
                or column_name like '%{search}%'
        """
        )
        return sorted(list(tables.name))

    def now_query(self):
        return "cast(date_trunc('second', now()) as timestamp)"

    @contextmanager
    def _connection(self):
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(**self.db_parameters)
            conn.set_session(autocommit=True)
            if self.role != "no role":
                conn.cursor().execute(f"set role {self.role};")
            return conn
        except Exception as e:
            if conn is not None:
                conn.close()
            raise PostgresError(e)

    @classmethod
    def _get_role(cls, role=None):
        if role is None:
            if getpass.getuser() in cls.DEVELOPER_USERS:
                role = "target_developer"
            else:
                role = "target_sandboxer"
        assert role in ("target_developer", "target_sandboxer", "no role")
        return role

    def get_table_rownum(self, table_name, date=None, date_column="date"):
        query = f"select count(*) from {table_name}"
        if date:
            query += f" where {date_column} = '{date}'"
        rownum = int(self.read(query).iloc[0, 0])
        return rownum


class PostgresError(Exception):
    pass
