import pandas as pd

from ...database.postgres import Postgres, PostgresError
from ..fixture import pg_temp_table_name, temp_table_df, temp_table_values


def test_pg_read():
    pg = Postgres("trg")
    result = pg.read(query="select 'query_result' as res")
    assert (result == pd.DataFrame({"res": ["query_result"]})).all().all()


def test_exec_sql(pg_temp_table_name):
    pg = Postgres("trg")
    row_num = pg.execute(f"insert into {pg_temp_table_name} values {temp_table_values}")
    assert int(row_num) == temp_table_df.shape[0]
