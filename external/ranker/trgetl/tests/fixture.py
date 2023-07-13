import pandas as pd
import pytest

from ..database import Clickhouse, Postgres


@pytest.fixture
def ch_temp_table_name():
    ch = Clickhouse('olap')
    temp_table_name = 'public.pytest_temp_table'
    ch.execute(
        f'''
    CREATE TABLE IF NOT EXISTS {temp_table_name} (
        date    Date,
        int8    Int8,
        string  String
    )
    ENGINE=MergeTree
    ORDER BY date
    ''',
        'olap',
    )
    yield temp_table_name
    ch.execute(f'DROP TABLE IF EXISTS {temp_table_name}', 'olap')


@pytest.fixture
def pg_temp_table_name():
    pg = Postgres()
    temp_table_name = 'sandbox.pytest_temp_table'
    pg.execute(
        f'''
    CREATE TABLE IF NOT EXISTS {temp_table_name} (
        date    date,
        int8    int,
        string  varchar(50)
    )
    '''
    )
    yield temp_table_name
    pg.execute(f'DROP TABLE IF EXISTS {temp_table_name}')


temp_table_df = pd.DataFrame(
    {
        'date': ['2021-05-20', '2022-01-01'],
        'int8': [-100, 55],
        'string': ['ascii123', 'кириллица'],
    }
)

temp_table_values = ', '.join(
    temp_table_df.applymap(lambda x: "'" + x + "'" if isinstance(x, str) else str(x)).apply(
        lambda row: '(' + ', '.join(row) + ')', axis=1
    )
)
