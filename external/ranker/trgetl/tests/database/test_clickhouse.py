import pandas as pd
import pytest

from ...database.clickhouse import Clickhouse, ClickhouseError
from ..fixture import ch_temp_table_name, temp_table_df, temp_table_values  # noqa: F401

SOME_ACTUAL_TABLES = {
    'default.banner_pad_d1': ['banner_id', 'pad_id', 'shows', 'amount'],
    'dictionary.package': ['package_id', 'name'],
}


def test_ch_insert_to_wrong_db():
    ch = Clickhouse('events')
    with pytest.raises(ClickhouseError):
        ch.insert(table_name='table_name', df=pd.DataFrame())


def test_ch_insert_olap(ch_temp_table_name):  # noqa: F811
    ch = Clickhouse('olap')
    row_num = ch.insert(table_name=ch_temp_table_name, df=temp_table_df)
    assert int(row_num) == temp_table_df.shape[0]


def test_ch_insert_by_execute(ch_temp_table_name):  # noqa: F811
    ch = Clickhouse('olap')
    row_num = ch.execute(f'insert into {ch_temp_table_name} values {temp_table_values}', return_rownum=True)
    assert int(row_num) == temp_table_df.shape[0]


def test_ch_execute_events():
    ch = Clickhouse('events')
    result = ch.execute("select 'query_result'")
    assert result == b'query_result'


def test_ch_read_events():
    ch = Clickhouse('events')
    result = ch.read("select 'query_result' as res")
    assert (result == pd.DataFrame({'res': ['query_result']})).all().all()


def test_ch_wrong_query():
    ch = Clickhouse('olap', max_memory_usage_gb=1)
    with pytest.raises(ClickhouseError):
        ch.read('wrong sql')


def test_retry(capsys):
    ch = Clickhouse('olap', retries=2, retry_sleep=0)
    with pytest.raises(ClickhouseError):
        ch.read("select '{}'".format(list(ch.RETRY_ERRORS)[0]))
    captured = capsys.readouterr()
    assert 'retrying...' in captured.out


def test_truncate(ch_temp_table_name):  # noqa: F811
    ch = Clickhouse('olap')
    rownum_query = f'select count() from {ch_temp_table_name}'
    ch.insert(ch_temp_table_name, temp_table_df)
    assert int(ch.execute(rownum_query)) > 0
    deleted_rownum = ch.truncate(ch_temp_table_name)
    assert deleted_rownum == temp_table_df.shape[0]
    assert int(ch.execute(rownum_query)) == 0


def test_delete(ch_temp_table_name):  # noqa: F811
    ch = Clickhouse('olap')
    date = temp_table_df.date.iloc[0]
    rownum_query = f'select count() from {ch_temp_table_name}'
    ch.insert(ch_temp_table_name, temp_table_df)
    assert int(ch.execute(rownum_query)) > 0
    deleted_rownum = ch.delete(ch_temp_table_name, date)
    assert deleted_rownum == temp_table_df.query(f"date == '{date}'").shape[0]
    assert int(ch.execute(rownum_query)) == temp_table_df.query(f"date != '{date}'").shape[0]


def test_tables():
    ch = Clickhouse('olap')
    tables = ch.tables()
    assert isinstance(tables, list)
    assert len(set(SOME_ACTUAL_TABLES) - set(tables)) == 0


@pytest.mark.parametrize('table_name, expected_columns', SOME_ACTUAL_TABLES.items())
def test_columns(table_name, expected_columns):
    ch = Clickhouse('olap')
    columns = ch.columns(table_name)
    assert isinstance(columns, list)
    assert len(set(expected_columns) - set(columns)) == 0


@pytest.mark.parametrize('table_name, expected_columns', SOME_ACTUAL_TABLES.items())
def test_columns_with_dtypes(table_name, expected_columns):
    ch = Clickhouse('olap')
    df = ch.columns(table_name, return_dtypes=True)
    assert isinstance(df, pd.DataFrame)
    df_columns = {'name', 'type', 'comment'}
    assert len(df_columns - set(df.columns)) == 0
    assert len(set(expected_columns) - set(df['name'])) == 0


@pytest.mark.parametrize('table_name, expected_columns', SOME_ACTUAL_TABLES.items())
def test_search_tables(table_name, expected_columns):
    ch = Clickhouse('olap')
    search_result_by_table_name = ch.search_tables(table_name[2:-5])
    assert table_name in search_result_by_table_name
    search_result_by_column = ch.search_tables(expected_columns[0])
    assert table_name in search_result_by_column
