import csv
import datetime as dt
import getpass
import hashlib
import io
import socket
import time
import urllib
from pathlib import Path
from typing import Hashable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import requests
import requests_kerberos

from .. import tokens
from ..filesystem import Filesystem
from . import helpers
from .base_database import BaseDatabase, measure, retry


class Clickhouse(BaseDatabase):
    RAW_MODE = True

    DB_URLS = {
        'events': 'http://clickhouse.hp.rbdev.mail.ru/clusters/trg_events/shards/distributed/'
        '?max_query_size=100000000&allow_experimental_window_functions=1&',
        'eventspass': 'http://ch.trg.corp.mail.ru/trg_events/shards/distributed/'
        '?max_query_size=100000000&allow_experimental_window_functions=1&',
        'dbhistory': 'http://clickhouse.hp.rbdev.mail.ru/clusters/trg_web/shards/1/' '?max_query_size=100000000&',
        'olap': 'http://clickhouse.hp.rbdev.mail.ru/trg1/'
        '?database=default&max_query_size=100000000'
        '&max_execution_time=7200'
        '&max_memory_usage=32000000000'
        '&timeout_before_checking_execution_speed=1'
        '&allow_experimental_window_functions=1'
        '&mutations_sync=1'
        '&insert_deduplicate=0'
        '&',
        'mnt': 'http://clickhouse.hp.rbdev.mail.ru/mnt/?',
        'vk': 'http://ch-vk-ads-analytics.hp.rbdev.mail.ru/' '?allow_experimental_window_functions=1' '&',
        'vkold': 'http://ch-vk-ads.hp.rbdev.mail.ru/?',
        'vketl': 'http://ch-vk-etl.hp.rbdev.mail.ru/' '?allow_experimental_window_functions=1'
        # '&insert_deduplicate=0'
        '&',
        'vk_all': 'http://ch-vk.hp.rbdev.mail.ru/?',
        'cx_hub': 'http://notify-endpoint.hp.rbdev.mail.ru/api/events/adtech/?',
        'gpmd': 'http://clickhouse.hp.rbdev.mail.ru/clusters/videomatic_v2/shards/general/?',
    }

    DB_TITLES = {
        'events': 'trg CH distributed',
        'dbhistory': 'trg CH logs',
        'olap': 'clickhouse-trg1',
        'mnt': 'clickhouse-mnt',
        'vk': 'vk clickhouse ads_analytics',
        'vk_all': 'system',
        'vkold': 'vk clickhouse ads_analytics',
        'vketl': 'vk clickhouse all',
        'gpmd': 'videomatic cluster',
    }

    DB_AUTH_METHODS = {
        'events': 'kerberos',
        'eventspass': 'password',
        'dbhistory': 'kerberos',
        'olap': 'kerberos',
        'mnt': 'kerberos',
        'vk': 'kerberos',
        'vkold': 'kerberos',
        'vketl': 'kerberos',
        'vk_all': 'kerberos',
        'cx_hub': 'kerberos',
        'gpmd': 'kerberos',
    }

    DB_MUTATE_ALLOWED = ['olap', 'vketl', 'cx_hub']

    RETRY_ERRORS = {
        'DB::Exception: Possible deadlock avoided. Client should retry': 'Possible deadlock avoided',
        'DB::Exception: Cannot read all data. Bytes read': 'Data read error',
        'DB::Exception: ODBCBridgeHelper: clickhouse-odbc-bridge is not responding': 'Waiting for ODBCBridgeHelper',
        'DB::Exception: Cannot schedule a task': 'Scheduling failed',
        'DB::NetException: Timeout exceeded while reading from socket': 'Socket timeout encountered',
        'DB::NetException: Timeout exceeded while writing to socket': 'Socket timeout encountered',
        "ConnectionResetError(104, 'Connection reset by peer'))": 'Connection reset by peer',
        '502 Bad Gateway 502 Bad Gateway kittenx': 'Kittenx encountered',
    }

    def __init__(
        self,
        db: str = None,
        max_memory_usage_gb: int = None,
        retries: int = 10,
        retry_sleep: int = 10,
        user: str = None,
        password: str = None,
    ):
        if db is None:
            db = 'olap'

        assert db in self.DB_URLS, f'Unknown db: {db}'
        self.db = db
        self.retries = retries
        self.retry_sleep = retry_sleep

        self.url = self.DB_URLS[db]
        if max_memory_usage_gb is not None:
            assert 1 <= max_memory_usage_gb <= 64, "max_memory_usage_gb should be in [1, 64]"
            self.url = self._add_url_property(self.url, max_memory_usage_gb=max_memory_usage_gb)

        self.prefix_pattern = self._get_prefix_pattern(max_memory_usage_gb)

        if user is None:
            user = getattr(tokens, f'CH_{self.db.upper()}_USER', None)
        self.user = user
        if password is None:
            password = getattr(tokens, f'CH_{self.db.upper()}_PASSWORD', None)
        self.password = password

    def insert(
        self,
        table_name: str,
        df: pd.DataFrame,
        return_rownum: bool = True,
    ) -> Optional[int]:
        df = self._fillna(df)
        df = self._cast_bool_as_int(df)
        data = df.to_csv(
            index=False, header=False, sep='\t', float_format='%.12g', quoting=csv.QUOTE_NONE, escapechar='\\'
        )
        rownum = self.insert_string(table_name, data, return_rownum)
        return rownum

    def insert_string(
        self,
        table_name: str,
        data: Union[str, bytes],
        return_rownum: bool = True,
        format: str = None,
    ) -> Optional[int]:
        self._assert_mutate_allowed()
        if format is None:
            format = 'TabSeparated'
        self._assert_mutate_allowed()
        func_prefix = 'function ' if '(' in table_name else ''
        query = f'insert into {func_prefix}{table_name} format {format}'
        qhash, prefixed_query = self._get_prefixed_query(query, is_insert=True)

        response = self._request_db(prefixed_query, data)

        if response:
            raise ClickhouseError('INSERT RESULT MESSAGE: ' + str(response))

        if return_rownum:
            query_info = self._wait_for_query_info(qhash)
            rownum = query_info['written_rows']
            return rownum
        else:
            return None

    def read(self, query: str = None, path: Union[str, Path] = None) -> pd.DataFrame:
        if query is None and path is None:
            raise ClickhouseError('Neither query not path is set in Clickhouse.read')
        if query is None:
            assert path is not None
            query = self._read_query_from_file(path=path)

        if ' format ' not in query:
            query += ' format CSVWithNames'
        if 'format CSV' in query:
            _sep = ','
        elif 'format TabSeparated' in query:
            _sep = '\t'
        else:
            raise AssertionError('Unknown format: https://clickhouse.yandex/docs/en/interfaces/formats/')

        data = self.execute(query, return_str=True)
        df = pd.read_csv(io.StringIO(data), sep=_sep, parse_dates=True)
        return df

    @measure
    def read_string(self, query: str, return_str: bool = False) -> str:
        return self.execute(query, return_str=return_str)

    def execute(
        self,
        query: str,
        return_rownum: bool = False,
        wait: bool = False,
        return_str: bool = False,
    ) -> str:
        qhash, prefixed_query = self._get_prefixed_query(query)
        try:
            response = self._request_db(prefixed_query, return_str=return_str)
        except ClickhouseError as e:
            if return_rownum and '504 Gateway Time-out nginx' in str(e):
                print('Nginx timeout encountered, waiting for query finish...')
            else:
                raise e

        if wait or return_rownum:
            query_info = self._wait_for_query_info(qhash)
            if return_rownum:
                rownum = query_info['written_rows']
                return rownum

        return response

    def delete(self, table_name: str, date: Union[str, dt.date], date_column: str = 'date') -> int:
        self._assert_mutate_allowed()
        initial_rownum = self.get_table_rownum(table_name, date, date_column)
        if initial_rownum == 0:
            print(f'Nothing to delete: {table_name}, {date}')
            return 0
        if table_name.startswith('cluster(') and table_name.endswith(')'):
            cluster, table_name, *_ = table_name[8:-1].split(',')
            table_name = '{} on cluster {}'.format(table_name, cluster[1:-1])
        delete_query = f"alter table {table_name} " f"delete where {date_column} = '{date}'"
        self.execute(delete_query)
        if self._wait_for_deletion(table_name, date, date_column):
            return initial_rownum
        else:
            raise ClickhouseError(f'Delete failed: {table_name}, {date}')

    def truncate(self, table_name: str) -> int:
        self._assert_mutate_allowed()
        initial_rownum = self.get_table_rownum(table_name)
        if initial_rownum == 0:
            print(f'Nothing to truncate: {table_name}')
            return 0
        truncate_query = f"truncate table if exists {table_name} on cluster trg"
        self.execute(truncate_query)
        if self._wait_for_deletion(table_name):
            return initial_rownum
        else:
            raise ClickhouseError(f'Truncate failed: {table_name}')

    def columns(self, table_name: str, return_dtypes: bool = False) -> Union[list, pd.DataFrame]:
        schema, table = table_name.split('.')
        columns: pd.DataFrame = self.read(
            f'''
            select name, type, comment from system.columns
            where database = '{schema}' and table = '{table}'
        '''
        )
        if columns.shape[0] == 0:
            raise ClickhouseError(f'Table {table_name} not found in clickhouse')
        if return_dtypes:
            return columns
        else:
            return list(columns.name)

    def tables(self) -> list:
        tables = self.read(
            '''
            select concat(database, '.', name) as name
            from system.tables
        '''
        )
        return list(tables.name)

    def search_tables(self, search_phrase: str) -> list:
        tables = self.read(
            f'''
            select distinct concat(database, '.', table) as table_name
            from system.columns
            where table_name like '%{search_phrase}%'
                or name like '%{search_phrase}%'
        '''
        )
        return list(tables.table_name)

    def show_create_table(self, table_name: str, return_result: bool = False) -> Optional[str]:
        ddl = self.execute(f'show create table {table_name}', return_str=True)
        ddl = ddl.replace('\\n', '\n')
        ddl = ddl.replace("\\'", "'")
        ddl = ddl.replace('`', '')
        if return_result:
            return ddl
        else:
            print(ddl)
            return None

    def now_query(self) -> str:
        return 'now()'

    def hdfs_query(
        self,
        path: Union[Path, str],
        format_: str,
        structure: List[Tuple[str, str]],
    ) -> str:
        hdfs_formats = {
            'orc': 'ORC',
            'parquet': 'Parquet',
        }
        format_ = hdfs_formats[format_]
        structure_str = ', '.join(name + ' ' + self.standartize_dtype(dtype) for name, dtype in structure)
        query = f"hdfs('hdfs://hacluster{path}', '{format_}', '{structure_str}')"
        return query

    def insert_into_public(
        self,
        table_name: str,
        df: pd.DataFrame,
        load_dttm: bool = True,
    ) -> int:
        user = getpass.getuser().replace('.', '')
        table_name = f'public.{user}__{table_name}'

        ddl_query = self._ddl_from_dataframe(df=df, table_name=table_name, load_dttm=load_dttm)
        self.execute(ddl_query)
        print(f'Created {table_name}')

        if load_dttm:
            now = dt.datetime.now()
            now = now.replace(microsecond=0)
            load_dttm_series = pd.Series([now] * df.shape[0])
            df.insert(0, 'load_dttm', load_dttm_series)

        rownum: int = self.insert(table_name=table_name, df=df)  # type: ignore
        return rownum

    def explain(self, query: str, explain_type: str = 'PLAN') -> str:
        if explain_type.lower() not in ['ast', 'syntax', 'plan', 'pipeline']:
            raise ClickhouseError(
                'explain type should be one of `AST`, `SYNTAX`, `PLAN`, `PIPELINE`,' f' but got `{explain_type}`'
            )
        explain_prefix = 'EXPLAIN ' + explain_type
        query = explain_prefix + query
        return self.read_string(query, return_str=True)

    @staticmethod
    def standartize_dtype(dtype: str) -> str:
        dtypes = {
            'array<string>': 'Array(String)',
            'bigint': 'Int64',
            'boolean': 'Int8',
            'int': 'Int32',
            'string': 'String',
            'tinyint': 'Int8',
            'int64': 'Int64',
            'float32': 'Float32',
            'float64': 'Float64',
            'bool': 'Int8',
            'object': 'String',
            'datetime64[ns]': 'DateTime',
            'blob': 'String',
            'char': 'String',
            'date': 'Date',
            'datetime': 'DateTime',
            'decimal': 'Float64',
            'double': 'Float64',
            'enum': 'String',
            'float': 'Float64',
            'json': 'String',
            'longtext': 'String',
            'mediumtext': 'String',
            'set': 'String',
            'smallint': 'Int32',
            'text': 'String',
            'timestamp': 'String',
            'varchar': 'String',
        }
        dtype = dtypes.get(dtype, dtype)
        return dtype

    def _assert_mutate_allowed(self) -> None:
        if self.db not in self.DB_MUTATE_ALLOWED:
            raise ClickhouseError(f'Delete is not allowed for db {self.db}')

    def _add_url_property(self, url: str, **properties: Hashable) -> str:
        for property_name, property_value in properties.items():
            if property_name == 'max_memory_usage_gb':
                property_name = 'max_memory_usage'
                assert isinstance(property_value, (int, float))
                property_value *= 2**30
                property_value = int(property_value)
            url += urllib.parse.urlencode({property_name: str(property_value)}) + '&'  # type: ignore  # noqa: E501
        return url

    def _get_prefix_pattern(self, max_memory_usage_gb: int = None) -> str:
        user = getpass.getuser()
        host = socket.gethostname()
        prefix = f'Username: {user}@corp.mail.ru, Hostname: {host}, Qhash: {{qhash}}'
        if max_memory_usage_gb is not None:
            prefix += f', Maxmem: {max_memory_usage_gb}'
        return '/* ' + prefix + ' */\n'

    def _get_prefixed_query(self, query: str, is_insert: bool = False) -> Tuple[str, str]:
        qhash_params = [query, self.db, int(is_insert), int(time.time() * 1000)]
        qsign = helpers.csvit(qhash_params, sep='|')
        qhash = hashlib.md5(qsign.encode()).hexdigest()
        prefix = self.prefix_pattern.format(qhash=qhash)
        prefixed_query = prefix + query
        return qhash, prefixed_query

    @retry
    def _request_db(
        self,
        query: str,
        data: Union[bytes, str] = None,
        return_str: bool = False,
    ) -> Union[bytes, str]:
        auth = self._get_auth()
        data = data.encode() if isinstance(data, str) else data
        params = {}

        if data is None:
            data = query.encode()
        else:
            params['query'] = query

        response = requests.post(
            url=self.url,
            headers={'Content-type': 'bytes/plain'},
            cookies={},
            auth=auth,
            data=data,
            params=params,
        )

        error = self._response_error(response)
        if error is not None:
            if 'DB::Exception: Syntax error:' in error[:1000]:
                error += '\n\nQUERY:\n\n' + query
            raise ClickhouseError(error)

        if return_str:
            result = response.text
            new_line = '\n'
        else:
            result = response.content
            new_line = b'\n'  # type: ignore

        if len(result) <= 1000:
            result = result.rstrip(new_line)
        return result

    def _response_error(self, response: requests.Response) -> Optional[str]:
        if response.status_code != 200:
            return f'status_code={response.status_code}\n' + helpers.clean_html(response.text)

        exception_markers = ['DB::Exception', 'DB::NetException']
        text_end = response.text[-1000:]
        for exception_marker in exception_markers:
            if exception_marker in text_end:
                return text_end[text_end.find(exception_marker) :].strip()

        return None

    def _get_auth(self) -> Union[requests_kerberos.HTTPKerberosAuth, Tuple[str, str]]:
        auth_method = self.DB_AUTH_METHODS[self.db]
        if auth_method == 'kerberos':
            auth_dict = {"mutual_authentication": requests_kerberos.DISABLED}
            if self.db in self.DB_MUTATE_ALLOWED:
                auth_dict["force_preemptive"] = True
            auth = requests_kerberos.HTTPKerberosAuth(**auth_dict)

        elif auth_method == 'password':
            if self.user is None or self.password is None:
                raise ClickhouseError(
                    f'User or Password not set for clickhouse {self.db}: ' f'{self.user} {self.password}'
                )
            auth = (self.user, self.password)

        else:
            auth = None

        return auth

    def get_table_rownum(self, table_name: str, date: Union[dt.date, str] = None, date_column: str = 'date') -> int:
        query = f'select count() from {table_name}'
        if date:
            query += f" where {date_column} = '{date}'"
        rownum = int(self.execute(query))
        return rownum

    def _wait_for_query_info(
        self,
        qhash: str,
        minutes_wait: int = 60,
        sleep_seconds: int = 1,
    ) -> dict:
        start_time = time.time()
        time.sleep(sleep_seconds)

        exception_before_start = self._exception_before_start_occured(qhash, (minutes_wait + 60))
        if exception_before_start is not None:
            raise ClickhouseError(f"ExceptionBeforeStart occured for Qhash: {qhash}\n{exception_before_start}")

        while helpers.timeit(start_time) / 60.0 <= minutes_wait:
            query_info = self._get_query_info(qhash, minutes_wait=minutes_wait)
            if query_info:
                return query_info
            time.sleep(sleep_seconds)
        raise ClickhouseError(f"No queries for Qhash: {qhash} in system.query_log")

    def _get_query_info(
        self,
        qhash: str,
        minutes_wait: int = 60,
    ) -> dict:
        """Get query info by Qhash field in comments"""
        user_name = getpass.getuser()
        query_info_sql: str = Filesystem.read_misc_query('system.query_log.written_rows')
        query_info_sql = query_info_sql.format(
            user_name=user_name,
            minutes_wait=minutes_wait,
            query_hash=qhash,
        )
        query_info_df: pd.DataFrame = self.read(query_info_sql)

        if query_info_df.shape[0]:
            query_info: dict = query_info_df.iloc[0, :].to_dict()
        else:
            query_info = dict()
        return query_info

    def _exception_before_start_occured(self, qhash: str, minutes_wait: int = 120) -> Optional[str]:
        user_name = getpass.getuser()
        exception_info_sql: str = Filesystem.read_misc_query('system.query_log.exception_before_start')
        exception_info_sql = exception_info_sql.format(
            user_name=user_name,
            minutes_wait=minutes_wait,
            query_hash=qhash,
        )
        exception_info_df: pd.DataFrame = self.read(exception_info_sql)
        if exception_info_df.shape[0] > 0:
            exception: str = exception_info_df.loc[0, 'exception']
            return exception
        else:
            return None

    def _wait_for_deletion(
        self,
        table_name: str,
        date: Union[dt.date, str] = None,
        date_column: str = 'date',
        minutes_wait: int = 5,
        sleep_seconds: int = 1,
    ) -> bool:
        start_time = time.time()
        while helpers.timeit(start_time) / 60.0 <= minutes_wait:
            final_rownum = self.get_table_rownum(table_name, date, date_column)
            if final_rownum == 0:
                return True
            time.sleep(sleep_seconds)
        return False

    def _cast_bool_as_int(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bool_columns = df.columns[df.fillna(False).dtypes == bool]
        for colname in bool_columns:
            df[colname] = df[colname].fillna(False).astype(int)
        return df

    def _fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna(
            {
                colname: dt.datetime(2000, 1, 1, 0, 0, 0)
                for colname, dtype in df.dtypes.items()
                if pd.api.types.is_datetime64_any_dtype(dtype)
            }
        )
        return df

    def _read_query_from_file(self, path: Union[Path, str]) -> str:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise ClickhouseError(f'Path {path} does not exist')
        if not path.is_file():
            raise ClickhouseError(f'Path {path} is not a file')
        query = path.read_text()
        return query

    def _ddl_from_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        load_dttm: bool = False,
        order_by: Optional[Sequence] = None,
    ) -> str:
        column_types = df.dtypes.astype(str)
        columns = ',\n'.join(
            f'{column_name} {self.standartize_dtype(column_type)}' for column_name, column_type in column_types.items()
        )
        if load_dttm:
            columns = 'load_dttm DateTime,\n' + columns

        first_column = df.columns[0]
        order_by = order_by or first_column

        ddl = f'''
            CREATE TABLE {table_name}
            ({columns})
            ENGINE = MergeTree
            ORDER BY ({order_by})
            SETTINGS index_granularity = 8192'''

        return ddl


class ClickhouseError(Exception):
    pass
