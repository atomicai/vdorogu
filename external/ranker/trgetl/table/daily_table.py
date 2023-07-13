import calendar
import datetime as dt
import difflib
from abc import abstractmethod
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import pandas as pd

from ..database import Database
from ..database.base_database import BaseDatabase
from ..filesystem import TableInFilesystem
from . import helpers
from .exceptions import TableRunError


class DailyTable:
    def __init__(self, name: str):
        self.name = name
        self.filesystem_representation = TableInFilesystem(name)
        self._raw_parameters = self.filesystem_representation.get_parameters()
        self.parameters = self.filesystem_representation.get_standart_parameters()
        self.is_full = self.parameters['is_full']
        self.db = self._find_db()
        self.source_db = self._find_db(source=True)

        self.query: Union[str, ModuleType] = ''

    def __repr__(self) -> str:
        dependencies = self.filesystem_representation.extract_all_dependencies()
        name = (
            f"{type(self).__name__}('{self.name}') in {self.db}, full_reload={self.is_full}\n"
            f"    source: {self.source_db}, dependencies: {dependencies}\n"
            f"    parameters: {self._raw_parameters}"
        )
        return name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def _correct_date(function: Callable) -> Callable:  # type: ignore
        @wraps(function)
        def wrapped(self, date: Optional[Union[dt.date, str]] = None, date_correction: bool = False) -> int:  # type: ignore
            if self.is_full:
                date = None
            elif date is None:
                date_correction = True
                date = helpers.yesterday()
            elif isinstance(date, str):
                date = dt.datetime.strptime(date, '%Y-%m-%d').date()
            assert isinstance(date, (dt.date, type(None)))

            if date_correction:
                date = self._shift_date(date)

            schedule = self.parameters['schedule']
            daterange = self.parameters.get('daterange')
            if (
                daterange is not None
                and date_correction
                and not self.is_full
                and schedule == 'day'
                and isinstance(date, dt.date)
            ):
                rownum = 0
                for datelag in range(*daterange):
                    print(f'Date {datelag + 1} of range({daterange[0]}, {daterange[1]}):')
                    current_date = date - dt.timedelta(days=datelag)
                    rownum += function(self, current_date)
            elif (
                not self.is_full
                and date is not None
                and schedule != 'day'
                and date != helpers.to_start_of_interval(date, interval=schedule)
            ):
                print(f'Date {date} not in schedule "{schedule}", passing')
                rownum = 0
            else:
                rownum = function(self, date)

            return rownum

        return wrapped

    @_correct_date
    @helpers.log
    def run(self, date: Optional[dt.date] = None) -> int:
        query = self._prepare_query(self.query, date)
        print(f'Starting (date: {date}):\n{self}')

        if self.parameters['as_file']:
            rownum = self._file_upload(date, query)
        elif self.db == self.source_db:
            rownum = self._internal_upload(date, query)
        else:
            rownum = self._external_upload(date, query)

        if rownum == 0 and not self.parameters['allow_zero']:
            raise TableRunError(f'Empty insertion to {self.name}')

        if self.parameters.get('insert_as_file'):
            print(f'Inserted data to {self.name}')
        else:
            print(f'Inserted {rownum} rows to {self.name}')

        if self.filesystem_representation.is_dictionary():
            self.db.execute('system reload dictionaries')
            print('Dictionaries reloaded')

        if not self.parameters.get('insert_as_file'):
            self._check_rownum(rownum, date)

        return rownum

    @abstractmethod
    def _file_upload(self, date: Optional[dt.date], input: Union[str, tuple]) -> int:
        raise NotImplementedError

    def print_query(self) -> None:
        print(self.query)

    def _internal_upload(
        self,
        date: Optional[dt.date],
        query: Union[str, tuple],
    ) -> int:
        total_chunks, chunksize = self._calc_chunks(date)
        self._clear(date)
        if total_chunks > 1:
            rownum = self._internal_upload_with_chunks(date, query, total_chunks, chunksize)
        else:
            query = f'insert into {self.name} {query}'
            response = self.db.execute(query, return_rownum=True)
            try:
                rownum = int(response)
            except ValueError:
                rownum = 0
        return rownum

    def _external_upload(
        self,
        date: Optional[dt.date],
        query: Union[str, tuple],
    ) -> int:
        total_chunks, chunksize = self._calc_chunks(date)
        if total_chunks > 1:
            self._clear(date)
            rownum = self._external_upload_with_chunks(date, query, total_chunks, chunksize)
        else:
            data = self._read(query)
            self._clear(date)
            rownum = self._insert(data, date)
        return rownum

    def _internal_upload_with_chunks(
        self,
        date: Optional[dt.date],
        query: Union[str, tuple],
        total_chunks: int,
        chunksize: int,
    ) -> int:
        print(f'Reading in {total_chunks} chunks')
        rownum = 0
        query = f'insert into {self.name} {query}'
        for chunk in range(total_chunks):
            chunk_query = self._get_chunk_query(query, chunk, total_chunks, chunksize)
            response = self.db.execute(chunk_query, return_rownum=True)
            try:
                rownum += int(response)
            except ValueError:
                print(f'Could not convert response: {response.__repr__()} to Int')
            print(f'Table {self.name}: chunk {chunk+1} of {total_chunks}: ' f'total of {rownum} rows inserted')
        return rownum

    @abstractmethod
    def _get_chunk_query(self, query: Union[str, tuple], chunk: int, total_chunks: int, chunksize: int) -> str:
        raise NotImplementedError

    def _external_upload_with_chunks(
        self,
        date: Optional[dt.date],
        query: Union[str, tuple],
        total_chunks: int,
        chunksize: int,
    ) -> int:
        print(f'Reading in {total_chunks} chunks')
        rownum = 0
        for chunk in range(total_chunks):
            retries = self.parameters.get('retries', 1)
            while retries:
                try:
                    chunk_query = self._get_chunk_query(query, chunk, total_chunks, chunksize)
                    chunk_data = self._read(chunk_query)
                    rownum += self._insert(chunk_data)
                    print(f'Table {self.name}: chunk {chunk+1} of {total_chunks}: ' f'total of {rownum} rows inserted')
                    break
                except Exception as e:
                    retries -= 1
                    if retries:
                        print(str(e))
                    else:
                        raise (e)
        return rownum

    def _calc_chunks(self, date: Optional[dt.date] = None) -> Tuple[int, int]:
        total_chunks = self.parameters.get('total_chunks')
        chunksize = self.parameters.get('chunksize')
        if total_chunks is None and chunksize is None:
            return 1, 0
        elif total_chunks is not None and chunksize is not None:
            estimated_size = total_chunks * chunksize
        else:
            print(total_chunks, chunksize)
            estimated_size = self._estimate_size(date)
            if estimated_size is None:
                estimated_size = 1e10
            if chunksize is not None:
                total_chunks = round(estimated_size / chunksize) + 1
            elif total_chunks is not None:
                chunksize = round(estimated_size / total_chunks) + 1
        print(total_chunks, chunksize)
        return round(total_chunks), round(chunksize)

    @abstractmethod
    def _estimate_size(self, date: Optional[Union[str, dt.date]] = None) -> Optional[int]:
        raise NotImplementedError

    def _prepare_query(
        self,
        query: Union[str, ModuleType],
        date: Optional[dt.date],
    ) -> Union[str, tuple]:
        if isinstance(query, str) and '{' in query:
            common_parameters = helpers.query_parameters(date=date)
            individual_parameters = self.parameters['query_parameters']
            schedule = self.parameters['schedule']
            query = query.format(
                **common_parameters,
                **individual_parameters,
                schedule=schedule,
            )
            return query
        elif isinstance(query, ModuleType):
            arguments = []
            flags = self.parameters['flags']
            main_function_name = self.parameters['main_function']
            date_flag = flags['date']
            if isinstance(date_flag, Iterable):
                date_flag = next(iter(date_flag))
            arguments += [date_flag, str(date)]
            return (query, main_function_name, arguments)
        return query

    def _read(
        self,
        query: Union[str, tuple],
    ) -> Union[pd.DataFrame, str, bytes]:
        if self.source_db.RAW_MODE and self.db.RAW_MODE:
            return self.source_db.read_string(query)
        else:
            return self.source_db.read(query)

    def _insert(self, data: Union[pd.DataFrame, str, bytes], date: Optional[Union[str, dt.date]] = None) -> int:
        if self.source_db.RAW_MODE and self.db.RAW_MODE:
            assert isinstance(data, (str, bytes))
            format = self.parameters.get('format')
            is_file_insertion = self.parameters.get('insert_as_file', False)
            if is_file_insertion:
                rownum = self.db.insert_file(self.name, data, date, format=format)
            else:
                rownum = self.db.insert_string(self.name, data, format=format)
        else:
            rownum = self.db.insert(self.name, data)
        assert rownum is not None
        return rownum

    def _clear(self, date: Optional[dt.date]) -> None:
        if self.parameters.get('skip_cleanup', False):
            print('Skipping cleanup at all')
        elif self.is_full:
            rownum = self.db.truncate(self.name)
            print(f'Truncated {rownum} rows')
        else:
            is_file_insertion = self.parameters.get('insert_as_file', False)
            rownum = self.db.delete(self.name, date, self.parameters['date_column'])
            print(f'Deleted {rownum} {"files" if is_file_insertion else "rows"} for date {date}')

    def _find_db(self, source: bool = False) -> BaseDatabase:
        if source:
            db_class, db = self.filesystem_representation.get_source_db()
        else:
            db_class, db = self.filesystem_representation.get_db()
        params = self._get_db_params(source)
        return Database(db_class, db, **params)

    def _get_db_params(self, source: bool) -> dict:
        params = {}
        max_memory_usage_gb = self.parameters.get('max_memory_usage_gb')
        if max_memory_usage_gb is not None:
            params['max_memory_usage_gb'] = max_memory_usage_gb
        return params

    def _extract_load_dttm_from_path(
        self,
        glob_path: str,
        file_path: str,
    ) -> str:
        difference = difflib.ndiff(glob_path, file_path)
        added_list = [letter[2] for letter in difference if letter[0] == '+']
        added = ''.join(added_list)

        load_dttm = None
        if added:
            try:
                load_dttm = dt.datetime.strptime(added, '%Y-%m-%d')
            except ValueError as e:
                print(e)

        if load_dttm is None:
            load_dttm = dt.datetime.now()
        load_dttm_string = load_dttm.strftime('%Y-%m-%d %H:%M:%S')
        return load_dttm_string

    def _check_rownum(
        self,
        inserted_rownum: int,
        date: Optional[dt.date],
    ) -> None:
        actual_rownum = self.db.get_table_rownum(self.name, date, self.parameters['date_column'])
        if self.parameters.get('check_rownum', True) and inserted_rownum != actual_rownum:
            raise TableRunError(f'Actual rownum differs from inserted: {actual_rownum} vs {inserted_rownum}')

    def _shift_date(self, date: Optional[dt.date]) -> Optional[dt.date]:
        if date is not None:
            datelag = self.parameters['datelag']
            schedule = self.parameters['schedule']
            if schedule != 'day':
                date += dt.timedelta(days=1)
                datelag += 1

                if schedule == 'week':
                    datelag *= 7
                elif schedule == 'month':
                    datelag *= (
                        calendar.monthrange(date.year, date.month - 1)[1]
                        if date.month > 1
                        else calendar.monthrange(date.year - 1, 12)[1]
                    )
                else:
                    raise ValueError(f'Unknown schedule "{schedule}"')
            date -= dt.timedelta(days=datelag)
        return date
