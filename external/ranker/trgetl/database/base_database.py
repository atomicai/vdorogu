import datetime as dt
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


class BaseDatabase:
    RAW_MODE = False
    DB_TITLES: Dict[str, str] = {}
    RETRY_ERRORS: Dict[str, str] = {}

    db = ""
    name = ""
    retries = 0
    retry_sleep = 0

    def __repr__(self) -> str:
        name = f"{type(self).__name__}('{self.db}')"
        return name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def hunam_readable_title(self) -> str:
        return self.DB_TITLES.get(self.db, str(self))

    def read(self, query: Union[str, tuple]) -> pd.DataFrame:
        raise NotImplementedError

    def read_string(self, query: Union[str, tuple]) -> str:
        raise NotImplementedError

    def execute(self, query: str, return_rownum: bool = False) -> str:
        raise NotImplementedError

    def insert(
        self,
        table_name: str,
        df: pd.DataFrame,
        return_rownum: bool = True,
    ) -> Optional[int]:
        raise NotImplementedError

    def insert_string(
        self,
        table_name: str,
        data: Union[str, bytes],
        return_rownum: bool = True,
        format: str = None,
    ) -> Optional[int]:
        raise NotImplementedError

    def insert_file(
        self,
        table_name: str,
        data: Union[str, bytes],
        date: Optional[Union[str, dt.date]],
        return_filenum: bool = True,
        format: str = None,
    ) -> Optional[int]:
        raise NotImplementedError

    def columns(self, table_name: str) -> list:
        raise NotImplementedError

    def truncate(self, table_name: str) -> int:
        raise NotImplementedError

    def delete(self, table_name: str, date: Optional[dt.date], date_column: str = "date") -> int:
        raise NotImplementedError

    def get_table_rownum(self, table_name: str, date: Union[dt.date, str] = None, date_column: str = "date") -> int:
        raise NotImplementedError

    def default_path(self) -> Path:
        raise NotImplementedError

    def rm(self, path: Union[str, Path]) -> int:
        raise NotImplementedError

    def save_as_file(
        self,
        query: str,
        path: Optional[Union[str, Path]] = None,
        format_: str = "orc",
        appname: Optional[str] = None,
        return_structure: bool = False,
    ) -> Optional[list]:
        raise NotImplementedError

    def hdfs_query(
        self,
        path: Union[Path, str],
        format_: str,
        structure: Optional[List[Tuple[str, str]]],
    ) -> str:
        raise NotImplementedError


def retry(function: Callable) -> Callable:
    @wraps(function)
    def wrapped(self: BaseDatabase, *args: Any, **kwargs: Any) -> None:
        retries = self.retries
        while True:
            try:
                start_time = time.time()
                return function(self, *args, **kwargs)

            except Exception as e:
                retry_errors_occured = [message for error, message in self.RETRY_ERRORS.items() if error in str(e)]
                if retries and retry_errors_occured:
                    execution_time = round(time.time() - start_time)
                    retries -= 1
                    print(f"{execution_time} sec: {retry_errors_occured[0]}, retrying...")
                    time.sleep(self.retry_sleep)
                else:
                    raise e

    return wrapped


def measure(function: Callable) -> Callable:
    @wraps(function)
    def wrapped(self: BaseDatabase, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = function(self, *args, **kwargs)
        execution_time = round(time.time() - start_time)
        size_gb = round(sys.getsizeof(result) / 2**30, 1)
        measure_report = f"{function.__name__}: {execution_time} sec"
        if size_gb >= 0.1:
            measure_report += f", {size_gb} Gb"
        print(measure_report)
        return result

    return wrapped
