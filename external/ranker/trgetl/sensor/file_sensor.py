import datetime as dt
from typing import Optional

from ..database import Hdfs
from .base_sensor import BaseSensor
from .exceptions import SensorNotReadyError


class FileSensor(BaseSensor):
    def __init__(self, table_name: str):
        super().__init__(table_name)
        self.query = self._get_query()
        self.db = self._find_db()

    def _get_query(self) -> str:
        return self.parameters.get("source")

    def _file_consistency(self, path: str, date: dt.datetime) -> bool:
        dir_cnt = len(Hdfs().ls(path, return_directories=True))
        if self.parameters.get("ignore_date"):
            success_cnt = len(Hdfs().ls(path + "/_SUCCESS"))
        else:
            date_format = self.parameters.get("date_format", "%Y-%m-%d")
            date_partition_format = self.parameters.get("date_partition_format", "dt=")
            success_cnt = len(Hdfs().ls(path + f"/{date_partition_format}{date.strftime(date_format)}/_SUCCESS"))
        return dir_cnt == success_cnt

    def run(self, date: Optional[dt.datetime] = None, date_correction: bool = False) -> bool:
        date = self._prepare_date(date, date_correction)
        print(f"Starting (date: {date}):\n{self}")
        query = self.query.format(date=date)
        print(f"Query:    {query}")
        response = self._file_consistency(query, date)
        if response:
            print(f"Success: {response}")
            return True
        else:
            raise SensorNotReadyError(f"FAILURE: Got negative response: {response} for query: {query}")
