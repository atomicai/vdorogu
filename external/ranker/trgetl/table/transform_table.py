import datetime as dt
from importlib import machinery
from types import ModuleType
from typing import Optional, Union

from .daily_table import DailyTable


class TransformTable(DailyTable):
    def __init__(self, name: str, is_full: Optional[bool] = False):
        super().__init__(name)
        self.query = self._get_query()

    def _estimate_size(self, date: Optional[Union[str, dt.date]] = None) -> Optional[int]:
        if self.is_full:
            rownum = self.db.read(f"select count(*) as cnt from {self.name}")
        else:
            date_column = self.parameters["date_column"]
            schedule = self.parameters["schedule"]
            rownum = self.db.read(
                f"""
                select case when count(distinct {date_column}) = 0 then 0
                    else count(*) / count(distinct {date_column})
                    end as cnt
                from {self.name}
                where {date_column} >= cast('{date}' as date) - interval 7 {schedule}
            """
            )
        rownum = rownum.astype(int).iloc[0, 0]
        if rownum == 0:
            return None
        else:
            return int(rownum)

    def _get_query(self) -> Union[str, ModuleType]:
        query_path = self.filesystem_representation.dataflow_path()
        if query_path.suffix == ".py":
            module_loader = machinery.SourceFileLoader(self.name, str(query_path))
            return module_loader.load_module()
        return self.filesystem_representation.dataflow_path().read_text()

    def _get_chunk_query(self, query: Union[str, tuple], chunk: int, total_chunks: int, chunksize: int) -> str:
        chunked_query = ""
        if isinstance(query, tuple):
            raise NotImplementedError
        elif isinstance(query, str):
            chunked_query = query.format(chunk=chunk, total_chunks=total_chunks, chunksize=chunksize)
        return chunked_query

    def _file_upload(self, date: Optional[dt.date], query: Union[str, tuple]) -> int:
        dir_path = self.source_db.default_path() / f"{self.name}.dt={date}"
        format_ = "orc"

        self.source_db.rm(dir_path)
        if isinstance(query, tuple):
            raise NotImplementedError
        elif isinstance(query, str):
            structure = self.source_db.save_as_file(
                query,
                dir_path,
                format_,
                appname=self.name,
                return_structure=True,
            )

        file_path = dir_path / "part-*"
        hdfs_query = self.db.hdfs_query(path=file_path, format_=format_, structure=structure)
        insert_query = f"insert into {self.name} select * from {hdfs_query}"

        self._clear(date)
        rownum = 0
        response = self.db.execute(insert_query, return_rownum=True)
        try:
            rownum = int(response)
        except ValueError:
            print(f"Could not convert response: {response.__repr__()} to Int")
        self.source_db.rm(dir_path)
        return rownum
