from .daily_table import DailyTable
from .exceptions import TableRunError


class DumpTable(DailyTable):
    def __init__(self, name, is_full=False):
        super().__init__(name)
        self.query = self._get_query()

    def _get_query(self):
        source = self.parameters["source"]
        if self.parameters["as_file"]:
            return source

        current_load_dttm = self.is_full and (
            self.source_db.db != "olap" or "load_dttm" not in self.source_db.columns(source)
        )
        if current_load_dttm:
            now_query = self.source_db.now_query()
            columns = f"{now_query} as load_dttm, *"
        else:
            columns = "*"

        query = f"select {columns} from {source}"
        if not self.is_full:
            date_column = self.parameters["source_date_column"]
            query += f" where {date_column} = '{{date}}'"
        return query

    def _estimate_size(self, date=None):
        source_table_name = self.parameters["source"]
        query = f"select count(*) as cnt from {source_table_name}"
        if not self.is_full:
            date_column = self.parameters["source_date_column"]
            query += f" where {date_column} = '{date}'"
        rownum = self.source_db.read(query)
        rownum = rownum.astype(int).iloc[0, 0]
        if rownum == 0 and not self.parameters["allow_zero"]:
            raise TableRunError("Trying to dump 0 rows")
        else:
            return int(rownum)

    def _get_chunk_query(self, query, chunk, total_chunks, chunksize):
        limit = chunksize
        offset = chunk * chunksize
        query += f" limit {limit} offset {offset}"
        return query

    def _file_upload(self, date, path):
        path = path.format(date=date)
        files = self.source_db.ls(path)
        latest_file = sorted(files)[-1]

        parse_dates = self.parameters.get("parse_dates")
        df = self.source_db.read(latest_file, parse_dates=parse_dates)

        if "load_dttm" not in df.columns:
            load_dttm = self._extract_load_dttm_from_path(path, latest_file)
            df.insert(0, "load_dttm", load_dttm)
        self._clear(date)
        rownum = self.db.insert(self.name, df)

        return rownum
