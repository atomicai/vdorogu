import datetime as dt

import dateutil

from ..database import Database
from ..filesystem import TableInFilesystem


class BaseChecker:
    def __init__(self, table_name, **parameters):
        self.table_name = table_name
        self.table_filesystem_representation = TableInFilesystem(table_name)
        self.table_parameters = self.table_filesystem_representation.get_standart_parameters()
        db_params = self.table_filesystem_representation.get_db()
        self.db = Database(*db_params)
        self._raw_parameters = parameters
        self.parameters = self._get_standart_parameters(**parameters)

    def __repr__(self):
        name = (f"{type(self).__name__}('{self.table_name}') " + self.parameters.get("checker_name", "")).strip()
        return name

    @classmethod
    def _parse_date(cls, date, cut_time=True):
        if not isinstance(date, dt.date):
            date = dateutil.parser.parse(date)
        if cut_time:
            if isinstance(date, dt.datetime):
                date = date.date()
        return date

    def _get_standart_parameters(self, **raw_checker_parameters):
        checker_parameters = raw_checker_parameters.copy()
        return checker_parameters
