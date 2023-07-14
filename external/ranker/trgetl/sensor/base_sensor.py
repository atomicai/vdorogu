import datetime as dt

import dateutil

from ..database import Database
from ..filesystem import TableInFilesystem


class BaseSensor:
    def __init__(self, table_name):
        self.table_name = table_name
        self.table_filesystem_representation = TableInFilesystem(table_name)
        self.query_path = self._get_sensor_path()
        self._raw_parameters = self.table_filesystem_representation.get_parameters()
        self.parameters = self.table_filesystem_representation.get_standart_parameters()
        self.db = self._find_db()

    def _get_sensor_path(self):
        return self.table_filesystem_representation.sensor_path()

    def _get_sensor_db(self):
        return self.table_filesystem_representation.get_sensor_db()

    def _find_db(self):
        db_type, db = self._get_sensor_db()
        if db_type is None:
            return None
        return Database(db_type, db)

    def _parse_date(self, date, cut_time=True):
        if not isinstance(date, dt.date):
            date = dateutil.parser.parse(date)
        if cut_time:
            if isinstance(date, dt.datetime):
                date = date.date()
        return date

    def _correct_date(self, date):
        datelag = self.parameters["datelag"]
        date -= dt.timedelta(days=datelag)
        return date

    def _default_date(self):
        date = dt.date.today() - dt.timedelta(days=1)
        return date

    def _prepare_date(self, date, date_correction) -> dt.datetime:
        if date is None:
            date_correction = True
            date = self._default_date()
        date = self._parse_date(date)
        if date_correction:
            date = self._correct_date(date)
        return self._parse_date(date)

    def __repr__(self):
        return f"{type(self).__name__}('{self.table_name}')"
