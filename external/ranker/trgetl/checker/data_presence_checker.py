import logging

from .base_checker import BaseChecker
from .exceptions import CheckerError

logger = logging.getLogger("Checker")


class DataPresenceChecker(BaseChecker):
    def __init__(self, table_name):
        super().__init__(table_name)

    def run(self, date):
        date = self._parse_date(date)
        logger.info('Starting (date: %s):\n%s', date, repr(self))
        ddl_type = self.table_filesystem_representation.get_ddl_type()
        if ddl_type == 'file':
            logger.info('Not Implemented for type %s. Skip.', ddl_type)
            return True
        if self.table_parameters['is_full']:
            return self._check_load_dttm(date)
        else:
            return self._check_date(date)

    def _check_load_dttm(self, date):
        columns = self.db.columns(self.table_name)
        if 'load_dttm' not in columns:
            raise CheckerError(f'Full table {self.table_name} has no field load_dttm')
        query = f'select load_dttm from {self.table_name} limit 1'
        min_load_dttm = self.db.read(query)
        if min_load_dttm.shape[0] < 1:
            logger.info('FAILURE: got no load_dttm, expected at least %s', date)
            return False
        min_load_dttm = min_load_dttm.iloc[0, 0]
        load_date = self._parse_date(min_load_dttm)
        logger.info('Query:    %s', query)
        if load_date >= date:
            logger.info('Success: %s', min_load_dttm)
            return True
        else:
            logger.info('FAILURE: got load_dttm %s, expected at least %s', min_load_dttm, date)
            return False

    def _check_date(self, date):
        date_column = self.table_parameters['date_column']
        query = f"select count(*) from {self.table_name} where {date_column} = '{date}'"
        rownum = self.db.read(query)
        rownum = rownum.iloc[0, 0]
        rownum = int(rownum)
        logger.info('Query:    %s', query)
        if rownum > 0:
            logger.info('Success: %s', rownum)
            return True
        else:
            logger.info('FAILURE: got %s row numbers, expected at least 1', rownum)
            return False
