import datetime as dt
import getpass
import logging
import sys

import pandas as pd

from ..database.clickhouse import Clickhouse, ClickhouseError
from ..filesystem import Filesystem, TableInFilesystem
from .base_checker import BaseChecker
from .comparison_checker import ComparisonChecker
from .data_presence_checker import DataPresenceChecker
from .exceptions import CheckerFailedError

logger = logging.getLogger("Checker")


class Checker:
    def __init__(self, table_name):
        self.table_name = table_name
        self.table_filesystem_representation = TableInFilesystem(table_name)
        self.table_parameters = self.table_filesystem_representation.get_standart_parameters()

        self.checker_pool = self._get_checker_pool()
        self._setup_logging()

    def __repr__(self):
        name = "Checkers:\n" + '\n'.join(str(checker) for checker in self.checker_pool)
        return name

    def _setup_logging(self) -> None:
        stream_formatter = logging.Formatter(
            fmt="%(message)s",
        )
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(stream_formatter)

        # file_formatter = logging.Formatter(
        #     fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        # )
        # file_handler = logging.FileHandler(filename='checker.log')
        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(file_formatter)

        logger = logging.getLogger("Checker")
        logger.setLevel(logging.DEBUG)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(stream_handler)
        # logger.addHandler(file_handler)
        logger.propagate = False

    def run(self, date=None, date_correction=False):
        date = self._prepare_date(date, date_correction)
        failed_checkers = []
        for checker in self.checker_pool:
            result = checker.run(date)
            if not result:
                failed_checkers.append(checker)
            logger.info('\n')
        self._save_checker_status(date, failed_checkers)
        if failed_checkers == []:
            logger.info('Check is successfull')
            return
        else:
            failed_checkers = [str(checker) for checker in failed_checkers]
            raise CheckerFailedError("Failed checkers:\n{}".format('\n'.join((failed_checkers))))

    def _get_checker_pool(self):
        checker_pool = []
        default_checkers = [
            self._data_presence_checker(),
            self._dump_size_checker(),
        ]
        default_checkers = [checker for checker in default_checkers if checker is not None]
        checker_pool += default_checkers
        checker_pool += self._custom_checkers()
        return checker_pool

    def _data_presence_checker(self):
        if not self.table_parameters['allow_zero'] and not self.table_filesystem_representation.is_sensor():
            return DataPresenceChecker(self.table_name)

    def _dump_size_checker(self):
        if self.table_filesystem_representation.get_type() in ('dump', 'dump-full'):
            if not self.table_parameters['as_file']:
                source = self.table_parameters['source']
                source_is_foreign_table = (
                    source in Filesystem.all_tables() and TableInFilesystem(source).get_ddl_type() == 'foreign_table'
                )
                if not source_is_foreign_table:
                    source_db = self.table_filesystem_representation.get_source_db()
                    return ComparisonChecker(
                        self.table_name,
                        metric='count(*)',
                        source_table_name=source,
                        source_db=source_db,
                        checker_name='dump_size',
                    )

    def _custom_checkers(self):
        checkers_params = self.table_filesystem_representation.get_checkers()
        checkers = [ComparisonChecker(self.table_name, **params) for params in checkers_params]
        return checkers

    def default_date(self):
        return self._prepare_date(date=None, date_correction=True)

    def _prepare_date(self, date, date_correction):
        if date is None:
            date_correction = True
            date = self._default_date()
        date = BaseChecker._parse_date(date)
        if date_correction:
            date = self._correct_date(date)
        return date

    def _default_date(self):
        date = dt.date.today() - dt.timedelta(days=1)
        return date

    def _correct_date(self, date):
        is_full = self.table_parameters['is_full']
        datelag = self.table_parameters['datelag']
        if is_full:
            date += dt.timedelta(days=1)
        date -= dt.timedelta(days=datelag)
        return date

    def _save_checker_status(self, date, failed_checkers):
        is_full = self.table_parameters['is_full']
        if is_full:
            date = '1970-01-01'
        error = len(failed_checkers) > 0
        check_time = dt.datetime.now()
        check_time = check_time - dt.timedelta(microseconds=check_time.microsecond)
        log_info = pd.DataFrame(
            dict(
                date=[str(date)],
                table_name=[self.table_name],
                is_full=[int(is_full)],
                error=[int(error)],
                check_time=[str(check_time)],
                user=[getpass.getuser()],
            )
        )
        olap = Clickhouse('olap')
        try:
            olap.execute(
                "alter table dwh.checker_status "
                "delete where table_name = '{table_name}' and date = '{date}'".format(
                    table_name=self.table_name.replace(r"'", r"\'"),
                    date=date,
                )
            )
            olap.insert('dwh.checker_status', log_info, return_rownum=False)
        except ClickhouseError as e:
            if 'Not enough privileges' in str(e):
                logger.error('FAILED to save checker result: not enough privileges')
            else:
                raise e
