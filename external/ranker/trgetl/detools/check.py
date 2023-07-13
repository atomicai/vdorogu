from matplotlib import pyplot as plt

try:
    from tqdm.notebook import tqdm
except ImportError:
    tqdm = iter

import datetime as dt
import logging
from typing import Dict, List, Union

import pandas as pd

from ..checker import Checker, CheckerFailedError, ComparisonChecker, DataPresenceChecker
from ..filesystem import Filesystem
from .helpers import myteam_report

DEFAULT_PLOT_SIZE = (12, 8)
DEFAULT_XTICKS_ROTATION = 20


logger = logging.getLogger("Checker")


@myteam_report
def date_range(
    table_name: str,
    start_date: Union[str, dt.date],
    end_date: Union[str, dt.date],
    plot: bool = False,
    silent: bool = False,
) -> Union[pd.DataFrame, List[dt.date]]:
    checkers = Checker(table_name).checker_pool
    if silent:
        logger.setLevel(logging.WARNING)
    _unskip_checkers(checkers)
    if plot:
        _mark_checkers_plot(checkers)

    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    if start_date < end_date:
        start_date, end_date = end_date, start_date
    total_days = (start_date - end_date).days + 1

    result_df = _form_report_df(checkers, start_date, total_days)

    if plot:
        for checker in checkers:
            _plot_checker(checker, result_df)
        return result_df
    elif not result_df.empty:
        list(result_df[~result_df['is_ok']]['date'].values)
    return list()


def _unskip_checkers(checker_pool: List[Union[ComparisonChecker, DataPresenceChecker]]) -> None:
    for checker in checker_pool:
        checker.parameters['skip'] = False


def _mark_checkers_plot(checker_pool: List[Union[ComparisonChecker, DataPresenceChecker]]) -> None:
    for checker in checker_pool:
        if isinstance(checker, ComparisonChecker):
            checker.parameters['plot'] = True


def _form_report_df(
    checker_pool: List[Union[ComparisonChecker, DataPresenceChecker]],
    start_date: dt.date,
    total_days: int,
) -> pd.DataFrame:
    report_dict: Dict[str, List] = dict(
        date=[],
        checker_name=[],
        self_control_value=[],
        source_control_value=[],
        is_ok=[],
    )
    for day in tqdm(range(total_days)):
        date = start_date - dt.timedelta(days=day)
        for checker in checker_pool:
            params = checker.parameters
            result = checker.run(date)
            logger.info('\n')
            report_dict['date'].append(date)
            report_dict['checker_name'].append(params.get('checker_name', id(checker)))
            report_dict['is_ok'].append(result)
            if isinstance(checker, ComparisonChecker):
                report_dict['self_control_value'].append(
                    checker.self_plot_metric if checker.self_plot_metric else checker.self_control_value
                )
                report_dict['source_control_value'].append(
                    checker.source_plot_metric if checker.source_plot_metric else checker.source_control_value
                )
            else:
                report_dict['self_control_value'].append(None)
                report_dict['source_control_value'].append(None)

    return pd.DataFrame(report_dict)


def _plot_checker(
    checker: Union[ComparisonChecker, DataPresenceChecker],
    result_df: pd.DataFrame,
) -> None:
    params = checker.parameters
    if params.get('plot', False):
        params = checker.parameters
        checker_name = params.get('checker_name', id(checker))
        slice_df = result_df[result_df['checker_name'] == checker_name]
        self_plot_type = getattr(plt, params.get('plot_plot_type', 'plot'))
        source_plot_type = getattr(plt, params.get('plot_source_plot_type', 'plot'))

        plt.figure(figsize=params.get('plot_figure_size', DEFAULT_PLOT_SIZE))
        if 'plot_metric' not in params or ('plot_metric' in params and 'plot_source_metric' in params):
            source_plot_type(
                slice_df['date'],
                slice_df['source_control_value'],
                label='source: '
                + (
                    params['plot_source_metric']
                    if 'plot_source_metric' in params
                    else params.get('source_metric', 'control_value')
                ),
                **params.get('plot_source_style', dict()),
            )
        self_plot_type(
            slice_df['date'],
            slice_df['self_control_value'],
            label='self: '
            + (params['plot_metric'] if 'plot_metric' in params else params.get('metric', 'control_value')),
            **params.get('plot_style', dict()),
        )
        plt.title(checker_name)
        plt.xticks(rotation=params.get('plot_xticks_rotation', DEFAULT_XTICKS_ROTATION))
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()


@myteam_report
def tables(date: Union[str, dt.date]) -> List[str]:
    if isinstance(date, str):
        date = dt.datetime.strptime(date, '%Y-%m-%d').date()

    tables = Filesystem.all_live_tables()
    failed = []

    for tname in tqdm(tables):
        checker = Checker(tname)
        try:
            checker.run(date)
        except CheckerFailedError:
            failed.append(tname)
        logger.info('\n')

    return failed
