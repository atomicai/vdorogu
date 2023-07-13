import datetime as dt
import getpass
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

from ..database.clickhouse import Clickhouse, ClickhouseError
from ..filesystem import Filesystem
from .exceptions import TableRunError


def yesterday() -> dt.date:
    return dt.date.today() - dt.timedelta(days=1)


def log(function: Callable) -> Callable:
    @wraps(function)
    def wrapped(self: object, *args: Any, **kwargs: Any) -> int:
        log = True
        start_time_sec = time.time()
        start_time = dt.datetime.now()
        start_time -= dt.timedelta(microseconds=start_time.microsecond)
        try:
            rownum = function(self, *args, **kwargs)
            success = 1
            error = ''
        except Exception as e:
            rownum = 0
            success = 0
            error = str(e)
            raise e
        except KeyboardInterrupt as e:
            log = False
            raise e
        finally:
            if log:
                log_info = pd.DataFrame(
                    dict(
                        date=[str(start_time.date())],
                        start_time=[str(start_time)],
                        process=[type(self).__name__],
                        target=[self.name],  # type: ignore
                        user=[getpass.getuser()],
                        success=[success],
                        rows_inserted=[int(rownum)],
                        arguments=[f'{args} {kwargs}'],
                        error=[error.replace('\n', '    ')],
                        execution_time_sec=[round(time.time() - start_time_sec, 4)],
                    )
                )
                olap = Clickhouse('olap')
                try:
                    olap.insert('dwh.process_log', log_info, return_rownum=False)
                except ClickhouseError as e:
                    if 'Not enough privileges' in str(e):
                        print('FAILED to log procces: not enough privileges')
                    else:
                        raise e
            print(f'{round(time.time() - start_time_sec)} sec')
        return rownum

    return wrapped


def query_parameters(date: Optional[Union[dt.date, str]] = None) -> Dict[str, str]:
    """Get query parameter from database"""
    olap = Clickhouse('olap')
    parameters = {}

    parameters['date'] = '{date}'
    if date is not None:
        parameters['date'] = str(date)
        try:
            parameters['active_experiments'] = (
                olap.read(
                    f"""
                with case when date_end > '2000-01-01' then date_end
                    else toDate('{date}') end as date_end_
                select groupUniqArray((experiment_group_id_type, experiment_group_id))
                from ab.experiments
                where toDate('{date}') between date_start and date_end_
            """
                )
                .iloc[0, 0]
                .strip('[]')
            )
            if parameters['active_experiments'] == '':
                parameters['active_experiments'] = "('',-1)"
        except ClickhouseError as e:
            if 'Not enough privileges' in str(e):
                parameters['active_experiments'] = "('',-1)"
                print(
                    'FAILED to get helpers.parameter `active_experiments`: '
                    "not enough privileges to `ab` schema. Set value `('',-1)`."
                )
            else:
                raise e

    parameters['youla_packages'] = (
        olap.read(
            """
        select groupUniqArray(package_id)
        from dictionary.package
        where package_type = 'youla'
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )
    if not parameters['youla_packages']:
        raise TableRunError('youla_packages is empty')

    parameters['audio_packages'] = (
        olap.read(
            """
        select groupUniqArray(package_id)
        from dictionary.package
        where medium = 'audio'
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )

    parameters['dsp_packages'] = (
        olap.read(
            """
        select groupUniqArray(package_id)
        from dictionary.package
        where package_type like '%dsp%'
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )

    parameters['dooh_packages'] = (
        olap.read(
            """
        select groupUniqArray(package_id)
        from dictionary.package
        where package_type like '%billboards%'
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )

    parameters['delivery_packages'] = (
        olap.read(
            """
        select groupUniqArray(package_id)
        from dictionary.package
        where package_type like '%delivery%'
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )

    parameters['other_packages'] = (
        olap.read(
            """
        select groupUniqArray(package_id)
        from dictionary.package
        where model not in ('ocpm','cpc','cpm','cpi')
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )

    parameters['cpm_packages'] = (
        olap.read(
            """
        select groupUniqArray(package_id)
        from dictionary.package
        where model in ('cpm')
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )

    parameters['dzen_banner_ids'] = (
        olap.read(
            """
        select groupUniqArray(banner_id)
        from dump.rb_banner_types
        where group_id in (select profile_id from dump.rb_banners_group where type_id = 1135161)
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )

    parameters['dzen_pad_ids'] = (
        olap.read(
            """
        select groupUniqArray(pad_id)
        from dictionary.pad
        where name_parent_1c like ('%Дзен%')
    """
        )
        .iloc[0, 0]
        .strip('[]')
    )

    instream_packages = [
        405,
        544,
        545,
        556,
        558,
        559,
        563,
        644,
        698,
        703,
        704,
        720,
        756,
        757,
        758,
        780,
        860,
        877,
        878,
        879,
        880,
        917,
        923,
        940,
        941,
        943,
        1059,
        1060,
        1180,
        1209,
        1210,
        1227,
        1232,
        1256,
        1260,
        1265,
        1315,
        1316,
        1317,
        1378,
        1413,
        1421,
        1430,
        1461,
        1481,
        1504,
        1505,  # https://jira.vk.team/browse/TRG-48673
        1376,  # https://jira.vk.team/browse/TRG-50373
        721,
        1436,  # https://jira.vk.team/browse/TRG-52152
        284,
        285,
        1752,
        1753,
        1896,
        1898,
        1902,  # https://jira.vk.team/browse/TRG-53959
    ]
    parameters['instream_packages'] = ','.join([str(e) for e in instream_packages])

    parameters['pad_project'] = (Filesystem.read_misc_query('pad_project')).strip('\t\n ')
    parameters['ios_mobile_app_re'] = r'id\\d{5,}$'

    parameters['total_chunks'] = '{total_chunks}'
    parameters['chunk'] = '{chunk}'
    parameters['chunksize'] = '{chunksize}'

    parameters['curly_braces'] = '{}'
    parameters['left_cb'] = '{'
    parameters['right_cb'] = '}'

    advertiser_id = [
        4819958,
        11030260,
        10711735,
        9155455,
        11674194,
        11510358,
        10076536,
        9335027,
        9837476,
        11579902,
        11292272,
        9337306,
        11281726,
        11903515,
        9918291,
        10967763,
        11677214,
        10844935,
        11072300,
        10131020,
        10419255,
        11524622,
        12065833,
        11195366,
        1030344,
        11501852,
        9437782,
    ]
    parameters['advertiser_id'] = ','.join([str(e) for e in advertiser_id])

    return parameters


def to_start_of_interval(date: Union[dt.date, str], interval: str) -> dt.date:
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)

    if interval == 'month':
        date = date.replace(day=1)
    elif interval == 'week':
        date -= dt.timedelta(days=date.weekday())
    else:
        raise ValueError(f'Unknown interval: "{interval}"')

    return date
