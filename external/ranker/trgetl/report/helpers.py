import datetime as dt
import getpass
import time
from functools import wraps
from typing import Callable

import pandas as pd

from ..database.clickhouse import Clickhouse, ClickhouseError


def log(send_func: Callable) -> Callable:
    @wraps(send_func)
    def wrapped(self: object, report_dict: dict, ready: bool) -> None:
        start_time_sec = time.time()
        start_time = dt.datetime.now()
        start_time -= dt.timedelta(microseconds=start_time.microsecond)
        try:
            send_func(self, report_dict, ready)
            success = report_dict['check']
            error = report_dict['raw_error_msg']
        except Exception as e:
            success = False
            error = str(e)
            raise e
        finally:
            log_info = pd.DataFrame(
                dict(
                    date=[str(start_time.date())],
                    start_time=[str(start_time)],
                    name=[self.name],  # type: ignore
                    callable=[report_dict['report_name']],
                    user=[getpass.getuser()],
                    success=[success],
                    error=[error],
                    execution_time_sec=[round(time.time() - start_time_sec, 4)],
                )
            )
            olap = Clickhouse('olap')
            try:
                olap.insert('dwh.reports_log', log_info, return_rownum=False)
            except ClickhouseError as e:
                if 'Not enough privileges' in str(e):
                    print('FAILED to log report: not enough privileges')
                else:
                    raise e
            print(f'{round(time.time() - start_time_sec)} sec')

    return wrapped
