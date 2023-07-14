try:
    from tqdm.notebook import tqdm
except ImportError:
    tqdm = iter

import datetime as dt
from typing import Union

from ..checker import Checker, CheckerFailedError
from ..table import Table
from .graph import DependencyGraph
from .helpers import myteam_report


@myteam_report
def date_range(
    table_name: str,
    start_date: Union[dt.date, str],
    end_date: Union[dt.date, str],
    check: bool = True,
    precheck: bool = False,
    retries: int = 3,
    in_public: bool = False,
) -> None:
    table = Table(table_name)
    checker = Checker(table_name)
    if in_public:
        new_name = "public." + table_name.split(".")[-1]
        table.name = new_name
        checker.table_name = new_name
    assert not table.is_full, "date range works for not full tables"

    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    if start_date < end_date:
        start_date, end_date = end_date, start_date
    total_days = (start_date - end_date).days + 1

    def run(date: dt.date) -> None:
        current_retry = retries
        while current_retry:
            current_retry -= 1
            try:
                table.run(date),
                if check:
                    print()
                    checker.run(date)
                print("\n")
                return
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                if current_retry:
                    print(f"ERROR: {e}")
                else:
                    raise e

    for day in tqdm(range(total_days)):
        date = start_date - dt.timedelta(days=day)
        if precheck:
            try:
                checker.run(date)
            except CheckerFailedError:
                run(date)
        else:
            run(date)


@myteam_report
def tables(date: Union[str, dt.date]) -> list:
    if isinstance(date, str):
        date = dt.datetime.strptime(date, "%Y-%m-%d").date()

    tables = DependencyGraph().all_live_tables()
    full_tables = [tname for tname in tqdm(tables) if not getattr(Table(tname), "is_full", False)]
    to_reload = []

    for tname in tqdm(full_tables):
        checker = Checker(tname)
        try:
            checker.run(date)
        except CheckerFailedError:
            to_reload.append(tname)
        print("\n")

    failed = []
    for tname in tqdm(to_reload):
        table = Table(tname)
        checker = Checker(tname)
        try:
            table.run(date)
            checker.run(date)
        except Exception:
            failed.append(tname)
        print("\n")

    return failed
