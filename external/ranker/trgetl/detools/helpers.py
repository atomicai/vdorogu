import datetime as dt
import time
from argparse import ArgumentParser
from functools import wraps
from typing import List, Optional

from ..filesystem import FilesystemError
from ..sender import MyteamBot

PG_TABLES_TO_IMPORT = {
    "ads_budget_v",
    "ads_day_v",
    "ads_usages_stat_v",
    "bonus_agencies_v",
    "campaign_partner_share_v",
    "conversions_v",
    "main_goals_v",
    "merchant_day_v",
    "package_platform_15m_v",
    "pad_discounts_v",
    "profile_source_uniques",
    "project_uniques",
    "query_log_v",
    "rewarded_apps_cpa_v",
    "tps_day_v",
    "url_object_funnel_v",
    "url_object_id_event_v",
    "user_funnel_v",
    "visitors_ids_daily",
}
DB_TABLES = {"postgres": PG_TABLES_TO_IMPORT}


def myteam_report(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        start = time.time()
        try:
            return function(*args, **kwargs)
        finally:
            runtime = round(time.time() - start)
            try:
                MyteamBot().send(f"calculation finished, {runtime} sec runtime")
            except FilesystemError as e:
                print("Failed to send ping to myteam")
                print(e)

    return wrapped


def get_tables_to_import(db="postgres"):
    tables = DB_TABLES.get(db)
    if tables is not None:
        return tables
    else:
        raise DatabaseNotExists(f"Database {db} not allowed to be imported to")


class DatabaseNotExists(Exception):
    pass


def cli_get_date_param(table_name: str, raw_arguments: Optional[List[str]] = None) -> str:
    parser = ArgumentParser(
        prog=table_name,
        description="Script for daily loading table {}".format(print()),
    )

    parser.add_argument(
        "-d",
        "--date",
        metavar="YYYY-MM-DD",
        default=(dt.date.today() - dt.timedelta(1)).strftime("%Y-%m-%d"),
        help="Loading date",
    )

    return parser.parse_args(raw_arguments).date
