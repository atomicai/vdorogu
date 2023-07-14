import pytest
from lib.trgetl.detools.helpers import DatabaseNotExists, get_tables_to_import

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


def test_get_tables_to_import():
    result = get_tables_to_import(db="postgres")
    assert (
        PG_TABLES_TO_IMPORT == result
    ), f"should be exectly same sets, but got {result} instead of {PG_TABLES_TO_IMPORT}"


def test_get_tables_to_import_another_db():
    with pytest.raises(DatabaseNotExists):
        get_tables_to_import(db="another")
