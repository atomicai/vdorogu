from textwrap import dedent

import pytest
from lib.trgetl.detools import create
from lib.trgetl.detools.helpers import get_tables_to_import

PG_TABLES_TO_IMPORT = {
    "ch_olap.ads_budget_v",
    "ch_olap.ads_day_v",
    "ch_olap.ads_usages_stat_v",
    "ch_olap.bonus_agencies_v",
    "ch_olap.campaign_partner_share_v",
    "ch_olap.conversions_v",
    "ch_olap.main_goals_v",
    "ch_olap.merchant_day_v",
    "ch_olap.package_platform_15m_v",
    "ch_olap.pad_discounts_v",
    "ch_olap.profile_source_uniques",
    "ch_olap.project_uniques",
    "ch_olap.query_log_v",
    "ch_olap.rewarded_apps_cpa_v",
    "ch_olap.tps_day_v",
    "ch_olap.url_object_funnel_v",
    "ch_olap.url_object_id_event_v",
    "ch_olap.user_funnel_v",
    "ch_olap.visitors_ids_daily",
}


def test_drop_foreign_tables(capsys):
    etalon_queries = []
    for table in PG_TABLES_TO_IMPORT:
        etalon_queries.append(f"QUERY: DROP FOREIGN TABLE {table} cascade;")
    create._drop_foreign_tables(db="postgres", schema="ch_olap")
    out, _ = capsys.readouterr()
    result_queries = out.strip().split("\n")
    assert all(
        t in etalon_queries for t in result_queries
    ), f"should be exectly same query lists, but got {result_queries} instead of {etalon_queries}"


def test_drop_foreign_tables_another_db():
    with pytest.raises(NotImplementedError):
        create._drop_foreign_tables(db="another")


def test_import_foreign_tables(capsys):
    etalon_query = (
        dedent(
            f"""
        QUERY: IMPORT FOREIGN SCHEMA "default"
        LIMIT TO ({', '.join(get_tables_to_import())})
        FROM SERVER clickhouse_trg1_fdw_svr INTO ch_olap;
    """
        )
        .strip()
        .replace("\n", " ")
    )
    create._import_foreign_tables(db="postgres", schema="ch_olap")
    out, _ = capsys.readouterr()
    result_query = out.strip()
    assert etalon_query == result_query, "should be {etalon_query}, but got {result_query}"


def test_import_foreign_tables_another_db():
    with pytest.raises(NotImplementedError):
        create._import_foreign_tables(db="another")


def test_create_views_on_foreign_tables(capsys):
    etalon_queries = []
    for table in PG_TABLES_TO_IMPORT:
        emart_tab = "emart." + table.replace(".", "_")
        if not emart_tab.endswith("_v"):
            emart_tab += "_v"
        etalon_queries.append(f"""QUERY: CREATE OR REPLACE VIEW {emart_tab} AS ( SELECT * FROM {table} );""")
    create._create_views_on_foreign_tables(db="postgres", schema="ch_olap")
    out, _ = capsys.readouterr()
    result_queries = out.strip().split("\n")
    assert all(
        t in etalon_queries for t in result_queries
    ), f"should be exectly same query lists, but got {result_queries} instead of {etalon_queries}"


def test_create_views_on_foreign_tables_another_db():
    with pytest.raises(NotImplementedError):
        create._create_views_on_foreign_tables(db="another")


def test_postgres_tables_from():
    etalon_tables = list(PG_TABLES_TO_IMPORT)
    for table in PG_TABLES_TO_IMPORT:
        emart_tab = "emart." + table.replace(".", "_")
        if not emart_tab.endswith("_v"):
            emart_tab += "_v"
        etalon_tables.append(emart_tab)
    result_tables = create.postgres_tables_from()
    assert all(
        t in etalon_tables for t in result_tables
    ), f"should be exectly same table lists, but got {result_tables} instead of {etalon_tables}"
