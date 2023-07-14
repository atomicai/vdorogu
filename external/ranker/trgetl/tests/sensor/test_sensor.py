import datetime as dt
from textwrap import dedent

import pytest

from ... import filesystem
from ...database import Database
from ...sensor import Sensor, SensorError, SensorNotReadyError

SENSOR_NAME = "test_target.events"
OUTBOUND_DB = [
    Database("ch", "events"),
    Database("ch", "vk"),
    Database("hdfs", None),
    Database("pg", None),
    Database("spark", None),
]


class SensorObject:
    def __init__(self, table_name, query_path, query, raw_parameters, parameters, db_type, db, repr, table_exists):
        self.table_name = table_name
        self.query_path = query_path
        self.query = query
        self.raw_parameters = raw_parameters
        self.parameters = parameters
        self.db = Database(db_type, db) if db_type else None
        self.repr = repr
        self.table_exists = table_exists


SENSORS = [
    SensorObject(
        "test_target.events",
        filesystem.SENSOR_PATH / "ch" / "events" / "test_target.events.sql",
        dedent(
            """
            select count() > 1000
            from target.events
            where date = '{date}'
        """
        ).strip(),
        dict(),
        {
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "datelag": 0,
            "as_file": False,
            "is_full": False,
        },
        "ch",
        "events",
        "TableSensor('test_target.events')",
        True,
    ),
    SensorObject(
        "test_some_vk",
        filesystem.SENSOR_PATH / "ch" / "vk" / "test_some_vk.sql",
        dedent(
            """
            select count() > 1000
            from some_vk_table
            where date = '{date}'
        """
        ).strip(),
        dict(),
        {
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "datelag": 0,
            "as_file": False,
            "is_full": False,
        },
        "ch",
        "vk",
        "TableSensor('test_some_vk')",
        False,
    ),
    SensorObject(
        "test_some_pg",
        filesystem.SENSOR_PATH / "pg" / "test_some_pg.sql",
        dedent(
            """
            select count() > 1000
            from some_pg_table
            where date = '{date}'
        """
        ).strip(),
        dict(),
        {
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "datelag": 0,
            "as_file": False,
            "is_full": False,
        },
        "pg",
        None,
        "TableSensor('test_some_pg')",
        False,
    ),
    SensorObject(
        "default.ads_clients",
        filesystem.SENSOR_PATH / "ch" / "vk" / "default.ads_clients.sql",
        dedent(
            """
            /*parameter:datelag=-1*/

            with row_cnts as (
                select
                    dt as date,
                    row_number() over(order by date) as num,
                    count() as cnt
                from ads_clients
                where date >= '{date}' - 14
                group by date
                having cnt > 1000000
            ),
            cur as (
                select num, cnt
                from row_cnts
                where date = '{date}'
            )
            select
                ifNull((select cnt from cur) / avgIf(
                    cnt, (select num from cur) - num <= 2 and (select num from cur) - num > 0
                ), 0) > 0.93 as threshold
            from
                row_cnts
        """
        ).strip(),
        {
            "datelag": -1,
        },
        {
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "datelag": -1,
            "as_file": False,
            "is_full": False,
        },
        "ch",
        "vk",
        "TableSensor('default.ads_clients')",
        True,
    ),
    SensorObject(
        "ods_rb_data.access_log",
        filesystem.SENSOR_PATH / "hdfs.json",
        "/dwh/ods/ods_rb_data.db/access_log/source=*",
        {"source": "/dwh/ods/ods_rb_data.db/access_log/source=*", "as_file": True, "datelag": 1},
        {
            "source": "/dwh/ods/ods_rb_data.db/access_log/source=*",
            "as_file": True,
            "datelag": 1,
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "is_full": False,
        },
        "hdfs",
        None,
        "FileSensor('ods_rb_data.access_log')",
        True,
    ),
    SensorObject(
        "ods_target_data.antifraud",
        filesystem.SENSOR_PATH / "hdfs.json",
        "/dwh/ods/ods_target_data.db/antifraud",
        {"source": "/dwh/ods/ods_target_data.db/antifraud", "as_file": True},
        {
            "source": "/dwh/ods/ods_target_data.db/antifraud",
            "as_file": True,
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "datelag": 0,
            "is_full": False,
        },
        "hdfs",
        None,
        "FileSensor('ods_target_data.antifraud')",
        True,
    ),
]


def id_sensor(sensors):
    return [s.table_name for s in sensors]


@pytest.fixture(params=SENSORS, ids=id_sensor(SENSORS))
def etalon_sensor(request):
    return request.param


def test_sensor_initialization_and_check(create_samples, etalon_sensor):
    if not etalon_sensor.repr:
        with pytest.raises(AttributeError):
            Sensor(etalon_sensor.table_name)
    else:
        sensor = Sensor(etalon_sensor.table_name)
        assert (
            sensor.table_name == etalon_sensor.table_name
        ), f"Sensor name should be {etalon_sensor.table_name}, but got {sensor.table_name}"
        assert isinstance(
            sensor.table_filesystem_representation, filesystem.TableInFilesystem
        ), "there are should be TableInFilesystem object"
        assert (
            sensor.query_path == etalon_sensor.query_path
        ), f"""Sensor filesystem path shoul be {
                etalon_sensor.query_path
            }, but got {sensor.query_path}"""
        assert (
            sensor.query[:20] == etalon_sensor.query[:20]
        ), f"Sensor query sould be {etalon_sensor.query}, but got {sensor.query}"
        assert (
            sensor._raw_parameters == etalon_sensor.raw_parameters
        ), f"""Sensor query sould be {
                etalon_sensor.raw_parameters
            }, but got {sensor._raw_parameters}"""
        assert (
            sensor.parameters == etalon_sensor.parameters
        ), f"Sensor query sould be {etalon_sensor.parameters}, but got {sensor.parameters}"
        assert sensor.db == etalon_sensor.db, f"Sensors db should be {etalon_sensor.db}, but got {sensor.db}"


def test_sensor_repr(create_samples, etalon_sensor):
    if not etalon_sensor.repr:
        with pytest.raises(AttributeError):
            Sensor(etalon_sensor.table_name)
    else:
        assert (
            Sensor(etalon_sensor.table_name).__repr__() == etalon_sensor.repr
        ), f"should be {etalon_sensor.repr} but got {Sensor(etalon_sensor.table_name)}"


@pytest.mark.parametrize(
    "date, etalon_date",
    [
        pytest.param("2020-08-10 8:45:10", dt.date(2020, 8, 10), id="2020-08-10 8:45:10"),
        pytest.param("2021-08-22", dt.date(2021, 8, 22), id="2021-08-22"),
        pytest.param("2021-09-09", dt.date(2021, 9, 9), id="2021-09-09"),
        pytest.param("2021-09-10", dt.date(2021, 9, 10), id="2021-09-10"),
        pytest.param(dt.date.today(), dt.date.today(), id="today_date"),
        pytest.param(dt.datetime.today(), dt.date.today(), id="today_datetime"),
    ],
)
def test_prepare_date(create_samples, etalon_sensor, date, etalon_date):
    if not etalon_sensor.repr:
        with pytest.raises(AttributeError):
            Sensor(etalon_sensor.table_name)._prepare_date(date, date_correction=False)
    else:
        result = Sensor(etalon_sensor.table_name)._prepare_date(date, date_correction=False)
        assert result == etalon_date, f"should be {etalon_date}, but got {result}"


@pytest.mark.parametrize(
    "date, etalon_date",
    [
        pytest.param("2020-08-10 8:45:10", dt.date(2020, 8, 10), id="2020-08-10 8:45:10"),
        pytest.param("2021-08-22", dt.date(2021, 8, 22), id="2021-08-22"),
        pytest.param("2021-09-09", dt.date(2021, 9, 9), id="2021-09-09"),
        pytest.param("2021-09-10", dt.date(2021, 9, 10), id="2021-09-10"),
        pytest.param(dt.date.today(), dt.date.today(), id="today_date"),
        pytest.param(dt.datetime.today(), dt.date.today(), id="today_datetime"),
    ],
)
def test_prepare_date_with_date_correction(create_samples, etalon_sensor, date, etalon_date):
    if not etalon_sensor.repr:
        with pytest.raises(AttributeError):
            Sensor(etalon_sensor.table_name)._prepare_date(date, date_correction=True)
    else:
        result = Sensor(etalon_sensor.table_name)._prepare_date(date, date_correction=True)
        datelag = etalon_sensor.parameters.get("datelag")
        if datelag:
            etalon_date -= dt.timedelta(days=datelag)
        assert result == etalon_date, f"should be {etalon_date}, but got {result}"


def sensor_run_success(table_name, date=None):
    result = Sensor(table_name).run(date)
    assert result is True, f"should be True, but got {result}"


def test_run_raises(create_samples, etalon_sensor):
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    if etalon_sensor.table_exists:
        with pytest.raises(SensorNotReadyError):
            Sensor(etalon_sensor.table_name).run(tomorrow)
    else:
        if etalon_sensor.repr:
            with pytest.raises(SensorError):
                Sensor(etalon_sensor.table_name).run(tomorrow)
        else:
            with pytest.raises(AttributeError):
                Sensor(etalon_sensor.table_name).run(tomorrow)


def test_run(create_samples, etalon_sensor):
    if etalon_sensor.table_exists and etalon_sensor.db in OUTBOUND_DB:
        sensor_run_success(etalon_sensor.table_name)
    elif etalon_sensor.table_exists:
        with pytest.raises(SensorNotReadyError):
            Sensor(etalon_sensor.table_name).run()
    elif etalon_sensor.repr:
        with pytest.raises(SensorError):
            Sensor(etalon_sensor.table_name).run()
    else:
        with pytest.raises(AttributeError):
            Sensor(etalon_sensor.table_name).run()


@pytest.mark.parametrize(
    "date",
    [
        pytest.param("2021-08-31 8:45:10", id="2021-08-31 8:45:10"),
        pytest.param("2021-08-28", id="2021-08-28"),
        pytest.param("2021-09-09", id="2021-09-09"),
        pytest.param(dt.date.today(), id="today_date"),
        pytest.param(dt.datetime.today(), id="today_datetime"),
        pytest.param(dt.date.today() + dt.timedelta(days=1), id="tomorrow_date"),
        pytest.param(dt.datetime.today() + dt.timedelta(days=1), id="tomorrow_datetime"),
    ],
)
def test_run_with_date(capsys, create_samples, etalon_sensor, date):
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    today = dt.date.today()
    if not etalon_sensor.repr:
        with pytest.raises(AttributeError):
            Sensor(etalon_sensor.table_name).run(date)
    else:
        date = Sensor(etalon_sensor.table_name)._parse_date(date)
        if etalon_sensor.table_exists:
            if date < today:
                sensor_run_success(etalon_sensor.table_name, date)
            elif date < tomorrow and etalon_sensor.db != Database("hdfs", None) and etalon_sensor.db in OUTBOUND_DB:
                sensor_run_success(etalon_sensor.table_name, date)
            else:
                with pytest.raises(SensorNotReadyError):
                    Sensor(etalon_sensor.table_name).run(date)
        else:
            with pytest.raises(SensorError):
                Sensor(etalon_sensor.table_name).run(date)
