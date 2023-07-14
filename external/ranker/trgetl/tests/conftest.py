from pathlib import Path
from textwrap import dedent

import pytest

from .. import filesystem

TEMP_SENSORS = {
    "ch/events/test_target.events.sql": dedent(
        """\
        select count() > 1000
        from target.events
        where date = '{date}'
    """
    ),
    "ch/olap/test_default.ads_day_v.sql": dedent(
        """\
        select count() > 1000
        from default.ads_day_v
        where date = '{date}'
    """
    ),
    "ch/olap/test_default.banner_day.sql": dedent(
        """\
        select count() > 1000
        from default.banner_day
        where date = '{date}'
    """
    ),
    "ch/vk/test_some_vk.sql": dedent(
        """\
        select count() > 1000
        from some_vk_table
        where date = '{date}'
    """
    ),
    "pg/test_emart.pay_method.sql": dedent(
        """\
        select count(*) > 1000
        from emart.pay_method
        where load_dttm >= '{date}'
    """
    ),
    "pg/test_some_pg.sql": dedent(
        """\
        select count() > 1000
        from some_pg_table
        where date = '{date}'
    """
    ),
    "spark/test_some_spark.sql": dedent(
        """\
        select count() > 1000
        from some_spark_table
        where date = '{date}'
    """
    ),
}


@pytest.fixture(scope="session")
def create_samples():
    for sensor_path, query in TEMP_SENSORS.items():
        filepath = filesystem.SENSOR_PATH / Path(sensor_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as f:
            f.write(query)
    yield
    for sensor_path in TEMP_SENSORS.keys():
        filepath = filesystem.SENSOR_PATH / Path(sensor_path)
        filepath.unlink()
        for parent in filepath.parents:
            if parent == filesystem.SENSOR_PATH:
                break
            try:
                parent.rmdir()
            except Exception:
                break
