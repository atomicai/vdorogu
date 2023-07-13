from pathlib import Path
from typing import Any, Dict, List

import pytest
from mock import patch  # type: ignore
from mock.mock import MagicMock  # type: ignore

from ... import filesystem

REPO_PATH = Path(__file__).parent.parent.parent.parent.parent
REPORT_SAMPLES_PATH = Path(__file__).parent.parent / "report" / "report_samples"
SENSOR_NAME = "test_target.events"
TABLE_NAME = "default.banner_day"
VIEW_NAME = "default.ads_day_v"
TABLE_DEPENDENCIES = {"default.banner_pad_day_raw"}
STANDARD_PARAMETERS = {
    "schedule": "day",
    "date_column": "date",
    "source_date_column": "date",
    "query_parameters": {},
    "allow_zero": False,
    "datelag": 0,
    "as_file": False,
    "is_full": False,
}
TABLE_DATAFLOW_PATH = filesystem.DATAFLOW_PATH / "transform" / "ch" / "olap" / (TABLE_NAME + ".sql")
TABLE_DDL_PATH = filesystem.DDL_PATH / "ch" / "table" / TABLE_NAME.replace(".", "/")
VIEW_DATAFLOW_PATH = filesystem.DDL_PATH / "ch" / "view" / VIEW_NAME.replace(".", "/") / "0002.sql"
VIEW_DDL_PATH = filesystem.DDL_PATH / "ch" / "view" / VIEW_NAME.replace(".", "/")
MISC_QUERY_PATH = filesystem.MISC_PATH / "pad_project.sql"
TE_SENSOR_PATH = filesystem.SENSOR_PATH / "ch" / "events" / (SENSOR_NAME + ".sql")
SCHEMA_LIST = [
    pytest.param(("ch", "events"), id="ch.events"),
    pytest.param(("ch", "olap"), id="ch.olap"),
    pytest.param(("ch", "vk"), id="ch.vk"),
]
DB_TYPE_LIST = [
    pytest.param("ch", id="ch"),
    pytest.param("hdfs", id="hdfs"),
    pytest.param("pg", id="pg"),
    pytest.param("spark", id="spark"),
]
DB_LIST = [
    pytest.param("events", id="events"),
    pytest.param("olap", id="olap"),
    pytest.param("vk", id="vk"),
]


class FilesystemObject:
    def __init__(
        self,
        name,
        obj_type,
        obj_paths,
        obj_db_type,
        obj_db,
        obj_source_type,
        obj_source_db,
        obj_sensor_type,
        obj_sensor_db,
        is_view,
        is_sensor,
        dependencies,
        raw_parameters,
        parameters,
    ):
        self.name = name
        self.obj = filesystem.TableInFilesystem(name)
        self.type = obj_type
        self.paths = obj_paths
        self.schema = obj_db_type, obj_db
        self.source_schema = obj_source_type, obj_source_db
        self.sensor_schema = obj_sensor_type, obj_sensor_db
        self.is_view = is_view
        self.is_sensor = is_sensor
        self.dependencies = dependencies
        self.raw_parameters = raw_parameters
        self.parameters = parameters


FILESYSTEM_OBJECTS = [
    FilesystemObject(
        "test",
        None,
        [filesystem.REPORT_PATH / "test.py"],
        None,
        None,
        "",
        None,
        "",
        None,
        False,
        False,
        set(),
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "test_target.events",
        "sensor",
        [filesystem.SENSOR_PATH / "ch" / "events" / "test_target.events.sql"],
        None,
        None,
        "ch",
        "events",
        "ch",
        "events",
        False,
        True,
        set(),
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "test_some_vk",
        "sensor",
        [filesystem.SENSOR_PATH / "ch" / "vk" / "test_some_vk.sql"],
        None,
        None,
        "ch",
        "vk",
        "ch",
        "vk",
        False,
        True,
        set(),
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "test_some_pg",
        "sensor",
        [filesystem.SENSOR_PATH / "pg" / "test_some_pg.sql"],
        None,
        None,
        "pg",
        None,
        "pg",
        None,
        False,
        True,
        set(),
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "test_some_spark",
        "sensor",
        [filesystem.SENSOR_PATH / "spark" / "test_some_spark.sql"],
        None,
        None,
        "spark",
        None,
        "spark",
        None,
        False,
        True,
        set(),
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "default.banner_day",
        "transform",
        [
            filesystem.DDL_PATH / "ch" / "table" / TABLE_NAME.replace(".", "/"),
            filesystem.DATAFLOW_PATH / "transform" / "ch" / "olap" / (TABLE_NAME + ".sql"),
        ],
        "ch",
        None,
        "ch",
        "olap",
        "ch",
        "olap",
        False,
        False,
        {"default.banner_pad_day_raw"},
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "emart.pay_method",
        "transform-full",
        [
            filesystem.DDL_PATH / "pg" / "table" / "emart.pay_method.sql",
            filesystem.DATAFLOW_PATH / "transform-full" / "pg" / "emart.pay_method.sql",
        ],
        "pg",
        None,
        "pg",
        None,
        "pg",
        None,
        False,
        False,
        {
            "emart.transactions_daily",
            "stg.accounts_dump",
            "stg.currency_target_dump",
            "stg.pay_method_dump",
        },
        dict(),
        {
            "schedule": "day",
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "datelag": 0,
            "as_file": False,
            "is_full": True,
        },
    ),
    FilesystemObject(
        "default.ads_day_v",
        "view",
        [
            filesystem.DDL_PATH / "ch" / "view" / VIEW_NAME.replace(".", "/"),
            filesystem.DDL_PATH / "ch" / "view" / VIEW_NAME.replace(".", "/") / "0002.sql",
        ],
        "ch",
        None,
        "",
        None,
        "ch",
        None,
        True,
        False,
        set(),
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "default.trgui_users",
        "transform",
        [
            filesystem.DDL_PATH / "ch" / "table" / "default" / "trgui_users",
            filesystem.DATAFLOW_PATH / "transform" / "ch" / "dbhistory" / "default.trgui_users.sql",
        ],
        "ch",
        None,
        "ch",
        "dbhistory",
        "ch",
        "dbhistory",
        False,
        False,
        set(),
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "vk.ads_clients",
        "transform",
        [
            filesystem.DDL_PATH / "ch" / "table" / "vk" / "ads_clients",
            filesystem.DATAFLOW_PATH / "transform" / "ch" / "vk" / "vk.ads_clients.sql",
            filesystem.SENSOR_PATH / "ch" / "vk" / "default.ads_clients",
        ],
        "ch",
        None,
        "ch",
        "vk",
        "ch",
        "vk",
        False,
        False,
        {"default.ads_clients"},
        {"chunksize": 4000000, "datelag": 0},
        {
            "chunksize": 4000000,
            "datelag": 0,
            "schedule": "day",
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "as_file": False,
            "is_full": False,
        },
    ),
    FilesystemObject(
        "raw.user_alive_yesterday",
        "transform-full",
        [
            filesystem.DDL_PATH / "ch" / "table" / "raw" / "user_alive_yesterday",
            filesystem.DATAFLOW_PATH / "transform-full" / "ch" / "events" / "raw.user_alive_yesterday.sql",
        ],
        "ch",
        None,
        "ch",
        "events",
        "ch",
        "events",
        False,
        False,
        {"target.events"},
        dict(),
        {
            "schedule": "day",
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "datelag": 0,
            "as_file": False,
            "is_full": True,
        },
    ),
    FilesystemObject(
        "default.affiliate_offer",
        "transform-full",
        [
            filesystem.DDL_PATH / "ch" / "table" / "default" / "affiliate_offer",
            filesystem.DATAFLOW_PATH / "transform-full" / "ch" / "olap" / "default.affiliate_offer.sql",
        ],
        "ch",
        None,
        "ch",
        "olap",
        "ch",
        "olap",
        False,
        False,
        {"dump.accounts", "dump.offer"},
        dict(),
        {
            "schedule": "day",
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "datelag": 0,
            "as_file": False,
            "is_full": True,
        },
    ),
    FilesystemObject(
        "default.project_amount_report_day",
        "transform",
        [
            filesystem.DDL_PATH / "ch" / "table" / "default" / "project_amount_report_day",
            filesystem.DATAFLOW_PATH / "transform" / "ch" / "olap" / "default.project_amount_report_day.sql",
        ],
        "ch",
        None,
        "ch",
        "olap",
        "ch",
        "olap",
        False,
        False,
        {
            "default.banner_pad_day_raw",
            "dump.rb_banner_types",
            "default.rb_creative_templates_dump",
            "dictionary.pad",
            "default.banners_compile_hist",
        },
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "default.banners_compile_hist",
        "sensor",
        [
            filesystem.DDL_PATH / "ch" / "table" / "default" / "banners_compile_hist",
            filesystem.SENSOR_PATH / "ch" / "olap" / "default.banners_compile_hist.sql",
        ],
        "ch",
        None,
        "ch",
        "olap",
        "ch",
        "olap",
        False,
        True,
        set(),
        dict(),
        STANDARD_PARAMETERS,
    ),
    FilesystemObject(
        "default.ads_clients",
        "sensor",
        [filesystem.SENSOR_PATH / "ch" / "vk" / "default.ads_clients.sql"],
        None,
        None,
        "ch",
        "vk",
        "ch",
        "vk",
        False,
        True,
        set(),
        {"datelag": -1},
        {
            "datelag": -1,
            "schedule": "day",
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "as_file": False,
            "is_full": False,
        },
    ),
    FilesystemObject(
        "ods_rb_data.access_log",
        "sensor",
        [filesystem.SENSOR_PATH / "hdfs.json"],
        None,
        None,
        "hdfs",
        None,
        "hdfs",
        None,
        False,
        True,
        set(),
        {
            "source": "/dwh/ods/ods_rb_data.db/access_log",
            "as_file": True,
            "datelag": 1,
            "ignore_date": True,
        },
        {
            "source": "/dwh/ods/ods_rb_data.db/access_log",
            "as_file": True,
            "datelag": 1,
            "ignore_date": True,
            "schedule": "day",
            "date_column": "date",
            "source_date_column": "date",
            "query_parameters": {},
            "allow_zero": False,
            "is_full": False,
        },
    ),
    FilesystemObject(
        "default.dummy", None, [], None, None, "", None, "", None, False, False, set(), dict(), STANDARD_PARAMETERS
    ),
]


def id_oject(objects):
    return [f"{o.type}({o.name})" for o in objects]


@pytest.fixture(params=FILESYSTEM_OBJECTS, ids=id_oject(FILESYSTEM_OBJECTS))
def filesystem_object(request):
    return request.param


@pytest.mark.parametrize(
    "result, etalon_answer",
    [
        pytest.param(filesystem.REPORT_PATH, REPO_PATH / "common" / "trgetl" / "report", id="report"),  # noqa
        pytest.param(filesystem.SENSOR_PATH, REPO_PATH / "common" / "trgetl" / "sensor", id="sensor"),  # noqa
        pytest.param(filesystem.DATAFLOW_PATH, REPO_PATH / "common" / "trgetl" / "data-flow", id="data-flow"),  # noqa
        pytest.param(filesystem.DDL_PATH, REPO_PATH / "common" / "trgetl" / "ddl", id="ddl"),  # noqa
    ],
)
def test_get_filesystem_path(result: str, etalon_answer: str) -> None:
    assert etalon_answer == result, f"should be {etalon_answer}, but got {result}"


@pytest.mark.parametrize(
    "name, ddl_path, etalon_answer",
    [
        pytest.param("view_name", filesystem.DDL_PATH / "ch" / "view" / "view_name", "view", id="view"),
        pytest.param("table_name", filesystem.DDL_PATH / "ch" / "table" / "schema" / "table_name", "table", id="table"),
        pytest.param("sensor_name", None, None, id="sensor"),
        pytest.param(
            "foreign_table_name",
            filesystem.DDL_PATH / "ch" / "foreign_table" / "schema" / "foreign_table_name",
            "foreign_table",
            id="foreign_table",
        ),
        pytest.param(
            "dictionary_name",
            filesystem.DDL_PATH / "ch" / "dictionary" / "dictionary_name.xml",
            "dictionary",
            id="dictionary",
        ),
        pytest.param(
            "function_name", filesystem.DDL_PATH / "pg" / "function" / "function_name.sql", "function", id="function"
        ),
        pytest.param("index_name", filesystem.DDL_PATH / "pg" / "index" / "index_name.sql", "index", id="index"),
        pytest.param("schema_name", filesystem.DDL_PATH / "pg" / "schema" / "index_name.sql", "schema", id="schema"),
        pytest.param("hdfs_table", filesystem.DDL_PATH / "hdfs" / "file" / "hdfs.json", "file", id="hdfs_file"),
    ],
)
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.ddl_path")
def test_get_ddl_type(mock_ddl_path: MagicMock, name: str, ddl_path: Path, etalon_answer: str) -> None:
    if ddl_path is not None:
        mock_ddl_path.return_value = ddl_path
    else:
        mock_ddl_path.side_effect = filesystem.FilesystemNotFoundError()
    filesystem_obj = filesystem.TableInFilesystem(name)
    if etalon_answer is None:
        with pytest.raises(filesystem.FilesystemNotFoundError):
            filesystem_obj.get_ddl_type()
    else:
        test_result = filesystem_obj.get_ddl_type()
        mock_ddl_path.assert_called()
        assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


@pytest.mark.parametrize(
    "name, ddl_type, etalon_answer",
    [
        pytest.param("view_name", "view", True, id="view"),
        pytest.param("table_name", "table", False, id="table"),
        pytest.param("sensor_name", "sensor", False, id="sensor"),
        pytest.param("foreign_table_name", "foreign_table", False, id="foreign_table"),
        pytest.param("dictionary_name", "dictionary", False, id="dictionary"),
        pytest.param("function_name", "function", False, id="function"),
        pytest.param("index_name", "index", False, id="index"),
        pytest.param("schema_name", "schema", False, id="schema"),
    ],
)
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.get_ddl_type")
def test_is_view(mock_ddl_type: MagicMock, name: str, ddl_type: str, etalon_answer: bool) -> None:
    mock_ddl_type.return_value = ddl_type
    filesystem_obj = filesystem.TableInFilesystem(name)
    test_result = filesystem_obj.is_view()
    mock_ddl_type.assert_called()
    assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


@pytest.mark.parametrize(
    "name, dataflow_path_exists, sensor_path, etalon_answer",
    [
        pytest.param("view_name", True, None, False, id="view"),
        pytest.param("broken_view_name", False, None, False, id="broken_view"),
        pytest.param("table_name", True, None, False, id="table"),
        pytest.param("broken_table_name", False, None, False, id="broken_table"),
        pytest.param("sensor_name", False, filesystem.SENSOR_PATH / "ch" / "schema" / "sensor_name", True, id="sensor"),
        pytest.param(
            "sensor_with_dataflow_name",
            True,
            filesystem.SENSOR_PATH / "ch" / "schema" / "sensor_name",
            False,
            id="sensor_with_dataflow",
        ),
        pytest.param("broken_sensor_name", False, None, False, id="broken_sensor"),
        pytest.param("foreign_table_name", True, None, False, id="foreign_table"),
        pytest.param("broken_foreign_table_name", False, None, False, id="broken_foreign_table"),
        pytest.param("dictionary_name", True, None, False, id="dictionary"),
        pytest.param("broken_dictionary_name", False, None, False, id="broken_dictionary"),
        pytest.param("function_name", True, None, False, id="function"),
        pytest.param("broken_function_name", False, None, False, id="broken_function"),
        pytest.param("index_name", True, None, False, id="index"),
        pytest.param("broken_index_name", False, None, False, id="broken_index"),
        pytest.param("schema_name", True, None, False, id="schema"),
        pytest.param("broken_schema_name", False, None, False, id="broken_schema"),
    ],
)
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.sensor_path")
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.dataflow_path_exists")
def test_is_sensor(
    mock_dataflow_path_exists: MagicMock,
    mock_sensor_path: MagicMock,
    name: str,
    dataflow_path_exists: bool,
    sensor_path: bool,
    etalon_answer: bool,
) -> None:
    mock_dataflow_path_exists.return_value = dataflow_path_exists
    mock_sensor_path.return_value = sensor_path
    filesystem_obj = filesystem.TableInFilesystem(name)
    test_result = filesystem_obj.is_sensor()
    mock_dataflow_path_exists.assert_called()
    if not dataflow_path_exists:
        mock_sensor_path.assert_called()
    assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


@pytest.mark.parametrize(
    "name, is_view, is_sensor, metadata_path, etalon_answer",
    [
        pytest.param("view_name", True, False, filesystem.DDL_PATH / "ch" / "view" / "view_name", "view", id="view"),
        pytest.param(
            "transform_table_name",
            False,
            False,
            filesystem.DATAFLOW_PATH / "transform" / "ch" / "olap" / "schema.table_name.sql",
            "transform",
            id="transform_table",
        ),
        pytest.param(
            "transform_full_table_name",
            False,
            False,
            filesystem.DATAFLOW_PATH / "transform_full" / "ch" / "olap" / "schema.table_name.sql",
            "transform_full",
            id="transform_full_table_name",
        ),
        pytest.param(
            "dump_table_name",
            False,
            False,
            filesystem.DATAFLOW_PATH / "dump" / "ch" / "olap.json",
            "dump",
            id="dump_table",
        ),
        pytest.param(
            "dump_full_table_name",
            False,
            False,
            filesystem.DATAFLOW_PATH / "dump_full" / "ch" / "olap.json",
            "dump_full",
            id="dump_full_table",
        ),
        pytest.param(
            "sensor_name", False, True, filesystem.SENSOR_PATH / "ch" / "schema" / "sensor_name", "sensor", id="sensor"
        ),
        pytest.param(
            "foreign_table_name",
            False,
            False,
            filesystem.DDL_PATH / "ch" / "foreign_table" / "schema" / "foreign_table_name",
            None,
            id="foreign_table",
        ),
        pytest.param(
            "dictionary_name",
            False,
            False,
            filesystem.DDL_PATH / "ch" / "dictionary" / "dictionary_name",
            None,
            id="dictionary",
        ),
        pytest.param(
            "function_name", False, False, filesystem.DDL_PATH / "pg" / "function" / "function_name", None, id="function"
        ),
        pytest.param("index_name", False, False, filesystem.DDL_PATH / "pg" / "index" / "index_name", None, id="index"),
        pytest.param(
            "schema_name", False, False, filesystem.DDL_PATH / "pg" / "schema" / "schema_name", None, id="schema"
        ),
    ],
)
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.metadata_path")
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.is_sensor")
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.is_view")
def test_get_type(
    mock_is_view: MagicMock,
    mock_is_sensor: MagicMock,
    mock_metadata_path: MagicMock,
    name: str,
    is_view: bool,
    is_sensor: bool,
    metadata_path: Path,
    etalon_answer: str,
) -> None:
    mock_is_view.return_value = is_view
    mock_is_sensor.return_value = is_sensor
    if etalon_answer is not None:
        mock_metadata_path.return_value = metadata_path
    else:
        mock_metadata_path.side_effect = filesystem.FilesystemNotFoundError()
    filesystem_obj = filesystem.TableInFilesystem(name)
    test_result = filesystem_obj.get_type()
    mock_is_view.assert_called()
    if not is_view:
        mock_is_sensor.assert_called()
    if not is_view and not is_sensor:
        mock_metadata_path.assert_called()
    assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


@pytest.mark.parametrize(
    "name, query_paths, query_separate_schema_paths, etalon_answer",
    [
        pytest.param(
            "view_name",
            [filesystem.DDL_PATH / "ch" / "view" / "view_name"],
            [],
            filesystem.DDL_PATH / "ch" / "view" / "view_name",
            id="view",
        ),
        pytest.param(
            "table_name",
            [filesystem.DDL_PATH / "ch" / "olap" / "table_name"],
            [],
            filesystem.DDL_PATH / "ch" / "olap" / "table_name",
            id="table",
        ),
        pytest.param(
            "schema.table_name",
            [],
            [filesystem.DDL_PATH / "ch" / "olap" / "schema" / "table_name"],
            filesystem.DDL_PATH / "ch" / "olap" / "schema" / "table_name",
            id="schema.table",
        ),
        pytest.param(
            "table_multiple_query_paths_name",
            [
                filesystem.DDL_PATH / "ch" / "olap" / "table_name",
                filesystem.DDL_PATH / "ch" / "events" / "table_name",
                filesystem.DDL_PATH / "pg" / "table" / "table_name",
            ],
            [],
            None,
            id="table_multiple_query_paths",
        ),
        pytest.param(
            "schema.table_multiple_query_paths_name",
            [],
            [
                filesystem.DDL_PATH / "ch" / "olap" / "schema" / "table_name",
                filesystem.DDL_PATH / "ch" / "events" / "schema" / "table_name",
                filesystem.DDL_PATH / "pg" / "table" / "schema" / "table_name",
            ],
            None,
            id="schema.table_multiple_query_paths",
        ),
        pytest.param(
            "table_multiple_query_paths_mixture_name",
            [
                filesystem.DDL_PATH / "ch" / "olap" / "table_name",
                filesystem.DDL_PATH / "ch" / "events" / "table_name",
                filesystem.DDL_PATH / "pg" / "table" / "table_name",
            ],
            [
                filesystem.DDL_PATH / "ch" / "olap" / "schema" / "table_name",
                filesystem.DDL_PATH / "ch" / "events" / "schema" / "table_name",
                filesystem.DDL_PATH / "pg" / "table" / "schema" / "table_name",
            ],
            None,
            id="table_multiple_query_paths_mixture",
        ),
        pytest.param("table_no_query_paths_name", [], [], None, id="table_no_query_paths"),
    ],
)
def test_ddl_path(
    name: str, query_paths: List[Path], query_separate_schema_paths: List[Path], etalon_answer: Path
) -> None:
    def _find_query_path_side_effect(path_template: str) -> List[Path]:
        if path_template == f"**/{name}*":
            return query_paths
        elif path_template == "**/{}".format(name.replace(".", "/")):
            return query_separate_schema_paths
        else:
            return []

    def ddl_path_call() -> Path:
        mock_glob = MagicMock(side_effect=_find_query_path_side_effect)
        with patch("pathlib.Path.glob", new=mock_glob):
            test_result = filesystem.TableInFilesystem(name).ddl_path()
        mock_glob.assert_called()
        assert (
            mock_glob.call_count == 3
        ), f"should be exectly 3 calls for `pathlib.Path.glob`, but got {mock_glob.call_count} calls"
        return test_result

    if etalon_answer is None:
        with pytest.raises((filesystem.FilesystemNotFoundError, filesystem.FilesystemDuplicatedError)):
            ddl_path_call()
    else:
        test_result = ddl_path_call()
        assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


@pytest.mark.parametrize(
    "name, is_view, is_dir, query_paths, query_separate_schema_paths, aggregator_paths, sql_paths, etalon_answer",
    [
        pytest.param(
            "schema.view_name",
            True,
            True,
            [],
            [filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name"],
            [],
            [
                filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0001.sql",
                filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0002.sql",
            ],
            filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0002.sql",
            id="view",
        ),
        pytest.param(
            "view_name",
            True,
            True,
            [filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name"],
            [filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name"],
            [],
            [
                filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0001.sql",
                filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0002.sql",
            ],
            filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0002.sql",
            id="view_no_schema",
        ),
        pytest.param(
            "view_name",
            True,
            True,
            [filesystem.DDL_PATH / "ch" / "view" / "schema_one" / "view_name"],
            [filesystem.DDL_PATH / "ch" / "view" / "schema_two" / "view_name"],
            [],
            [],
            None,
            id="view_duplicated",
        ),
        pytest.param("view_name", True, False, [], [], [], [], None, id="view_not_found"),
        pytest.param(
            "schema.table_name",
            False,
            False,
            [filesystem.DATAFLOW_PATH / "ch" / "olap" / "schema.table_name.sql"],
            [],
            [],
            [],
            filesystem.DATAFLOW_PATH / "ch" / "olap" / "schema.table_name.sql",
            id="table",
        ),
        pytest.param(
            "schema.table_name",
            False,
            False,
            [
                filesystem.DATAFLOW_PATH / "ch" / "olap" / "schema.table_name.sql",
                filesystem.DATAFLOW_PATH / "ch" / "tableau" / "schema.table_name.sql",
            ],
            [],
            [],
            [],
            None,
            id="table_duplicated",
        ),
        pytest.param("schema.table_name", False, False, [], [], [], [], None, id="table_not_found"),
        pytest.param(
            "dump_table_name",
            False,
            False,
            [],
            [],
            [filesystem.DATAFLOW_PATH / "dump" / "ch" / "olap.json"],
            [],
            filesystem.DATAFLOW_PATH / "dump" / "ch" / "olap.json",
            id="dump_table",
        ),
        pytest.param(
            "dump_table_name",
            False,
            False,
            [],
            [],
            [
                filesystem.DATAFLOW_PATH / "dump" / "ch" / "olap.json",
                filesystem.DATAFLOW_PATH / "dump" / "ch" / "dump.json",
            ],
            [],
            None,
            id="dump_table_duplicated",
        ),
        pytest.param("dump_table_name", False, False, [], [], [], [], None, id="dump_table_not_found"),
        pytest.param(
            "table_multiple_paths_name",
            False,
            False,
            [filesystem.DATAFLOW_PATH / "ch" / "olap" / "table_multiple_paths_name.sql"],
            [],
            [filesystem.DATAFLOW_PATH / "dump" / "ch" / "olap.json"],
            [],
            None,
            id="table_multiple_paths",
        ),
    ],
)
@patch("pathlib.Path.read_text")
@patch("pathlib.Path.is_dir")
@patch("json.loads")
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.is_view")
def test_metadata_path(
    mock_is_view: MagicMock,
    mock_json_loads: MagicMock,
    mock_is_dir: MagicMock,
    mock_read_text: MagicMock,
    name: str,
    is_view: bool,
    is_dir: bool,
    query_paths: List[Path],
    query_separate_schema_paths: List[Path],
    aggregator_paths: List[Path],
    sql_paths: List[Path],
    etalon_answer: Path,
) -> None:
    def _find_query_path_side_effect(path_template: str) -> List[Path]:
        if path_template == f"**/{name}*":
            return query_paths
        elif path_template == "**/{}".format(name.replace(".", "/")):
            return query_separate_schema_paths
        elif path_template == "**/*.json":
            return aggregator_paths
        elif path_template == "*.sql":
            return sql_paths
        else:
            return []

    def metadata_path_call() -> Path:
        with patch("pathlib.Path.glob", new=mock_glob):
            test_result = filesystem.TableInFilesystem(name).metadata_path()
        mock_glob.assert_called()
        assert (
            mock_glob.call_count >= 3
        ), f"should be more then 3 calls for `pathlib.Path.glob`, but got {mock_glob.call_count} calls"
        if aggregator_paths:
            mock_json_loads.assert_called()
            assert mock_json_loads.call_count == len(aggregator_paths), (
                f"should be exectly {len(aggregator_paths)} calls for `json.loads`, "
                f"but got {mock_json_loads.call_count} calls"
            )
        return test_result

    mock_is_view.return_value = is_view
    mock_json_loads.return_value = name
    mock_is_dir.return_value = is_dir
    mock_read_text.return_value = "{}"
    mock_glob = MagicMock(side_effect=_find_query_path_side_effect)
    if etalon_answer is None:
        with pytest.raises((filesystem.FilesystemNotFoundError, filesystem.FilesystemDuplicatedError)):
            metadata_path_call()
    else:
        test_result = metadata_path_call()
        assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


@pytest.mark.parametrize(
    "name, metadata_path, parameters, etalon_answer",
    [
        pytest.param(
            "schema.view_name",
            filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0002.sql",
            dict(),
            filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0002.sql",
            id="view",
        ),
        pytest.param(
            "schema.view_name",
            filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0002.sql",
            dict(query=filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0001.sql"),
            filesystem.DDL_PATH / "ch" / "view" / "schema" / "view_name" / "0001.sql",
            id="view_with_query",
        ),
        pytest.param(
            "schema.table_name",
            filesystem.DATAFLOW_PATH / "ch" / "olap" / "schema.table_name.sql",
            dict(),
            filesystem.DATAFLOW_PATH / "ch" / "olap" / "schema.table_name.sql",
            id="table",
        ),
        pytest.param(
            "schema.table_name",
            filesystem.DATAFLOW_PATH / "ch" / "olap" / "schema.table_name.sql",
            dict(query=filesystem.DATAFLOW_PATH / "ch" / "events" / "schema.table_name.sql"),
            filesystem.DATAFLOW_PATH / "ch" / "events" / "schema.table_name.sql",
            id="table_with_query",
        ),
    ],
)
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.get_parameters")
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.metadata_path")
def test_dataflow_path(
    mock_metadata_path: MagicMock,
    mock_get_parameters: MagicMock,
    name: str,
    metadata_path: Path,
    parameters: Dict[str, Any],
    etalon_answer: Path,
) -> None:
    mock_metadata_path.return_value = metadata_path
    mock_get_parameters.return_value = parameters
    test_result = filesystem.TableInFilesystem(name).dataflow_path()
    mock_metadata_path.assert_called()
    mock_get_parameters.assert_called()
    assert etalon_answer == test_result, f"should be {etalon_answer}, but got {test_result}"


@pytest.mark.parametrize(
    "name, dataflow_path_exists, query_paths, aggregator_paths, etalon_answer",
    [
        pytest.param(
            "sensor_name",
            False,
            [filesystem.SENSOR_PATH / "ch" / "events" / "sensor_name.sql"],
            [],
            filesystem.SENSOR_PATH / "ch" / "events" / "sensor_name.sql",
            id="sensor",
        ),
    ],
)
@patch("pathlib.Path.read_text")
@patch("json.loads")
@patch("lib.trgetl.filesystem.table_in_fs.TableInFilesystem.dataflow_path_exists")
def test_sensor_path(
    mock_dataflow_path_exists: MagicMock,
    mock_json_loads: MagicMock,
    mock_read_text: MagicMock,
    name: str,
    dataflow_path_exists: bool,
    query_paths: List[Path],
    aggregator_paths: List[Path],
    etalon_answer: Path,
) -> None:
    def _find_query_path_side_effect(path_template: str) -> List[Path]:
        if path_template == f"**/{name}*":
            return query_paths
        elif path_template == "**/*.json":
            return aggregator_paths
        else:
            return []

    def sensor_path_call() -> Path:
        mock_glob = MagicMock(side_effect=_find_query_path_side_effect)
        with patch("pathlib.Path.glob", new=mock_glob):
            test_result = filesystem.TableInFilesystem(name).sensor_path()
        mock_glob.assert_called()
        assert (
            mock_glob.call_count == 3
        ), f"should be exectly 3 calls for `pathlib.Path.glob`, but got {mock_glob.call_count} calls"
        if aggregator_paths:
            mock_json_loads.assert_called()
            assert mock_json_loads.call_count == len(aggregator_paths), (
                f"should be exectly {len(aggregator_paths)} calls for `json.loads`, "
                f"but got {mock_json_loads.call_count} calls"
            )
        return test_result

    mock_dataflow_path_exists.return_value = dataflow_path_exists
    mock_json_loads.return_value = name
    mock_read_text.return_value = "{}"
    if dataflow_path_exists:
        with pytest.raises((filesystem.FilesystemDataflowPathFound,)):
            sensor_path_call()
    else:
        test_result = sensor_path_call()
        assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


# def test_sensor_path(create_samples, filesystem_object):
#     if filesystem_object.obj.dataflow_path_exists():
#         with pytest.raises(filesystem.FilesystemDataflowPathFound):
#             filesystem_object.obj.sensor_path()
#     elif not filesystem_object.obj.sensor_path_exists():
#         with pytest.raises(filesystem.FilesystemNotFoundError):
#             filesystem_object.obj.sensor_path()
#     else:
#         test_result = filesystem_object.obj.sensor_path()
#         paths_list = filesystem_object.paths
#         assert test_result in paths_list, (
#             f"""should be one of {
#                 list(map(str, paths_list))
#             }, but got {test_result}"""
#         )


def test_get_db(create_samples, filesystem_object):
    if not filesystem_object.obj.ddl_path_exists():
        with pytest.raises(filesystem.FilesystemNotFoundError):
            filesystem_object.obj.get_db()
    else:
        test_result = filesystem_object.obj.get_db()
        etalon_answer = filesystem_object.schema
        assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


def test_get_source_db(create_samples, filesystem_object):
    test_result = filesystem_object.obj.get_source_db()
    etalon_answer = filesystem_object.source_schema
    assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


def test_get_sensor_db(create_samples, filesystem_object):
    test_result = filesystem_object.obj.get_sensor_db()
    etalon_answer = filesystem_object.sensor_schema
    assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


def test_all_sensors(create_samples):
    etalon_sensor_set = {
        sensor_obj.name
        for sensor_obj in FILESYSTEM_OBJECTS
        if filesystem.TableInFilesystem(sensor_obj.name).sensor_path_exists()
    }
    all_sensor_result = filesystem.Filesystem().all_sensors()
    assert all(
        sensor_name in all_sensor_result for sensor_name in etalon_sensor_set
    ), f"""should be all sensor names in {etalon_sensor_set}, but got {all_sensor_result}"""


@pytest.mark.parametrize("db_type", DB_TYPE_LIST)
def test_all_sensors_with_type(create_samples, db_type):
    etalon_sensor_type_set = {
        sensor_obj.name
        for sensor_obj in FILESYSTEM_OBJECTS
        if filesystem.TableInFilesystem(sensor_obj.name).sensor_path_exists()
        and (sensor_obj.schema[0] == db_type or sensor_obj.source_schema[0] == db_type)
    }
    sensor_type_result = filesystem.Filesystem().all_sensors(table_type=db_type)
    assert all(
        sensor_name in sensor_type_result for sensor_name in etalon_sensor_type_set
    ), f"""should be all sensor names in {
            etalon_sensor_type_set
        }, but got {sensor_type_result}"""


@pytest.mark.parametrize("schema", SCHEMA_LIST)
def test_all_sensors_with_schema(create_samples, schema):
    etalon_sensor_schema_set = {
        sensor_obj.name
        for sensor_obj in FILESYSTEM_OBJECTS
        if filesystem.TableInFilesystem(sensor_obj.name).sensor_path_exists()
        and (
            sensor_obj.schema == schema
            or sensor_obj.source_schema == schema
            or (schema == ("ch", "olap") and sensor_obj.schema == ("ch", None))
        )
    }
    sensor_schema_result = filesystem.Filesystem().all_sensors(table_type=schema[0], table_db=schema[1])
    assert all(
        sensor_name in sensor_schema_result for sensor_name in etalon_sensor_schema_set
    ), f"""should be all sensor names in {
            etalon_sensor_schema_set
        }, but got {sensor_schema_result}"""


@pytest.mark.parametrize("db", DB_LIST)
def test_all_sensors_ignore_db_only(create_samples, db):
    etalon_sensor_set = {
        sensor_obj.name
        for sensor_obj in FILESYSTEM_OBJECTS
        if filesystem.TableInFilesystem(sensor_obj.name).sensor_path_exists()
    }
    sensor_db_only_result = filesystem.Filesystem().all_sensors(table_db=db)
    assert all(
        sensor_name in sensor_db_only_result for sensor_name in etalon_sensor_set
    ), f"""should be all sensor names in {
            etalon_sensor_set
        }, but got {sensor_db_only_result}"""


def test_dependencies(create_samples, filesystem_object):
    test_result = filesystem_object.obj.extract_live_dependencies()
    etalon_answer = filesystem_object.dependencies
    assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


def test_get_parameters(create_samples, filesystem_object):
    test_result = filesystem_object.obj.get_parameters()
    etalon_answer = filesystem_object.raw_parameters
    assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"


def test_get_standard_parameters(create_samples, filesystem_object):
    test_result = filesystem_object.obj.get_standart_parameters()
    etalon_answer = filesystem_object.parameters
    assert test_result == etalon_answer, f"should be {etalon_answer}, but got {test_result}"
