import datetime as dt
import getpass
import io
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pytest
from mock import call, patch  # type: ignore
from mock.mock import MagicMock  # type: ignore

from ... import filesystem
from ...report import Report, ReportError
from ...sender import Email, MyteamBot

CURRENT_DATETIME = dt.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
YESTERDAY_DATETIME = CURRENT_DATETIME - dt.timedelta(days=1)
TEST_RECEIVER = getpass.getuser() + "@corp.mail.ru"
TEST_ERROR_RECEIVERS = [getpass.getuser() + "@corp.mail.ru", getpass.getuser() + "@corp.mail.ru"]
DEFAULT_ERROR_RECEIVER = Report.DEFAULT_ERROR_RECEIVER
DEFAULT_PRIORITY = 100
REPORT_SAMPLES_PATH = Path(__file__).parent / "report_samples"
REPORT_PATH = filesystem.REPORT_PATH
MSG_FAILED_START = "Report send failed with message:\n"

RAW_MSG_ALL_IS_OK = ""
SUBJECT_NOT_SET = ""

RAW_MSG_ANOTHER_REPORT_FAILED = (
    'ERROR: report "{name}" callable "{report_name}" is OK, but another report callables failed'
)
MSG_ANOTHER_REPORT_FAILED = MSG_FAILED_START + RAW_MSG_ANOTHER_REPORT_FAILED

RAW_MSG_NOT_A_DICT = 'ERROR: report "{name}" callable "{report_name}" does not return dictionary'
MSG_NOT_A_DICT = MSG_FAILED_START + RAW_MSG_NOT_A_DICT

RAW_MSG_NOT_ALL_FIELDS = 'ERROR: report "{name}" callable "{report_name}" output does not contain all required fields'
MSG_NOT_ALL_FIELDS = MSG_FAILED_START + RAW_MSG_NOT_ALL_FIELDS

RAW_MSG_TABLE_CHECK_FAILED = 'ERROR: report "{name}" callable "{report_name}" table checker failed'
MSG_TABLE_CHECK_FAILED = MSG_FAILED_START + RAW_MSG_TABLE_CHECK_FAILED

RAW_MSG_CUSTOM_CHECK_FAILED = 'ERROR: report "{name}" callable "{report_name}" custom checker failed'  # noqa: E501
MSG_CUSTOM_CHECK_FAILED = MSG_FAILED_START + RAW_MSG_CUSTOM_CHECK_FAILED


def test_import_class_report() -> None:
    from lib.trgetl.report import Report  # noqa: F401
    from lib.trgetl.report import __all__  # noqa: F401


@pytest.mark.parametrize(
    "name, modules, module_callables, etalon_callables",
    [
        pytest.param(
            "report_1",
            {"module": REPORT_PATH / "report_1.py"},
            {"module": ["report_callable_1", "report_callable_2"]},
            {"module": {"report_callable_1": lambda: None, "report_callable_2": lambda: None}},
            id="simple",
        ),
        pytest.param(
            "report_package",
            {
                "report_1": REPORT_PATH / "report_package/report_1.py",
                "report_2": REPORT_PATH / "report_package/report_2.py",
            },
            {
                "report_1": ["report_callable_1", "report_callable_2"],
                "report_2": ["report_callable_3", "report_callable_4"],
            },
            {
                "report_1": {"report_callable_1": lambda: None, "report_callable_2": lambda: None},
                "report_2": {"report_callable_3": lambda: None, "report_callable_4": lambda: None},
            },
            id="package",
        ),
    ],
)
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
def test_get_callables(
    mock_modules: MagicMock,
    mock_std_params: MagicMock,
    mock_dag_constants: MagicMock,
    name: str,
    modules: Dict[str, Path],
    module_callables: Dict[str, List[str]],
    etalon_callables: Dict[str, Dict[str, Callable]],
) -> None:
    def _import_module_side_effect(name: str, path: Path) -> ModuleType:
        module = ModuleType(name)
        for clbl in module_callables[name]:
            setattr(module, clbl, lambda: None)
        return module

    mock_modules.return_value = modules
    mock_import_module = MagicMock(side_effect=_import_module_side_effect)
    with patch("lib.trgetl.report.Report._import_report_module", new=mock_import_module):
        report = Report(name)
    mock_import_module.assert_called()
    assert mock_import_module.call_count == len(module_callables.keys()), (
        f"should be {len(module_callables.keys())} calls for `Report._import_report_module`, "
        f"but got {mock_import_module.call_count} calls"
    )
    mock_modules.assert_called()
    assert (
        mock_modules.call_count == 1
    ), f"should be exectly one call for `ReportInFilesystem.get_modules`, but got {mock_modules.call_count} calls"
    mock_std_params.assert_called()
    assert mock_std_params.call_count == 1, (
        "should be exectly one call for `ReportInFilesystem.get_standard_parameters`, "
        f"but got {mock_std_params.call_count} calls"
    )
    mock_dag_constants.assert_called()
    assert (
        mock_dag_constants.call_count == 1
    ), f"should be exectly one call for `Report._get_dag_constants`, but got {mock_dag_constants.call_count} calls"
    for module in report.callables.keys():
        etalon_module_callables = etalon_callables.get(module)
        assert etalon_module_callables, f"got unexpected module: {module}"
        report_module_callables = report.callables[module]
        assert (
            etalon_module_callables.keys() == report_module_callables.keys()
        ), f"should be {etalon_module_callables.keys()}, but got {report_module_callables.keys()}"


@pytest.mark.parametrize(
    "name, global_sender, callables, parameters, representation",
    [
        pytest.param(
            "multiattach",
            None,
            {"multiattach": {"report_multiattach": lambda: None}},
            {
                "schedule": "*/5 * * * *",
                "sender": "myteambot",
            },
            [
                (
                    "Report('multiattach'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '*/5 * * * *', 'sender': 'myteambot'},\n"
                    "module name: multiattach, callables:"
                ),
                "    report_multiattach",
            ],
            id="multiattach_None",
        ),
        pytest.param(
            "multiattach",
            MyteamBot(),
            {"multiattach": {"report_multiattach": lambda: None}},
            {
                "schedule": "*/5 * * * *",
                "sender": "myteambot",
            },
            [
                (
                    "Report('multiattach'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '*/5 * * * *', 'sender': 'myteambot'},\n"
                    "module name: multiattach, callables:"
                ),
                "    report_multiattach",
            ],
            id="multiattach_myteambot",
        ),
        pytest.param(
            "multiattach",
            Email(),
            {"multiattach": {"report_multiattach": lambda: None}},
            {
                "schedule": "*/5 * * * *",
                "sender": "myteambot",
            },
            [
                (
                    "Report('multiattach'), global sender: Email,\n"
                    "parameters: {'schedule': '*/5 * * * *', 'sender': 'myteambot'},\n"
                    "module name: multiattach, callables:"
                ),
                "    report_multiattach",
            ],
            id="multiattach_email",
        ),
        pytest.param(
            "dummy",
            None,
            {
                "dummy": {
                    "report_1": lambda: None,
                    "report_2": lambda: None,
                    "report_3": lambda: None,
                    "report_test": lambda: None,
                }
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('dummy'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: dummy, callables:"
                ),
                "    report_1",
                "    report_2",
                "    report_3",
                "    report_test",
            ],
            id="dummy_None",
        ),
        pytest.param(
            "dummy",
            MyteamBot(),
            {
                "dummy": {
                    "report_1": lambda: None,
                    "report_2": lambda: None,
                    "report_3": lambda: None,
                    "report_test": lambda: None,
                }
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('dummy'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: dummy, callables:"
                ),
                "    report_1",
                "    report_2",
                "    report_3",
                "    report_test",
            ],
            id="dummy_myteambot",
        ),
        pytest.param(
            "dummy",
            Email(),
            {
                "dummy": {
                    "report_1": lambda: None,
                    "report_2": lambda: None,
                    "report_3": lambda: None,
                    "report_test": lambda: None,
                }
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('dummy'), global sender: Email,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: dummy, callables:"
                ),
                "    report_1",
                "    report_2",
                "    report_3",
                "    report_test",
            ],
            id="dummy_email",
        ),
        pytest.param(
            "test_iOS_14_5_plus",
            None,
            {"test_iOS_14_5_plus": {"report_without_attachment": lambda: None, "report_with_attachment": lambda: None}},
            {
                "chat": "d.kulemin@corp.mail.ru",
                "schedule": "*/5 * * * *",
                "sender": "myteambot",
            },
            [
                (
                    "Report('test_iOS_14_5_plus'), global sender: MyteamBot,\n"
                    "parameters: {'chat': 'd.kulemin@corp.mail.ru', 'schedule': '*/5 * * * *', "
                    "'sender': 'myteambot'},\n"
                    "module name: test_iOS_14_5_plus, callables:"
                ),
                "    report_without_attachment",
                "    report_with_attachment",
            ],
            id="test_iOS_14_5_plus_None",
        ),
        pytest.param(
            "test_iOS_14_5_plus",
            MyteamBot(),
            {"test_iOS_14_5_plus": {"report_without_attachment": lambda: None, "report_with_attachment": lambda: None}},
            {
                "chat": "d.kulemin@corp.mail.ru",
                "schedule": "*/5 * * * *",
                "sender": "myteambot",
            },
            [
                (
                    "Report('test_iOS_14_5_plus'), global sender: MyteamBot,\n"
                    "parameters: {'chat': 'd.kulemin@corp.mail.ru', 'schedule': '*/5 * * * *', "
                    "'sender': 'myteambot'},\n"
                    "module name: test_iOS_14_5_plus, callables:"
                ),
                "    report_without_attachment",
                "    report_with_attachment",
            ],
            id="test_iOS_14_5_plus_myteambot",
        ),
        pytest.param(
            "test_iOS_14_5_plus",
            Email(),
            {"test_iOS_14_5_plus": {"report_without_attachment": lambda: None, "report_with_attachment": lambda: None}},
            {
                "chat": "d.kulemin@corp.mail.ru",
                "schedule": "*/5 * * * *",
                "sender": "myteambot",
            },
            [
                (
                    "Report('test_iOS_14_5_plus'), global sender: Email,\n"
                    "parameters: {'chat': 'd.kulemin@corp.mail.ru', 'schedule': '*/5 * * * *', "
                    "'sender': 'myteambot'},\n"
                    "module name: test_iOS_14_5_plus, callables:"
                ),
                "    report_without_attachment",
                "    report_with_attachment",
            ],
            id="test_iOS_14_5_plus_email",
        ),
        pytest.param(
            "report_package",
            None,
            {
                "report_1": {"report_content_1": lambda: None, "report_attachment": lambda: None},
                "report_2": {"report_content_2": lambda: None},
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('report_package'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: report_1, callables:"
                ),
                "    report_content_1",
                "    report_attachment",
                "module name: report_2, callables:",
                "    report_content_2",
            ],
            id="report_package_None",
        ),
        pytest.param(
            "report_package",
            MyteamBot(),
            {
                "report_1": {"report_content_1": lambda: None, "report_attachment": lambda: None},
                "report_2": {"report_content_2": lambda: None},
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('report_package'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: report_1, callables:"
                ),
                "    report_content_1",
                "    report_attachment",
                "module name: report_2, callables:",
                "    report_content_2",
            ],
            id="report_package_myteambot",
        ),
        pytest.param(
            "report_package",
            Email(),
            {
                "report_1": {"report_content_1": lambda: None, "report_attachment": lambda: None},
                "report_2": {"report_content_2": lambda: None},
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('report_package'), global sender: Email,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: report_1, callables:"
                ),
                "    report_content_1",
                "    report_attachment",
                "module name: report_2, callables:",
                "    report_content_2",
            ],
            id="report_package_email",
        ),
        pytest.param(
            "failed_checkers",
            None,
            {
                "failed_checkers": {
                    "report_nothing_failed": lambda: None,
                    "report_check_failed": lambda: None,
                    "report_custom_checker_failed": lambda: None,
                }
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('failed_checkers'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: failed_checkers, callables:"
                ),
                "    report_nothing_failed",
                "    report_check_failed",
                "    report_custom_checker_failed",
            ],
            id="failed_checkers_None",
        ),
        pytest.param(
            "failed_checkers",
            MyteamBot(),
            {
                "failed_checkers": {
                    "report_nothing_failed": lambda: None,
                    "report_check_failed": lambda: None,
                    "report_custom_checker_failed": lambda: None,
                }
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('failed_checkers'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: failed_checkers, callables:"
                ),
                "    report_nothing_failed",
                "    report_check_failed",
                "    report_custom_checker_failed",
            ],
            id="failed_checkers_myteambot",
        ),
        pytest.param(
            "failed_checkers",
            Email(),
            {
                "failed_checkers": {
                    "report_nothing_failed": lambda: None,
                    "report_check_failed": lambda: None,
                    "report_custom_checker_failed": lambda: None,
                }
            },
            {
                "schedule": "0 12 * * *",
            },
            [
                (
                    "Report('failed_checkers'), global sender: Email,\n"
                    "parameters: {'schedule': '0 12 * * *'},\n"
                    "module name: failed_checkers, callables:"
                ),
                "    report_nothing_failed",
                "    report_check_failed",
                "    report_custom_checker_failed",
            ],
            id="failed_checkers_email",
        ),
        pytest.param(
            "skipped_true",
            None,
            {"skipped_true": {"report_func": lambda: None}},
            {"schedule": "0 12 * * *", "skip": "true"},
            [
                (
                    "Report('skipped_true'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '0 12 * * *', 'skip': 'true'},\n"
                    "module name: skipped_true, callables:"
                ),
                "    report_func",
            ],
            id="skipped_true_None",
        ),
        pytest.param(
            "skipped_True",
            None,
            {"skipped_True": {"report_func": lambda: None}},
            {"schedule": "0 12 * * *", "skip": True},
            [
                (
                    "Report('skipped_True'), global sender: MyteamBot,\n"
                    "parameters: {'schedule': '0 12 * * *', 'skip': True},\n"
                    "module name: skipped_True, callables:"
                ),
                "    report_func",
            ],
            id="skipped_True_None",
        ),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
def test_report_initialization(
    mock_modules: MagicMock,
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    name: str,
    global_sender: Union[MyteamBot, Email],
    callables: dict,
    parameters: dict,
    representation: str,
) -> None:
    mock_callables.return_value = callables
    mock_std_params.return_value = parameters
    report = Report(name, global_sender)
    mock_modules.assert_called()
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    assert report.name == name, f"name should be {name}, but got {report.name}"
    etalon_sender_type = type(global_sender or MyteamBot())
    assert isinstance(
        report.sender, etalon_sender_type
    ), f"sender type should be {etalon_sender_type}, but got {type(report.sender)}"
    result_repr_list = [
        line.split(":")[0] if line.startswith("    ") else line for line in report.__repr__().split("\n\n")
    ]
    assert all(
        line in representation for line in result_repr_list
    ), f"representation should be {representation}, but got {result_repr_list}"
    assert report.execution_time is None, f"initial date should be None, but got {report.execution_time}"


@pytest.mark.parametrize(
    "time, schedule, execution_time",
    [
        pytest.param(None, "0 12 * * *", YESTERDAY_DATETIME.replace(hour=12), id="empty_time_daily"),
        pytest.param(None, "*/5 * * * *", CURRENT_DATETIME - dt.timedelta(minutes=5), id="empty_time_5min"),
        pytest.param(
            None,
            "45 17 */2 * *",
            (
                CURRENT_DATETIME.replace(hour=17, minute=45) - dt.timedelta(days=2)
                if CURRENT_DATETIME.day % 2 == 1
                else CURRENT_DATETIME.replace(hour=17, minute=45) - dt.timedelta(days=1)
            ),
            id="empty_time_odd_day_at_17_45",
        ),
        pytest.param(dt.datetime(2022, 11, 11, 0, 0), "0 12 * * *", dt.datetime(2022, 11, 11, 0, 0), id="datetime_time"),
        pytest.param("2022-11-11", "0 12 * * *", dt.datetime(2022, 11, 11, 0, 0), id="string_date"),
        pytest.param("2022-11-11 17:45", "0 12 * * *", dt.datetime(2022, 11, 11, 17, 45), id="string_time"),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_set_execution_time(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    time: Optional[Union[str, dt.datetime]],
    schedule: str,
    execution_time: dt.datetime,
) -> None:
    mock_std_params.return_value = dict(airflow=dict(schedule=schedule))
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    report._set_execution_time(time)
    assert (
        execution_time == report.execution_time
    ), f"execution_time should be {execution_time}, but got {report.execution_time}"


@pytest.mark.parametrize(
    "callable_out, etalon_skip",
    [
        pytest.param(None, False, id="nondict_None"),
        pytest.param(2, False, id="nondict_int"),
        pytest.param({}, False, id="no_skip_param"),
        pytest.param({"skip": False}, False, id="skip_false"),
        pytest.param({"skip": True}, True, id="skip_true"),
        pytest.param({"skip": "true"}, False, id="ignore_skip_param"),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_is_skip(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    callable_out: dict,
    etalon_skip: bool,
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    result_skip = report._is_skip(callable_out)
    assert etalon_skip == result_skip, f"should be {etalon_skip}, but got {result_skip}"


@pytest.mark.parametrize(
    "callable_out",
    [
        pytest.param(None, id="nondict"),
        pytest.param(dict(), id="no_custom_checker"),
        pytest.param(
            {
                "name": "test",
                "content": "table checker",
                "receivers": TEST_RECEIVER,
                "check": ["default.dummy_table"],
            },
            id="check_one_table",
        ),
        pytest.param(
            {
                "name": "test",
                "content": "table checker",
                "receivers": TEST_RECEIVER,
                "check": ["default.dummy_table_1", "default.dummy_table_2"],
            },
            id="check_multiple_tables",
        ),
    ],
)
@patch("lib.trgetl.report.Report._check_tables", return_value=True)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_check_tables(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    mock_check_tables: MagicMock,
    callable_out: dict,
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    report._check_tables(callable_out)
    mock_check_tables.assert_called_once_with(callable_out)


@pytest.mark.parametrize(
    "callable_out",
    [
        pytest.param(None, id="nondict"),
        pytest.param(dict(), id="no_custom_checker"),
        pytest.param(
            {
                "name": "test",
                "content": "custom checker",
                "receivers": TEST_RECEIVER,
                "custom_check": "_custom_checker",
            },
            id="custom_checker",
        ),
    ],
)
@patch("lib.trgetl.report.Report._custom_check", return_value=True)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_custom_check(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    mock_custom_check: MagicMock,
    callable_out: dict,
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    report._custom_check(callable_out)
    mock_custom_check.assert_called_once_with(callable_out)


@pytest.mark.parametrize(
    "callable_out, etalon_check",
    [
        pytest.param(None, False, id="nondict_None"),
        pytest.param(2, False, id="nondict_int"),
        pytest.param(
            {
                3: None,
                "error_receivers": [TEST_RECEIVER, TEST_RECEIVER],
                "send_params": {
                    "dummy": "dummy",
                },
            },
            False,
            id="no_necessery_fields",
        ),
        pytest.param(
            {"name": "test", "content": "sometext", "receivers": TEST_RECEIVER}, True, id="all_necessery_fields"
        ),
        pytest.param(
            {
                "name": "test",
                "content": "table checker",
                "receivers": TEST_RECEIVER,
                "check": ["default.dummy_table"],
            },
            True,
            id="check_one_table",
        ),
        pytest.param(
            {
                "name": "test",
                "content": "table checker",
                "receivers": TEST_RECEIVER,
                "check": ["default.dummy_table_1", "default.dummy_table_2"],
            },
            True,
            id="check_multiple_tables",
        ),
        pytest.param(
            {
                "name": "test",
                "content": "custom checker",
                "receivers": TEST_RECEIVER,
                "custom_check": "_custom_checker",
            },
            True,
            id="custom_checker",
        ),
    ],
)
@patch("lib.trgetl.report.Report._custom_check", return_value=True)
@patch("lib.trgetl.report.Report._check_tables", return_value=True)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_check(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    mock_check_tables: MagicMock,
    mock_custom_check: MagicMock,
    callable_out: dict,
    etalon_check: bool,
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    try:
        report._check(callable_out)
        result_check = True
    except ReportError:
        result_check = False
    assert etalon_check == result_check, f"should be {etalon_check}, but got {result_check}"


@pytest.mark.parametrize(
    "callable_out, etalon_name",
    [
        pytest.param(None, "name", id="nondict_name"),
        pytest.param(dict(), "name", id="name"),
        pytest.param({"name": "report_name"}, "report_name", id="report_name"),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_get_name(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    callable_out: dict,
    etalon_name: str,
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    result_name = report._get_name(callable_out)
    assert etalon_name == result_name, f"should be {etalon_name}, but got {result_name}"


@pytest.mark.parametrize(
    "callable_out, etalon_sender",
    [
        pytest.param(None, type(None), id="nondict_None"),
        pytest.param(dict(), type(None), id="None"),
        pytest.param({"sender": MyteamBot()}, MyteamBot, id="myteambot"),
        pytest.param({"sender": Email()}, Email, id="email"),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_get_sender(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    callable_out: dict,
    etalon_sender: type,
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    result = report._get_sender(callable_out)
    assert isinstance(result, etalon_sender), f"should be {etalon_sender.__name__}, but got {result}"


@pytest.mark.parametrize(
    "names, callables, etalon_dictionary_count",
    [
        pytest.param(
            ["report_func"],
            [lambda: {"name": "name", "content": "this is text", "receivers": "test@mail.ru"}],
            1,
            id="simple",
        ),
        pytest.param(
            ["report_func_with_skip"],
            [lambda: {"name": "name", "content": "this is text", "receivers": "test@mail.ru", "skip": True}],
            0,
            id="simple_with_skip",
        ),
        pytest.param(
            ["report_multiattach"],
            [
                lambda: {
                    "name": "multiattach",
                    "sender": MyteamBot.__name__,
                    "content": "some text for multiattach",
                    "receivers": TEST_RECEIVER,
                    "attachment": ["_form_attach", "_form_attach"],
                    "error_receivers": TEST_RECEIVER,
                }
            ],
            1,
            id="multiattach",
        ),
        pytest.param(
            ["report_1", "report_2", "report_3", "report_test"],
            [
                lambda: None,
                2,
                lambda: {
                    3: None,
                    "error_receivers": [TEST_RECEIVER, TEST_RECEIVER],
                    "send_params": {
                        "dummy": "dummy",
                    },
                },
                lambda: {"name": "test", "content": "sometext", "receivers": TEST_RECEIVER, "send_params": "dummy"},
            ],
            4,
            id="dummy",
        ),
        pytest.param(
            ["report_1", "report_2", "report_3", "report_test"],
            [
                None,
                lambda: 2,
                lambda: {
                    3: None,
                    "error_receivers": [TEST_RECEIVER, TEST_RECEIVER],
                    "skip": True,
                    "send_params": {
                        "dummy": "dummy",
                    },
                },
                lambda: {
                    "name": "test",
                    "content": "sometext",
                    "receivers": TEST_RECEIVER,
                    "skip": True,
                    "send_params": "dummy",
                },
            ],
            2,
            id="dummy_with_skips",
        ),
        pytest.param(
            ["report_nothing_failed", "report_check_failed", "report_custom_checker_failed"],
            [
                lambda: {
                    "name": "failed_checkers",
                    "content": "all is OK",
                    "receivers": TEST_RECEIVER,
                    "error_receivers": TEST_RECEIVER,
                    "priority": 1,
                    "send_params": dict(),
                },
                lambda: {
                    "name": "failed_checkers",
                    "content": "table checker failed",
                    "receivers": TEST_RECEIVER,
                    "error_receivers": TEST_RECEIVER,
                    "priority": 2,
                    "check": ["default.dummy_table"],
                    "send_params": {
                        "markup": None,
                    },
                },
                lambda: {
                    "name": "failed_checkers",
                    "content": "custom checker failed",
                    "receivers": TEST_RECEIVER,
                    "error_receivers": TEST_RECEIVER,
                    "priority": 3,
                    "custom_check": "_custom_checker",
                    "send_params": {
                        "one": 1,
                        "two": 2,
                    },
                },
            ],
            3,
            id="failed_checkers",
        ),
        pytest.param(
            ["report_without_attachment", "report_with_attachment"],
            [
                lambda: {
                    "name": "test_iOS_14_5_plus",
                    "content": "_form_report_text",
                    "receivers": TEST_RECEIVER,
                    "error_receivers": [TEST_RECEIVER, TEST_RECEIVER],
                    "priority": 1,
                    "check": ["default.source_os_access_requests_day"],
                    "custom_check": "_custom_checker",
                    "send_params": {
                        "markup": "MarkdownV2",
                    },
                },
                lambda: {
                    "name": "test_iOS_14_5_plus",
                    "content": "",
                    "attachment": "_prepare_attachment",
                    "receivers": [TEST_RECEIVER, TEST_RECEIVER],
                    "priority": 2,
                },
            ],
            2,
            id="test_iOS_14_5_plus",
        ),
    ],
)
@patch("lib.trgetl.report.Report._check_tables", return_value=True)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_get_callable_out(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    mock_check_tables: MagicMock,
    names: list,
    callables: list,
    etalon_dictionary_count: int,
) -> None:
    module_callables = dict(zip(names, callables))
    report = Report("name")
    report.execution_time = CURRENT_DATETIME
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    result_dictionaries = report._get_callable_out(module_callables)
    assert etalon_dictionary_count == len(
        result_dictionaries
    ), f"should be exactly {etalon_dictionary_count} out dictionaries, but got {len(result_dictionaries)} dictionaries"
    necessary_fields = {
        "report_name",
        "name",
        "sender",
        "error_receivers",
        "subject",
        "raw_error_msg",
        "error_msg",
        "priority",
        "check",
    }
    for dictionary in result_dictionaries:
        assert necessary_fields.issubset(
            set(dictionary.keys())
        ), f"fields {necessary_fields}, should be in {dictionary.keys()}"


@pytest.mark.parametrize(
    "name, report_dict, etalon_text, should_raise",
    [
        pytest.param(
            "multiattach", dict(content="some text for multiattach"), "some text for multiattach", False, id="multiattach"
        ),
        pytest.param("dummy", dict(content="sometext"), "sometext", False, id="dummy_report_test"),
        pytest.param(
            "test_iOS_14_5_plus",
            dict(content="some text for test_iOS_14_5_plus"),
            "some text for test_iOS_14_5_plus",
            False,
            id="test_iOS_14_5_plus_report_without_attachment",
        ),
        pytest.param("test_iOS_14_5_plus", dict(content=""), "", False, id="test_iOS_14_5_plus_report_with_attachment"),
        pytest.param(
            "report_package",
            dict(content="some text for report_package"),
            "some text for report_package",
            False,
            id="report_package_report_content",
        ),
        pytest.param("report_package", dict(content=""), "", False, id="report_package_report_attachment"),
        pytest.param(
            "failed_checkers", dict(content="all is OK"), "all is OK", False, id="failed_checkers_report_nothing_failed"
        ),
        pytest.param(
            "failed_checkers",
            dict(content="table checker failed"),
            "table checker failed",
            False,
            id="failed_checkers_report_check_failed",
        ),
        pytest.param(
            "failed_checkers",
            dict(content="custom checker failed"),
            "custom checker failed",
            False,
            id="failed_checkers_report_custom_checker_failed",
        ),
        pytest.param(
            "content_with_date",
            dict(
                content=lambda date: f"content has date: {date} as input",
                content_date_arg_name="date",
            ),
            f"content has date: {CURRENT_DATETIME.date()} as input",
            False,
            id="content_with_date",
        ),
        pytest.param(
            "content_with_datetime",
            dict(
                content=lambda time: f"content has time: {time} as input",
                content_time_arg_name="time",
            ),
            f"content has time: {CURRENT_DATETIME} as input",
            False,
            id="content_with_datetime",
        ),
        pytest.param(
            "content_with_date_datetime_raise",
            dict(
                content=lambda date, time: (date, time),
                content_date_arg_name="date",
                content_time_arg_name="time",
            ),
            "",
            True,
            id="content_with_date_datetime_raise",
        ),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_get_content_text(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    name: str,
    report_dict: dict,
    etalon_text: str,
    should_raise: bool,
) -> None:
    report = Report("name")
    report.execution_time = CURRENT_DATETIME
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    if should_raise:
        with pytest.raises(ReportError):
            report._get_content_text(report_dict)
    else:
        result_text = report._get_content_text(report_dict)
        assert etalon_text == result_text, f"should be {etalon_text}, but got {result_text}"


@pytest.mark.parametrize(
    "name, report_dict, etalon_attachments, should_raise",
    [
        pytest.param(
            "multiattach",
            dict(attachment=[io.BytesIO(), io.BytesIO()]),
            [io.BytesIO(), io.BytesIO()],
            False,
            id="multiattach",
        ),
        pytest.param("dummy", dict(), None, False, id="dummy_report_3"),
        pytest.param("dummy", dict(), None, False, id="dummy_report_test"),
        pytest.param("test_iOS_14_5_plus", dict(), None, False, id="test_iOS_14_5_plus_report_without_attachment"),
        pytest.param(
            "test_iOS_14_5_plus",
            dict(attachment=io.BytesIO()),
            [io.BytesIO()],
            False,
            id="test_iOS_14_5_plus_report_with_attachment",
        ),
        pytest.param("report_package", dict(attachment=io.BytesIO()), [io.BytesIO()], False, id="report_package"),
        pytest.param("failed_checkers", dict(), None, False, id="failed_checkers"),
        pytest.param(
            "attachment_with_date",
            dict(
                attachment=lambda date: io.BytesIO(),
                attachment_date_arg_name="date",
            ),
            [io.BytesIO()],
            False,
            id="attachment_with_date",
        ),
        pytest.param(
            "attachment_with_datetime",
            dict(
                attachment=lambda time: io.BytesIO(),
                attachment_time_arg_name="time",
            ),
            [io.BytesIO()],
            False,
            id="attachment_with_datetime",
        ),
        pytest.param(
            "multiattach_with_date",
            dict(
                attachment=lambda date: [io.BytesIO(), io.BytesIO()],
                attachment_date_arg_name="date",
            ),
            [io.BytesIO(), io.BytesIO()],
            False,
            id="multiattach_with_date",
        ),
        pytest.param(
            "multiattach_with_datetime",
            dict(
                attachment=lambda time: [io.BytesIO(), io.BytesIO()],
                attachment_time_arg_name="time",
            ),
            [io.BytesIO(), io.BytesIO()],
            False,
            id="multiattach_with_datetime",
        ),
        pytest.param(
            "attachment_with_date_datetime_raise",
            dict(
                attachment=lambda date, time: io.BytesIO(),
                attachment_date_arg_name="date",
                attachment_time_arg_name="time",
            ),
            None,
            True,
            id="attachment_with_date_datetime_raise",
        ),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_get_attachment(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    name: str,
    report_dict: dict,
    etalon_attachments: Optional[List[Union[str, Path, io.BytesIO]]],
    should_raise: bool,
) -> None:
    report = Report("name")
    report.execution_time = CURRENT_DATETIME
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    if should_raise:
        with pytest.raises(ReportError):
            report._get_attachment(report_dict)
    else:
        result_attachments = report._get_attachment(report_dict)
        if etalon_attachments:
            assert len(etalon_attachments) == len(result_attachments), (  # type: ignore
                f"should be {len(etalon_attachments)}, but got {len(result_attachments)}"  # type: ignore  # noqa
            )
            etalon_types = sorted(map(lambda attachment: type(attachment).__name__, etalon_attachments))
            result_types = sorted(map(lambda attachment: type(attachment).__name__, result_attachments))  # type: ignore
            assert etalon_types == result_types, f"should be {etalon_types}, but got {result_types}"
        else:
            assert result_attachments is None, f"should be None, but got {result_attachments}"


@pytest.mark.parametrize(
    "name, report_dict, etalon_receivers",
    [
        pytest.param("multiattach", dict(receivers=TEST_RECEIVER), TEST_RECEIVER, id="multiattach"),
        pytest.param("dummy", dict(receivers=TEST_RECEIVER), TEST_RECEIVER, id="dummy_report_test"),
        pytest.param(
            "test_iOS_14_5_plus",
            dict(receivers=TEST_RECEIVER),
            TEST_RECEIVER,
            id="test_iOS_14_5_plus_report_without_attachment",
        ),
        pytest.param(
            "test_iOS_14_5_plus",
            dict(receivers=TEST_ERROR_RECEIVERS),
            TEST_ERROR_RECEIVERS,
            id="test_iOS_14_5_plus_report_with_attachment",
        ),
        pytest.param("report_package", dict(receivers=TEST_RECEIVER), TEST_RECEIVER, id="report_package"),
        pytest.param("failed_checkers", dict(receivers=TEST_RECEIVER), TEST_RECEIVER, id="failed_checkers"),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_receivers(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    name: str,
    report_dict: dict,
    etalon_receivers: Union[str, list],
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    result_receivers = report._get_receivers(report_dict)
    assert etalon_receivers == result_receivers, f"should be {etalon_receivers}, but got {result_receivers}"


@pytest.mark.parametrize(
    "name, report_dict, etalon_receivers",
    [
        pytest.param("multiattach", dict(error_receivers=TEST_RECEIVER), TEST_RECEIVER, id="multiattach"),
        pytest.param("dummy", dict(), DEFAULT_ERROR_RECEIVER, id="dummy_report_3"),
        pytest.param("dummy", dict(), DEFAULT_ERROR_RECEIVER, id="dummy_report_test"),
        pytest.param(
            "test_iOS_14_5_plus",
            dict(error_receivers=TEST_ERROR_RECEIVERS),
            TEST_ERROR_RECEIVERS,
            id="test_iOS_14_5_plus_report_without_attachment",
        ),
        pytest.param(
            "test_iOS_14_5_plus", dict(), DEFAULT_ERROR_RECEIVER, id="test_iOS_14_5_plus_report_with_attachment"
        ),
        pytest.param("report_package", dict(error_receivers=TEST_RECEIVER), TEST_RECEIVER, id="report_package"),
        pytest.param("failed_checkers", dict(error_receivers=TEST_RECEIVER), TEST_RECEIVER, id="failed_checkers"),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_error_receivers(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    name: str,
    report_dict: dict,
    etalon_receivers: Union[str, list],
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    result_receivers = report._get_error_receivers(report_dict)
    assert etalon_receivers == result_receivers, f"should be {etalon_receivers}, but got {result_receivers}"


@pytest.mark.parametrize(
    "name, report_dict, etalon_params",
    [
        pytest.param("multiattach", dict(), dict(), id="multiattach"),
        pytest.param("dummy", dict(send_params={"dummy": "dummy"}), {"dummy": "dummy"}, id="dummy_report_3"),
        pytest.param("dummy", dict(send_params="dummy"), dict(), id="dummy_report_test"),
        pytest.param(
            "test_iOS_14_5_plus",
            dict(send_params={"markup": "MarkdownV2"}),
            {"markup": "MarkdownV2"},
            id="test_iOS_14_5_plus",
        ),
        pytest.param(
            "failed_checkers",
            dict(send_params={"markup": None}),
            {"markup": None},
            id="failed_checkers_report_check_failed",
        ),
        pytest.param(
            "failed_checkers",
            dict(send_params={"one": 1, "two": 2}),
            {"one": 1, "two": 2},
            id="failed_checkers_report_custom_checker_failed",
        ),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_get_send_params(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    name: str,
    report_dict: dict,
    etalon_params: dict,
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    result_params = report._get_send_params(report_dict)
    assert etalon_params == result_params, f"should be {etalon_params}, but got {result_params}"


@pytest.mark.parametrize(
    "name, content, receivers, subject, attachments, send_params",
    [
        pytest.param(
            "multiattach",
            "some text for multiattach",
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            ["_form_attach", "_form_attach"],
            dict(),
            id="multiattach",
        ),
        pytest.param("dummy", MSG_NOT_A_DICT, DEFAULT_ERROR_RECEIVER, SUBJECT_NOT_SET, [], dict(), id="dummy_report_1_2"),
        pytest.param("dummy", MSG_NOT_ALL_FIELDS, TEST_ERROR_RECEIVERS, SUBJECT_NOT_SET, [], dict(), id="dummy_report_3"),
        pytest.param(
            "dummy",
            MSG_ANOTHER_REPORT_FAILED,
            DEFAULT_ERROR_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="dummy_report_test",
        ),
        pytest.param(
            "test_iOS_14_5_plus",
            "some test text",
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            {"markup": "MarkdownV2"},
            id="test_iOS_14_5_plus_report_without_attachment",
        ),
        pytest.param(
            "test_iOS_14_5_plus",
            "",
            [TEST_RECEIVER, TEST_RECEIVER],
            SUBJECT_NOT_SET,
            ["_prepare_attachment"],
            dict(),
            id="test_iOS_14_5_plus_report_with_attachment",
        ),
        pytest.param(
            "report_package",
            "some test text",
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="report_package_report_content",
        ),
        pytest.param(
            "report_package",
            "",
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            ["attachment_func"],
            dict(),
            id="report_package_report_attachment",
        ),
        pytest.param(
            "failed_checkers",
            MSG_ANOTHER_REPORT_FAILED,
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="failed_checkers_report_nothing_failed",
        ),
        pytest.param(
            "failed_checkers",
            MSG_TABLE_CHECK_FAILED,
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="failed_checkers_report_check_failed",
        ),
        pytest.param(
            "failed_checkers",
            MSG_CUSTOM_CHECK_FAILED,
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="failed_checkers_report_custom_checker_failed",
        ),
    ],
)
@patch("lib.trgetl.sender.MyteamBot.send")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_myteambot_send(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    mock_send: MagicMock,
    name: str,
    content: str,
    receivers: Union[str, list],
    subject: str,
    attachments: Optional[Sequence[Union[str, Path, io.BytesIO]]],
    send_params: dict,
) -> None:
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    report._send(report.sender, content, receivers, subject, attachments, send_params)
    mock_send.assert_called_once_with(content, receivers, subject=subject, attachments=attachments, **send_params)


@pytest.mark.parametrize(
    "name, content, receivers, subject, attachments, send_params",
    [
        pytest.param(
            "multiattach",
            "some text for multiattach",
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            ["_form_attach", "_form_attach"],
            dict(),
            id="multiattach",
        ),
        pytest.param("dummy", MSG_NOT_A_DICT, DEFAULT_ERROR_RECEIVER, SUBJECT_NOT_SET, [], dict(), id="dummy_report_1_2"),
        pytest.param("dummy", MSG_NOT_ALL_FIELDS, TEST_ERROR_RECEIVERS, SUBJECT_NOT_SET, [], dict(), id="dummy_report_3"),
        pytest.param(
            "dummy",
            MSG_ANOTHER_REPORT_FAILED,
            DEFAULT_ERROR_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="dummy_report_test",
        ),
        pytest.param(
            "test_iOS_14_5_plus",
            "some test text",
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            {"markup": "MarkdownV2"},
            id="test_iOS_14_5_plus_report_without_attachment",
        ),
        pytest.param(
            "test_iOS_14_5_plus",
            "",
            [TEST_RECEIVER, TEST_RECEIVER],
            SUBJECT_NOT_SET,
            ["_prepare_attachment"],
            dict(),
            id="test_iOS_14_5_plus_report_with_attachment",
        ),
        pytest.param(
            "report_package",
            "some test text",
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="report_package_report_content",
        ),
        pytest.param(
            "report_package",
            "",
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            ["attachment_func"],
            dict(),
            id="report_package_report_attachment",
        ),
        pytest.param(
            "failed_checkers",
            MSG_ANOTHER_REPORT_FAILED,
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="failed_checkers_report_nothing_failed",
        ),
        pytest.param(
            "failed_checkers",
            MSG_TABLE_CHECK_FAILED,
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="failed_checkers_report_check_failed",
        ),
        pytest.param(
            "failed_checkers",
            MSG_CUSTOM_CHECK_FAILED,
            TEST_RECEIVER,
            SUBJECT_NOT_SET,
            [],
            dict(),
            id="failed_checkers_report_custom_checker_failed",
        ),
    ],
)
@patch("lib.trgetl.sender.Email.send")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_email_send(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    mock_send: MagicMock,
    name: str,
    content: str,
    receivers: Union[str, list],
    subject: str,
    attachments: Optional[Sequence[Union[str, Path, io.BytesIO]]],
    send_params: dict,
) -> None:
    report = Report("name", Email())
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    report._send(report.sender, content, receivers, subject, attachments, send_params)
    mock_send.assert_called_once_with(content, receivers, subject=subject, attachments=attachments, **send_params)


@pytest.mark.parametrize(
    "params, module, is_skip",
    [
        pytest.param(
            {"scripts": {"module_1": {"skip": True}, "module_2": {"skip": False}}}, "module_1", True, id="module_skipped"
        ),
        pytest.param(
            {"scripts": {"module_1": {"skip": True}, "module_2": {"skip": False}}},
            "module_2",
            False,
            id="module_not_skipped",
        ),
        pytest.param(
            {"scripts": {"module_1": {"skip": True}, "module_2": {"skip": False}}},
            "module_3",
            False,
            id="module_not_in_params",
        ),
    ],
)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants")
@patch("lib.trgetl.report.Report._get_callables")
def test_is_module_skipped(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    params: Dict[str, Any],
    module: str,
    is_skip: bool,
) -> None:
    mock_std_params.return_value = params
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()
    skip_answer = report._is_module_skipped(module)
    assert is_skip == skip_answer, f"should be {is_skip}, but got {skip_answer}"


@pytest.mark.parametrize(
    "modules, time, dictionaries, ready, parameters",
    [
        pytest.param(
            ["multiattach"],
            CURRENT_DATETIME,
            [
                {
                    "report_multiattach": {
                        "report_name": "report_multiattach",
                        "name": "multiattach",
                        "sender": MyteamBot.__name__,
                        "content": "some text for multiattach",
                        "attachment": ["_form_attach", "_form_attach"],
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": TEST_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                }
            ],
            True,
            dict(),
            id="multiattach",
        ),
        pytest.param(
            ["dummy"],
            CURRENT_DATETIME,
            [
                {
                    "report_1": {
                        "report_name": "report_1",
                        "name": "dummy",
                        "sender": MyteamBot.__name__,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "raw_error_msg": RAW_MSG_NOT_ALL_FIELDS,
                        "error_msg": MSG_NOT_ALL_FIELDS,
                        "check": False,
                    },
                    "report_2": {
                        "report_name": "report_2",
                        "name": "dummy",
                        "sender": MyteamBot.__name__,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "raw_error_msg": RAW_MSG_NOT_ALL_FIELDS,
                        "error_msg": MSG_NOT_ALL_FIELDS,
                        "check": False,
                    },
                    "report_3": {
                        "report_name": "report_3",
                        "name": "dummy",
                        "sender": MyteamBot.__name__,
                        "error_receivers": TEST_ERROR_RECEIVERS,
                        "subject": SUBJECT_NOT_SET,
                        "raw_error_msg": RAW_MSG_NOT_ALL_FIELDS,
                        "error_msg": MSG_NOT_ALL_FIELDS,
                        "check": False,
                    },
                    "report_test": {
                        "report_name": "report_test",
                        "name": "test",
                        "sender": MyteamBot.__name__,
                        "content": "sometext",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                }
            ],
            False,
            dict(),
            id="dummy",
        ),
        pytest.param(
            ["test_iOS_14_5_plus"],
            CURRENT_DATETIME,
            [
                {
                    "report_without_attachment": {
                        "report_name": "report_without_attachment",
                        "name": "test_iOS_14_5_plus",
                        "sender": MyteamBot.__name__,
                        "content": "_form_report_text",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": TEST_ERROR_RECEIVERS,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": {"markup": "MarkdownV2"},
                    },
                    "report_with_attachment": {
                        "report_name": "report_with_attachment",
                        "name": "test_iOS_14_5_plus",
                        "sender": MyteamBot.__name__,
                        "content": "",
                        "attachment": ["_prepare_attachment"],
                        "receivers": [TEST_RECEIVER, TEST_RECEIVER],
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                }
            ],
            True,
            dict(),
            id="test_iOS_14_5_plus",
        ),
        pytest.param(
            ["report_1", "report_2"],
            CURRENT_DATETIME,
            [
                {
                    "report_content_1": {
                        "report_name": "report_content_1",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_1",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                    "report_attachment": {
                        "report_name": "report_attachment",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "",
                        "attachment": ["attachment_func"],
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
                {
                    "report_content_2": {
                        "report_name": "report_content_2",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_2",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
            ],
            True,
            dict(),
            id="report_package",
        ),
        pytest.param(
            ["failed_checkers"],
            CURRENT_DATETIME,
            [
                {
                    "report_nothing_failed": {
                        "report_name": "report_nothing_failed",
                        "name": "failed_checkers",
                        "sender": MyteamBot.__name__,
                        "content": "all is OK",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                    "report_check_failed": {
                        "report_name": "report_check_failed",
                        "name": "failed_checkers",
                        "sender": MyteamBot.__name__,
                        "error_receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "raw_error_msg": RAW_MSG_NOT_ALL_FIELDS,
                        "error_msg": MSG_NOT_ALL_FIELDS,
                        "check": False,
                    },
                    "report_custom_checker_failed": {
                        "report_name": "report_custom_checker_failed",
                        "name": "failed_checkers",
                        "sender": MyteamBot.__name__,
                        "error_receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "raw_error_msg": RAW_MSG_NOT_ALL_FIELDS,
                        "error_msg": MSG_NOT_ALL_FIELDS,
                        "check": False,
                    },
                }
            ],
            False,
            dict(),
            id="failed_checkers",
        ),
        pytest.param(
            ["report_1", "report_2"],
            CURRENT_DATETIME,
            [
                {
                    "report_content_1": {
                        "report_name": "report_content_1",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_1",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                    "report_attachment": {
                        "report_name": "report_attachment",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "",
                        "attachment": ["attachment_func"],
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
                {
                    "report_content_2": {
                        "report_name": "report_content_2",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_2",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
            ],
            True,
            {"scripts": {"run_order": ["report_1", "report_2"]}},
            id="report_package_order",
        ),
        pytest.param(
            ["report_1", "report_2"],
            CURRENT_DATETIME,
            [
                {
                    "report_content_1": {
                        "report_name": "report_content_1",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_1",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                    "report_attachment": {
                        "report_name": "report_attachment",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "",
                        "attachment": ["attachment_func"],
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
                {
                    "report_content_2": {
                        "report_name": "report_content_2",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_2",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
            ],
            True,
            {"scripts": {"run_order": ["report_2", "report_1"]}},
            id="report_package_revers_order",
        ),
        pytest.param(
            ["report_1", "report_2", "report_3"],
            CURRENT_DATETIME,
            [
                {
                    "report_content_1": {
                        "report_name": "report_content_1",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_1",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                    "report_attachment": {
                        "report_name": "report_attachment",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "",
                        "attachment": ["attachment_func"],
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
                {
                    "report_content_2": {
                        "report_name": "report_content_2",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_2",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
                {
                    "report_content_3": {
                        "report_name": "report_content_3",
                        "name": "report_package",
                        "sender": MyteamBot.__name__,
                        "content": "content_func_3",
                        "attachment": None,
                        "receivers": TEST_RECEIVER,
                        "subject": SUBJECT_NOT_SET,
                        "error_receivers": DEFAULT_ERROR_RECEIVER,
                        "raw_error_msg": RAW_MSG_ALL_IS_OK,
                        "error_msg": MSG_ANOTHER_REPORT_FAILED,
                        "check": True,
                        "send_params": dict(),
                    },
                },
            ],
            True,
            {"scripts": {"run_order": ["report_3", "report_2"]}},
            id="report_package_partly_order",
        ),
    ],
)
@patch("lib.trgetl.database.Clickhouse.insert")
@patch("lib.trgetl.report.Report._send")
@patch("lib.trgetl.report.Report._custom_check", return_value=True)
@patch("lib.trgetl.report.Report._check_tables", return_value=True)
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_modules")
@patch("lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_standard_parameters")
@patch("lib.trgetl.report.Report._get_dag_constants", return_value=dict())
@patch("lib.trgetl.report.Report._get_callables")
def test_run(
    mock_callables: MagicMock,
    mock_dag_constants: MagicMock,
    mock_std_params: MagicMock,
    mock_modules: MagicMock,
    mock_check_tables: MagicMock,
    mock_custom_check: MagicMock,
    mock_send: MagicMock,
    mock_insert: MagicMock,
    modules: List[str],
    time: dt.datetime,
    dictionaries: List[Dict[str, Dict[str, Any]]],
    ready: bool,
    parameters: Dict[str, Any],
) -> None:
    callables: Dict[str, Dict[str, Any]] = dict(zip(modules, dictionaries))
    run_order = parameters.get("scripts", dict()).get("run_order")
    etalon_run_order = run_order + [m for m in modules if m not in run_order] if run_order else modules
    mock_std_params.return_value = parameters
    mock_callables.return_value = callables
    report = Report("name")
    mock_callables.assert_called()
    mock_dag_constants.assert_called()
    mock_std_params.assert_called()
    mock_modules.assert_called()

    def _run(module_name: Optional[str] = None) -> None:
        mock_send.reset_mock()
        mock_insert.reset_mock()
        if not ready:
            with pytest.raises(ReportError):
                report.run(time, module_name)
        else:
            report.run(time, module_name)
        mock_send.assert_called()
        mock_insert.assert_called()
        assert time == report.execution_time, f"execution_time should be {time}, but got {report.execution_time}"
        if module_name:
            all_dictionaries = callables[module_name]
        else:
            all_dictionaries = {k: v for d in [callables[m] for m in etalon_run_order] for k, v in d.items()}
        if ready:
            calls_cnt = len(all_dictionaries)
            assert (
                mock_send.call_count == calls_cnt
            ), f"method _send should be called {calls_cnt}, but got {mock_send.call_count}"
            assert (
                mock_insert.call_count == calls_cnt
            ), f"method insert should be called {calls_cnt}, but got {mock_insert.call_count}"
            calls = [
                call(
                    dictionary["sender"],
                    dictionary["content"],
                    dictionary["receivers"],
                    dictionary["subject"],
                    dictionary["attachment"],
                    dictionary["send_params"],
                )
                for dictionary in all_dictionaries.values()
            ]
            mock_send.assert_has_calls(calls)
        else:
            calls_cnt = len([dictionary for dictionary in all_dictionaries.values() if not dictionary["check"]])
            assert (
                mock_send.call_count == calls_cnt
            ), f"method _send should be called {calls_cnt}, but got {mock_send.call_count}"
            assert (
                mock_insert.call_count == calls_cnt
            ), f"method insert should be called {calls_cnt}, but got {mock_insert.call_count}"
            calls = [
                call(
                    dictionary["sender"],
                    dictionary["error_msg"].format(name="name", report_name=dictionary["report_name"]),
                    dictionary["error_receivers"],
                    dictionary["subject"],
                )
                for dictionary in all_dictionaries.values()
                if not dictionary["check"]
            ]
            mock_send.assert_has_calls(calls)

    for module in modules:
        _run(module)
    _run()
