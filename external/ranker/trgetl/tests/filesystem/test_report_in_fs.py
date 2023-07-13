import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Set

import pytest
from mock import patch  # type: ignore
from mock.mock import MagicMock  # type: ignore

from ...filesystem import REPORT_PATH, Filesystem, FilesystemDuplicatedError, FilesystemNotFoundError, ReportInFilesystem


@pytest.mark.parametrize(
    'report_file_paths, report_package_paths, side_effect, etalon_answer',
    [
        pytest.param(
            [
                REPORT_PATH / 'report_1.py',
                REPORT_PATH / 'report_2.py',
                REPORT_PATH / 'report_3.py',
            ],
            [
                REPORT_PATH / 'report_package_1',
                REPORT_PATH / 'report_package_2',
                REPORT_PATH / 'report_package_3',
            ],
            [False, False, False, True, True, True],
            {
                'report_1',
                'report_2',
                'report_3',
                'report_package_1',
                'report_package_2',
                'report_package_3',
            },
            id='normal_reports',
        ),
        pytest.param(
            [
                REPORT_PATH / 'report_1.py',
                REPORT_PATH / 'report_2.py',
                REPORT_PATH / 'report_3.py',
                REPORT_PATH / 'query_1.sql',
                REPORT_PATH / 'query_2.sql',
                REPORT_PATH / 'conf_1.json',
                REPORT_PATH / 'conf_2.yaml',
            ],
            [
                REPORT_PATH / 'report_package_1',
                REPORT_PATH / 'report_package_2',
                REPORT_PATH / 'report_package_3',
                REPORT_PATH / '__pycache__',
                REPORT_PATH / '__tokens__',
                REPORT_PATH / '.vscode',
                REPORT_PATH / '.idea',
                REPORT_PATH / '.DS_Store',
                REPORT_PATH / '.mypy_cache',
                REPORT_PATH / '.pytest_cache',
            ],
            [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True],
            {
                'report_1',
                'report_2',
                'report_3',
                'report_package_1',
                'report_package_2',
                'report_package_3',
            },
            id='reports_exclude_garbage',
        ),
        pytest.param(
            [
                REPORT_PATH / 'report_1.py',
                REPORT_PATH / 'report_2.py',
                REPORT_PATH / 'report_3.py',
            ],
            [
                REPORT_PATH / 'report_package_1',
                REPORT_PATH / 'report_package_2',
                REPORT_PATH / 'report_package_3',
                REPORT_PATH / 'report_1.py',
                REPORT_PATH / 'query_1.sql',
                REPORT_PATH / 'conf_1.json',
                REPORT_PATH / 'conf_2.yaml',
                REPORT_PATH / 'default.banner_pad_d1',
            ],
            [False, False, False, True, True, True, True, True, True, True, True],
            {
                'report_1',
                'report_2',
                'report_3',
                'report_package_1',
                'report_package_2',
                'report_package_3',
                'report_1.py',
                'query_1.sql',
                'conf_1.json',
                'conf_2.yaml',
                'default.banner_pad_d1',
            },
            id='reports_strange_legal_dirs',
        ),
    ],
)
@patch('pathlib.Path.is_dir')
def test_all_reports(
    mock_is_dir: MagicMock,
    report_file_paths: List[Path],
    report_package_paths: List[Path],
    side_effect: List[bool],
    etalon_answer: Set[str],
) -> None:
    def _find_report_paths_side_effect(path_template: str) -> List[Path]:
        if path_template == '*':
            return report_file_paths + report_package_paths
        return []

    mock_glob = MagicMock(side_effect=_find_report_paths_side_effect)
    mock_is_dir.side_effect = side_effect
    with patch('pathlib.Path.glob', new=mock_glob):
        all_reports_result = Filesystem().all_reports()
    mock_glob.assert_called()
    assert mock_glob.call_count == 1, (
        'should be exectly one call for `pathlib.Path.glob`, ' f'but got {mock_glob.call_count} calls'
    )
    mock_is_dir.assert_called()
    assert mock_is_dir.call_count == len(side_effect), (
        f'should be exectly {len(side_effect)} calls for `pathlib.Path.is_dir`, ' f'but got {mock_glob.call_count} calls'
    )

    assert all_reports_result == etalon_answer, f"""should be {etalon_answer}, but got {all_reports_result}"""


def test_ReportInFilesystem_initialization() -> None:
    result = ReportInFilesystem('name')
    assert isinstance(result, ReportInFilesystem), f'should be {ReportInFilesystem}, but got {type(result)}'


@pytest.mark.parametrize(
    'name, report_paths, etalon_path',
    [
        pytest.param('report_1', [REPORT_PATH / 'report_1.py'], REPORT_PATH / 'report_1.py', id='report_file'),
        pytest.param(
            'report_package', [REPORT_PATH / 'report_package'], REPORT_PATH / 'report_package', id='report_package'
        ),
        pytest.param(
            'report_1',
            [
                REPORT_PATH / 'report_1.py',
                REPORT_PATH / 'report_1',
                REPORT_PATH / 'report_1.asd',
            ],
            None,
            id='multiple_report_files',
        ),
        pytest.param('report_1', [], None, id='no_report_files'),
        pytest.param(
            'report_1',
            [
                REPORT_PATH / 'report_1.py',
                REPORT_PATH / 'report_1_source',
            ],
            REPORT_PATH / 'report_1.py',
            id='report_with_source',
        ),
        pytest.param(
            'report_package',
            [
                REPORT_PATH / 'report_package',
                REPORT_PATH / 'report_package_',
            ],
            REPORT_PATH / 'report_package',
            id='very_similar_reports',
        ),
    ],
)
def test_find_report_path(name: str, report_paths: List[Path], etalon_path: Path) -> None:
    def find_report_path_call() -> Path:
        mock_glob = MagicMock(
            side_effect=lambda path_template: iter([path for path in report_paths if path.name == path_template])
        )
        with patch('pathlib.Path.glob', new=mock_glob):
            result_path = ReportInFilesystem(name).find_report_path()
        mock_glob.assert_called()
        assert mock_glob.call_count == 2, (
            'should be exectly two calls for `pathlib.Path.glob`, ' f'but got {mock_glob.call_count} calls'
        )
        return result_path

    if etalon_path is None:
        with pytest.raises((FilesystemDuplicatedError, FilesystemNotFoundError)):
            find_report_path_call()
    else:
        result_path = find_report_path_call()
        assert etalon_path == result_path, f'should be {etalon_path} but got {result_path}'


@pytest.mark.parametrize(
    'name, report_path, report_files, is_dir, etalon_modules',
    [
        pytest.param(
            'report_1',
            REPORT_PATH / 'report_1.py',
            [REPORT_PATH / 'report_1.py'],
            False,
            {'report_1': REPORT_PATH / 'report_1.py'},
            id='single_file_report',
        ),
        pytest.param(
            'report_package',
            REPORT_PATH / 'report_package',
            [
                REPORT_PATH / 'report_package/report_1.py',
                REPORT_PATH / 'report_package/report_2.py',
                REPORT_PATH / 'report_package/report_3.py',
            ],
            True,
            {
                'report_1': REPORT_PATH / 'report_package/report_1.py',
                'report_2': REPORT_PATH / 'report_package/report_2.py',
                'report_3': REPORT_PATH / 'report_package/report_3.py',
            },
            id='py_files_only_package',
        ),
        pytest.param(
            'report_package',
            REPORT_PATH / 'report_package',
            [
                REPORT_PATH / 'report_package/report_1.py',
                REPORT_PATH / 'report_package/report_2.py',
                REPORT_PATH / 'report_package/report_3.py',
                REPORT_PATH / 'report_package/conf.yaml',
                REPORT_PATH / 'report_package/query.sql',
                REPORT_PATH / 'report_package/source',
                REPORT_PATH / 'report_package/source/query_1.sql',
                REPORT_PATH / 'report_package/source/query_2.sql',
                REPORT_PATH / 'report_package/source/hidden_report.py',
            ],
            True,
            {
                'report_1': REPORT_PATH / 'report_package/report_1.py',
                'report_2': REPORT_PATH / 'report_package/report_2.py',
                'report_3': REPORT_PATH / 'report_package/report_3.py',
            },
            id='mixed_files_package',
        ),
    ],
)
@patch('lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.find_report_path')
@patch('pathlib.Path.is_dir')
def test_get_modules(
    mock_is_dir: MagicMock,
    mock_report_path: MagicMock,
    name: str,
    report_path: Path,
    report_files: List[Path],
    is_dir: bool,
    etalon_modules: Dict[str, Path],
) -> None:
    def _find_module_paths_side_effect(path_template: str) -> List[Path]:
        if path_template == '*.py':
            return [report for report in report_files if report.parent == report_path and report.name.endswith('.py')]
        return []

    mock_report_path.return_value = report_path
    mock_is_dir.return_value = is_dir
    mock_glob = MagicMock(side_effect=_find_module_paths_side_effect)
    with patch('pathlib.Path.glob', new=mock_glob):
        modules_result = ReportInFilesystem(name).get_modules()
    if is_dir:
        mock_glob.assert_called()
        assert mock_glob.call_count == 1, (
            'should be exectly one call for `pathlib.Path.glob`, ' f'but got {mock_glob.call_count} calls'
        )
    mock_is_dir.assert_called()
    assert mock_is_dir.call_count == 1, (
        f'should be exectly one call for `pathlib.Path.is_dir`, ' f'but got {mock_glob.call_count} calls'
    )
    assert modules_result == etalon_modules, f"""should be {etalon_modules}, but got {modules_result}"""


@pytest.mark.parametrize(
    'name, report_path, is_dir, text, etalon_params',
    [
        pytest.param(
            'report_1',
            REPORT_PATH / 'report_1.py',
            False,
            dedent(
                """\
                parameter:chat = d.kulemin@corp.mail.ru
                parameter:schedule = */5 * * * *
                parameter:start_date = 2023-04-13
                parameter:sender = myteambot
                parameter:skip = False

                import datetime as dt
                from textwrap import dedent

                from trgetl.database import Clickhouse
                from trgetl.sender import MyteamBot
            """
            ),
            {
                'skip': False,
                'airflow': {
                    'chat': 'd.kulemin@corp.mail.ru',
                    'schedule': '*/5 * * * *',
                    'start_date': '2023-04-13',
                    'sender': 'myteambot',
                },
            },
            id='single_file_report_skip_False',
        ),
        pytest.param(
            'report_1',
            REPORT_PATH / 'report_1.py',
            False,
            dedent(
                """\
                parameter:chat = d.kulemin@corp.mail.ru
                parameter:schedule = */5 * * * *
                parameter:start_date = 2023-04-13
                parameter:sender = myteambot
                parameter:skip = false

                import datetime as dt
                from textwrap import dedent

                from trgetl.database import Clickhouse
                from trgetl.sender import MyteamBot
            """
            ),
            {
                'skip': False,
                'airflow': {
                    'chat': 'd.kulemin@corp.mail.ru',
                    'schedule': '*/5 * * * *',
                    'start_date': '2023-04-13',
                    'sender': 'myteambot',
                },
            },
            id='single_file_report_skip_false',
        ),
        pytest.param(
            'report_1',
            REPORT_PATH / 'report_1.py',
            False,
            dedent(
                """\
                parameter:chat = d.kulemin@corp.mail.ru
                parameter:schedule = */5 * * * *
                parameter:start_date = 2023-04-13
                parameter:sender = myteambot
                parameter:skip = True

                import datetime as dt
                from textwrap import dedent

                from trgetl.database import Clickhouse
                from trgetl.sender import MyteamBot
            """
            ),
            {
                'skip': True,
                'airflow': {
                    'chat': 'd.kulemin@corp.mail.ru',
                    'schedule': '*/5 * * * *',
                    'start_date': '2023-04-13',
                    'sender': 'myteambot',
                },
            },
            id='single_file_report_skip_True',
        ),
        pytest.param(
            'report_1',
            REPORT_PATH / 'report_1.py',
            False,
            dedent(
                """\
                parameter:chat = d.kulemin@corp.mail.ru
                parameter:schedule = */5 * * * *
                parameter:start_date = 2023-04-13
                parameter:sender = myteambot
                parameter:skip = true

                import datetime as dt
                from textwrap import dedent

                from trgetl.database import Clickhouse
                from trgetl.sender import MyteamBot
            """
            ),
            {
                'skip': True,
                'airflow': {
                    'chat': 'd.kulemin@corp.mail.ru',
                    'schedule': '*/5 * * * *',
                    'start_date': '2023-04-13',
                    'sender': 'myteambot',
                },
            },
            id='single_file_report_skip_true',
        ),
        pytest.param(
            'report_1',
            REPORT_PATH / 'report_1.py',
            False,
            dedent(
                """\
                import datetime as dt
                from textwrap import dedent

                from trgetl.database import Clickhouse
                from trgetl.sender import MyteamBot
            """
            ),
            {},
            id='file_report_no_params',
        ),
        pytest.param(
            'report_package',
            REPORT_PATH / 'report_package',
            True,
            dedent(
                """\
                airflow:
                    schedule:
                        6 10 * * *
                    start_date:
                        2023-04-13
                    sender:
                        class: lib.trgetl.sender.MyteamBot
                    error_receivers:
                        - d.kulemin@corp.mail.ru
                        - error_receiver@corp.mail.ru
            """
            ),
            {
                'airflow': {
                    'schedule': '6 10 * * *',
                    'start_date': datetime.date(2023, 4, 13),
                    'sender': {'class': 'lib.trgetl.sender.MyteamBot'},
                    'error_receivers': ['d.kulemin@corp.mail.ru', 'error_receiver@corp.mail.ru'],
                }
            },
            id='report_package',
        ),
        pytest.param('report_package', REPORT_PATH / 'report_package', True, '', {}, id='report_package_no_params'),
    ],
)
@patch('lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.find_report_path')
@patch('pathlib.Path.is_dir')
@patch('pathlib.Path.read_text')
def test_get_parameters(
    mock_read_text: MagicMock,
    mock_is_dir: MagicMock,
    mock_report_path: MagicMock,
    name: str,
    report_path: Path,
    is_dir: bool,
    text: str,
    etalon_params: Dict[str, Any],
) -> None:
    def _open_side_effect(conf_path: Path) -> str:
        if conf_path.name == 'conf.yml':
            return text
        return ''

    mock_open = MagicMock(side_effect=_open_side_effect)
    mock_read_text.return_value = text
    mock_is_dir.return_value = is_dir
    mock_report_path.return_value = report_path
    with patch('builtins.open', new=mock_open):
        result_params = ReportInFilesystem(name).get_parameters()
    mock_report_path.assert_called()
    assert mock_report_path.call_count == 1, (
        f'should be exectly one call for `ReportInFilesystem.find_report_path`, '
        f'but got {mock_report_path.call_count} calls'
    )
    mock_is_dir.assert_called()
    assert mock_is_dir.call_count == 1, (
        f'should be exectly one call for `pathlib.Path.is_dir`, ' f'but got {mock_is_dir.call_count} calls'
    )
    if is_dir:
        mock_open.assert_called()
        assert mock_open.call_count == 1, (
            f'should be exectly one call for `builtins.open`, ' f'but got {mock_open.call_count} calls'
        )
    else:
        mock_read_text.assert_called()
        assert mock_read_text.call_count == 1, (
            f'should be exectly one call for `pathlib.Path.read_text`, ' f'but got {mock_read_text.call_count} calls'
        )
    assert etalon_params == result_params, f'should be {etalon_params}, but got {result_params}'


@pytest.mark.parametrize(
    'name, raw_parameters, etalon_params',
    [
        pytest.param(
            'report_1',
            {
                'skip': True,
                'airflow': {
                    'chat': 'd.kulemin@corp.mail.ru',
                    'schedule': '*/5 * * * *',
                    'start_date': '2023-04-13',
                    'sender': 'myteambot',
                },
            },
            {
                'skip': True,
                'airflow': {
                    'chat': 'd.kulemin@corp.mail.ru',
                    'schedule': '*/5 * * * *',
                    'start_date': '2023-04-13',
                    'sender': 'myteambot',
                },
            },
            id='file_report',
        ),
        pytest.param(
            'report_1',
            {
                'skip': True,
                'airflow': {
                    'chat': 'd.kulemin@corp.mail.ru',
                    'start_date': '2023-04-13',
                    'sender': 'myteambot',
                },
            },
            {
                'skip': True,
                'airflow': {
                    'chat': 'd.kulemin@corp.mail.ru',
                    'schedule': '0 12 * * *',
                    'start_date': '2023-04-13',
                    'sender': 'myteambot',
                },
            },
            id='file_report_default_schedule',
        ),
        pytest.param(
            'report_package',
            {
                'airflow': {
                    'schedule': '6 10 * * *',
                    'start_date': datetime.date(2023, 4, 13),
                    'sender': {'class': 'MyteamBot'},
                    'error_receivers': ['d.kulemin@corp.mail.ru', 'error_receiver@corp.mail.ru'],
                }
            },
            {
                'airflow': {
                    'schedule': '6 10 * * *',
                    'start_date': datetime.date(2023, 4, 13),
                    'sender': {'class': 'MyteamBot'},
                    'error_receivers': ['d.kulemin@corp.mail.ru', 'error_receiver@corp.mail.ru'],
                }
            },
            id='report_package',
        ),
        pytest.param(
            'report_package',
            {
                'airflow': {
                    'start_date': datetime.date(2023, 4, 13),
                    'sender': {'class': 'MyteamBot'},
                    'error_receivers': ['d.kulemin@corp.mail.ru', 'error_receiver@corp.mail.ru'],
                }
            },
            {
                'airflow': {
                    'schedule': '0 12 * * *',
                    'start_date': datetime.date(2023, 4, 13),
                    'sender': {'class': 'MyteamBot'},
                    'error_receivers': ['d.kulemin@corp.mail.ru', 'error_receiver@corp.mail.ru'],
                }
            },
            id='report_package_default_schedule',
        ),
    ],
)
@patch('lib.trgetl.filesystem.report_in_fs.ReportInFilesystem.get_parameters')
def test_get_standard_parameters(
    mock_get_params: MagicMock, name: str, raw_parameters: Dict[str, Any], etalon_params: Dict[str, Any]
) -> None:
    mock_get_params.return_value = raw_parameters
    result_params = ReportInFilesystem(name).get_standard_parameters()
    assert etalon_params == result_params, f'should be {etalon_params}, but got {result_params}'
