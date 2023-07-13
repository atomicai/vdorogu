import datetime as dt
import importlib.machinery
import io
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import croniter  # type: ignore
import dateutil  # type: ignore

from ..checker import Checker, CheckerFailedError
from ..filesystem import ReportInFilesystem
from ..sender import Email, MyteamBot
from .exceptions import ReportError
from .helpers import log

logger = logging.getLogger("Report")


class Report:
    REQUIRED_FIELDS = {
        "content",
        "receivers",
    }

    DEFAULT_ERROR_RECEIVER = "d.kulemin@corp.mail.ru"
    DEFAULT_PRIORITY = 100

    def __init__(self, name: str, sender: Optional[Union[MyteamBot, Email]] = None):
        self.name = name
        self.filesystem_representation = ReportInFilesystem(name)
        self.modules = self.filesystem_representation.get_modules()
        self.parameters = self.filesystem_representation.get_standard_parameters()
        self._all_objects = self._get_all_module_callables()
        self.callables = self._get_callables()
        self.dag_constants = self._get_dag_constants()
        self.sender = sender or self.dag_constants.get(name, dict()).get("DAG_GLOBAL_SENDER") or MyteamBot()
        self.execution_time: Optional[dt.datetime] = None
        self._setup_logging()

    def __repr__(self) -> str:
        return "{0}('{1}'), global sender: {2},\nparameters: {3},\n{4}".format(
            type(self).__name__,
            self.name,
            type(self.sender).__name__,
            self.parameters,
            "\n\n".join(
                [
                    (
                        "module name: {}, callables:\n\n".format(module)
                        + ",\n\n".join(map(lambda item: "    {}: {}".format(item[0], item[1]()), clbls.items()))
                    )
                    for module, clbls in self.callables.items()
                ]
            ),
        )

    def _setup_logging(self) -> None:
        stream_formatter = logging.Formatter(
            fmt="%(message)s",
        )
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(stream_formatter)

        logger = logging.getLogger("Report")
        logger.setLevel(logging.DEBUG)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(stream_handler)
        logger.propagate = False

    def _get_all_module_callables(self, prefix: str = "") -> Dict[str, Any]:
        module_callables = dict()
        for module, path in self.modules.items():
            report_module = self._import_report_module(name=module, path=path)

            callables = {
                callable_: getattr(report_module, callable_)
                for callable_ in dir(report_module)
                if callable_.startswith(prefix)
            }
            module_callables[module] = callables
        return module_callables

    def _get_callables(self) -> dict:
        report_callables = self._get_objects_by_prefix("report_")
        return report_callables

    def _get_dag_constants(self) -> dict:
        dag_constants = self._get_objects_by_prefix("DAG_")
        return dag_constants

    def _get_objects_by_prefix(self, prefix: str) -> dict:
        module_objects = dict()
        for module, objects in self._all_objects.items():
            module_objects[module] = {name: value for name, value in objects.items() if name.startswith(prefix)}
        return module_objects

    def _import_report_module(self, name: str, path: Path) -> ModuleType:
        module_loader = importlib.machinery.SourceFileLoader(name, str(path))
        module = module_loader.load_module()
        return module

    def _set_execution_time(self, time: Optional[Union[str, dt.datetime]] = None) -> None:
        if time is None:
            current_time = dt.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            cron = croniter.croniter(self.parameters["airflow"]["schedule"], current_time)
            self.execution_time = cron.get_prev(dt.datetime)
        elif isinstance(time, str):
            self.execution_time = dateutil.parser.parse(time)
        else:
            self.execution_time = time

    def run(self, time: Optional[Union[str, dt.datetime]] = None, module_name: Optional[str] = None) -> None:
        self._set_execution_time(time)
        logger.info(f"Starting (time: {self.execution_time}):\n{self.name}")
        if module_name:
            self._run_module(module_name, self.callables[module_name])
        else:
            run_order = self.parameters.get("scripts", dict()).get("run_order")
            if run_order:
                ordered_callables = [(name, self.callables[name]) for name in run_order]
                ordered_callables += [
                    (name, modules) for name, modules in self.callables.items() if name not in run_order
                ]
            else:
                ordered_callables = [(name, modules) for name, modules in self.callables.items()]
            for name, module_callables in ordered_callables:
                self._run_module(name, module_callables)

    def _run_module(self, name: str, module_callables: Dict[str, Callable]) -> None:
        logger.info(f"Module: {name}")
        if self._is_module_skipped(name):
            logger.info("Skip.")
            return
        report_dicts = self._get_callable_out(module_callables)
        report_ready = False
        if all(report_dict["check"] is True for report_dict in report_dicts):
            logger.info("Check is successfull")
            report_ready = True
            for report_dict in report_dicts:
                self.send(report_dict, report_ready)
        else:
            logger.error("Check failed")
            for report_dict in [report_dict for report_dict in report_dicts if not report_dict["check"]]:
                self.send(report_dict, report_ready)
            raise ReportError('ERROR: report "{name}" send failed!'.format(name=self.name))

    def _is_module_skipped(self, module_name: str) -> bool:
        return self.parameters.get("scripts", dict()).get(module_name, dict()).get("skip", False)

    def _get_callable_out(self, module_callables: Dict[str, Any]) -> List[Dict[str, Any]]:
        dictionaries = []
        for report_name, callable_obj in module_callables.items():
            try:
                obj_dict = callable_obj() if callable(callable_obj) else callable_obj
                if self._is_skip(obj_dict):
                    continue
                self._check(obj_dict)
                dictionaries.append(
                    {
                        "report_name": report_name,
                        "name": self.name,
                        "sender": self._get_sender(obj_dict),
                        "content": self._get_content_text(obj_dict),
                        "attachment": self._get_attachment(obj_dict),
                        "receivers": self._get_receivers(obj_dict),
                        "error_receivers": self._get_error_receivers(obj_dict),
                        "subject": self._get_subject(obj_dict),
                        "raw_error_msg": "",
                        "error_msg": self._get_error_text(
                            f'ERROR: report "{self.name}" callable "{report_name}" is OK, '
                            "but another report callables failed"
                        ),
                        "priority": self._get_priority(obj_dict),
                        "check": True,
                        "send_params": self._get_send_params(obj_dict),
                    }
                )
            except ReportError as e:
                raw_error_message = str(e).format(name=self.name, report_name=report_name)
                dictionaries.append(
                    {
                        "report_name": report_name,
                        "name": self.name,
                        "sender": self._get_sender(obj_dict),
                        "error_receivers": self._get_error_receivers(obj_dict),
                        "subject": self._get_subject(obj_dict),
                        "raw_error_msg": raw_error_message,
                        "error_msg": self._get_error_text(raw_error_message),
                        "priority": self._get_priority(obj_dict),
                        "check": False,
                    }
                )
        return sorted(dictionaries, key=lambda obj: obj["priority"])

    @log
    def send(self, report_dict: dict, ready: bool) -> None:
        if ready:
            logger.info(f'Sending report for {report_dict["report_name"]}')
            self._send(
                report_dict["sender"] or self.sender,
                report_dict["content"],
                report_dict["receivers"],
                report_dict["subject"],
                report_dict["attachment"],
                report_dict["send_params"],
            )
        else:
            logger.info(f'Sending error report for {report_dict["name"]}')
            self._send(
                report_dict["sender"] or self.sender,
                report_dict["error_msg"],
                report_dict["error_receivers"],
                report_dict["subject"],
            )

    def _send(
        self,
        sender: Union[MyteamBot, Email],
        content: str,
        receivers: Union[str, list],
        subject: str,
        attachments: Optional[Sequence[Union[str, Path, io.BytesIO]]] = None,
        send_params: dict = dict(),
    ) -> None:
        receivers_represent = ", ".join(receivers) if isinstance(receivers, (list, set)) else receivers
        try:
            sender.send(content, receivers, subject=subject, attachments=attachments, **send_params)
            logger.info("Message sent to {receivers}".format(receivers=receivers_represent))
        except ConnectionRefusedError as e:
            logger.error("Message sent failed with error: {error}".format(error=e))

    def _is_skip(self, report_out: dict) -> bool:
        if isinstance(report_out, dict):
            skip_param = report_out.get("skip", False)
            return skip_param is True
        else:
            return False

    def _check(self, report_out: dict) -> bool:
        if isinstance(report_out, dict):
            if self.REQUIRED_FIELDS.issubset(set(report_out.keys())):
                try:
                    self._check_tables(report_out)
                except CheckerFailedError as e:
                    raise ReportError('ERROR: report "{name}" callable "{report_name}" ' + str(e))
                try:
                    self._custom_check(report_out)
                except Exception as e:
                    raise ReportError('ERROR: report "{name}" callable "{report_name}" ' + str(e))
                return True
            else:
                raise ReportError(
                    'ERROR: report "{name}" callable "{report_name}" output does not contain all required fields'
                )
        else:
            raise ReportError('ERROR: report "{name}" callable "{report_name}" does not return dictionary')

    def _get_name(self, report_dict: dict) -> str:
        if isinstance(report_dict, dict):
            name_obj = report_dict.get("name", None)
            return name_obj if isinstance(name_obj, str) else self.name
        return self.name

    def _get_sender(self, report_dict: dict) -> Optional[Union[MyteamBot, Email]]:
        if isinstance(report_dict, dict):
            return report_dict.get("sender", None)
        return None

    def _get_content_text(self, report_dict: dict) -> str:
        content_obj = report_dict["content"]
        date_arg = report_dict.get("content_date_arg_name", None)
        time_arg = report_dict.get("content_time_arg_name", None)
        if date_arg and time_arg:
            raise ReportError(
                ("ERROR: content function can get date ({date_arg}) or time ({time_arg}) argument, but not both").format(
                    date_arg=date_arg, time_arg=time_arg
                )
            )
        kwargs = dict()
        assert self.execution_time is not None
        if date_arg:
            kwargs[date_arg] = self.execution_time.date()
        if time_arg:
            kwargs[time_arg] = self.execution_time
        return content_obj(**kwargs) if callable(content_obj) else content_obj

    def _get_attachment(self, report_dict: dict) -> Optional[List[Union[str, Path, io.BytesIO]]]:
        attachments = report_dict.get("attachment", None)
        if attachments is None:
            return None
        date_arg = report_dict.get("attachment_date_arg_name", None)
        time_arg = report_dict.get("attachment_time_arg_name", None)
        if date_arg and time_arg:
            raise ReportError(
                (
                    "ERROR: attachment function can get date ({date_arg}) or time ({time_arg}) argument, but not both"
                ).format(date_arg=date_arg, time_arg=time_arg)
            )
        kwargs = dict()
        assert self.execution_time is not None
        if date_arg:
            kwargs[date_arg] = self.execution_time.date()
        if time_arg:
            kwargs[time_arg] = self.execution_time
        if not isinstance(attachments, (list, set, tuple)):
            attachments = [attachments]
        attachment_callable_out_list = []
        for attachment in attachments:
            if callable(attachment):
                attachment = attachment(**kwargs)
            if isinstance(attachment, (list, set, tuple)):
                for attachment_part in attachment:
                    if isinstance(attachment_part, (str, Path, io.BytesIO)):
                        attachment_callable_out_list.append(attachment_part)
            if isinstance(attachment, (str, Path, io.BytesIO)):
                attachment_callable_out_list.append(attachment)
        return attachment_callable_out_list

    def _get_receivers(self, report_dict: dict) -> Union[str, list]:
        return report_dict["receivers"]

    def _get_error_receivers(self, report_dict: dict) -> Union[str, list]:
        return (
            report_dict.get("error_receivers", self.DEFAULT_ERROR_RECEIVER)
            if isinstance(report_dict, dict)
            else self.DEFAULT_ERROR_RECEIVER
        )

    def _get_subject(self, report_dict: dict) -> str:
        if isinstance(report_dict, dict):
            return report_dict.get("subject", "")
        return ""

    def _get_priority(self, report_dict: dict) -> int:
        return (
            report_dict.get("priority", self.DEFAULT_PRIORITY) if isinstance(report_dict, dict) else self.DEFAULT_PRIORITY
        )

    def _get_send_params(self, report_dict: dict) -> dict:
        send_params_obj = report_dict.get("send_params", dict())
        if not isinstance(send_params_obj, dict):
            return dict()
        return send_params_obj

    def _check_tables(self, report_dict: dict) -> bool:
        tables = report_dict.get("check", None)
        if tables is None:
            return True
        try:
            if isinstance(tables, list):
                for table in tables:
                    Checker(table).run()
            else:
                Checker(tables).run()
            return True
        except Exception as e:
            raise CheckerFailedError("table checker failed: {error}".format(error=e))

    def _custom_check(self, report_dict: dict) -> bool:
        custom_checker = report_dict.get("custom_check", None)
        if custom_checker is None:
            return True
        try:
            if isinstance(custom_checker, list):
                for checker in custom_checker:
                    checker()
            else:
                custom_checker()
            return True
        except Exception as e:
            raise CheckerFailedError("custom checker failed: {error}".format(error=e))

    def _get_error_text(self, error_msg: str) -> str:
        return "Report send failed with message:\n" + error_msg
