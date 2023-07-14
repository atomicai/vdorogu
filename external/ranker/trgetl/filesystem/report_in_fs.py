import re
from itertools import chain
from pathlib import Path
from typing import Dict

import yaml  # type: ignore

from .filesystem import REPORT_PATH, Filesystem, FilesystemDuplicatedError, FilesystemNotFoundError


class ReportInFilesystem(Filesystem):
    AIRFLOW_PARAM_SECTION_NAME = "airflow"
    DEFAULT_SCHEDULE = "0 12 * * *"  # every day at 12:00

    def __init__(self, name: str):
        self.name = name

    def find_report_path(self) -> Path:
        path = list(chain(REPORT_PATH.glob(f"{self.name}.py"), REPORT_PATH.glob(f"{self.name}")))
        if len(path) == 1:
            return path[0]
        elif len(path) > 1:
            raise FilesystemDuplicatedError(f"Several paths found for report {self.name}:\n{path}")
        else:
            raise FilesystemNotFoundError(f"No report with name {self.name} found")

    def get_modules(self) -> Dict[str, Path]:
        report_path = self.find_report_path()
        if report_path.is_dir():
            return {module.stem: module for module in report_path.glob("*.py")}
        else:
            return {report_path.stem: report_path}

    def get_parameters(self) -> dict:
        try:
            metadata_path: Path = self.find_report_path()
        except FileNotFoundError:
            return dict()
        if metadata_path.is_dir():
            conf = yaml.safe_load(open(metadata_path / "conf.yml"))
            return conf if conf else dict()
        content = metadata_path.read_text()
        parameters = re.findall(r'parameter:\s*([^=]+?)\s*=[\s"\']*([^=\n"\']+)', content)
        conf = dict()
        for parameter, value in parameters:
            if parameter == "skip":
                if value in ["true", "True"]:
                    conf["skip"] = True
                else:
                    conf["skip"] = False
            else:
                conf.setdefault(self.AIRFLOW_PARAM_SECTION_NAME, dict())[parameter] = value
        return conf

    def get_standard_parameters(self) -> dict:
        raw_parameters = self.get_parameters()
        parameters = raw_parameters.copy()
        (parameters.setdefault(self.AIRFLOW_PARAM_SECTION_NAME, dict()).setdefault("schedule", self.DEFAULT_SCHEDULE))
        return parameters
