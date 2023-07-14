import datetime as dt
import logging
from typing import Hashable, Optional, Union

import pandas as pd

from ..database import Database
from ..filesystem import TableInFilesystem
from ..table.helpers import query_parameters
from .base_checker import BaseChecker
from .exceptions import CheckerError

logger = logging.getLogger("Checker")


class ComparisonChecker(BaseChecker):
    PRESET: dict = {
        "unique_key": {"metric": "count(*)", "source_metric": "count(distinct {key_column})"},
        "dict": {
            "metric": "count(*)",
            "condition": "{dict_field} != dictGet('{dict_name}', '{dict_field}', {dict_key})",
            "fixed_source": 0,
        },
        "events_amount": {
            "metric": "sum(amount)",
            "source_metric": "sum(sign * (a_amount + tps_a_amount + n_amount + tps_n_amount)) / 1e8",
            "source_table_name": "target.events",
            "source_db": ("ch", "events"),
            "source_condition": "charging_status_bit = 0",
        },
    }

    def __init__(self, table_name: str, **checker_parameters: Hashable):
        if "preset" in checker_parameters:
            preset_name = checker_parameters["preset"]
            preset = self.PRESET[preset_name].copy()
            preset.update(checker_parameters)
            checker_parameters = preset
            checker_parameters["checker_name"] = preset_name

        super().__init__(table_name, **checker_parameters)

        self.self_control_value: int
        self.source_control_value: int

        self.self_plot_metric: Optional[int] = None
        self.source_plot_metric: Optional[int] = None

        if "metric" not in self.parameters and "query" not in self.parameters:
            raise CheckerError("Metric or query must be set for ComparisonChecker")
        if "fixed_source" not in self.parameters:
            self.source_db = Database(*self.parameters["source_db"])

    def run(self, date: Union[str, dt.date]) -> bool:
        if self.parameters.get("skip", False):
            logger.info("Skip checker: %s", repr(self))
            return True
        if (
            self.parameters.get("checker_name", "") != "dump_size"
            and "source_table_name" in self.parameters
            and (
                self.parameters["source_table_name"]
                not in self.table_filesystem_representation.extract_all_dependencies()
            )
        ):
            logger.info("Skip senseless indirect checker: %s", repr(self))
            return True
        logger.info("Starting (date: %s):\n%s", date, repr(self))

        self._form_control_values(date)

        if self.parameters.get("plot", False):
            if "plot_metric" in self.parameters:
                self._form_plot_values(date)
            if "plot_source_metric" in self.parameters:
                self._form_plot_values(date, True)

        return self._controls_equal()

    def _form_control_values(self, date: Union[str, dt.date]) -> None:
        self_control_query = self._get_self_control_query(date)
        logger.info("Self query:    %s", self_control_query)
        self_control_df = self.db.read(self_control_query)
        self.self_control_value = self._extract_value_from_dataframe(self_control_df)

        if "fixed_source" in self.parameters:
            self.source_control_value = self.parameters["fixed_source"]
        else:
            source_control_query = self._get_source_control_query(date)
            logger.info("Source query:    %s", source_control_query)
            source_control_df = self.source_db.read(source_control_query)
            assert self._check_source_count(
                source_control_df
            ), f"seems like {self.parameters['source_table_name']} for the date: {date}, is empty"
            self.source_control_value = self._extract_value_from_dataframe(source_control_df)

    def _form_plot_values(self, date: Union[str, dt.date], from_source: bool = False) -> None:
        plot_query = self._get_plot_query(date, from_source)
        logger.info("%s plot query:    %s", "Source" if from_source else "Self", plot_query)
        plot_df = self.db.read(plot_query)
        if from_source:
            self.source_plot_metric = self._extract_value_from_dataframe(plot_df)
        else:
            self.self_plot_metric = self._extract_value_from_dataframe(plot_df)

    def _extract_value_from_dataframe(self, control_df: pd.DataFrame) -> int:
        control_df.fillna(0, inplace=True)
        assert control_df.shape == (1, 1), f"self control returned dataframe of shape {control_df.shape}"
        return round(control_df.iloc[0, 0])

    def _check_source_count(self, source_control_value: pd.DataFrame) -> bool:
        try:
            name, value = next(source_control_value.items())
            assert value.shape == (1,), f"source control returned dataframe of shape {value.shape}"
            if "count" in name and value[0] == 0:
                return False
            else:
                return True
        except StopIteration as e:
            raise e

    def _get_plot_query(self, date: Union[str, dt.date], from_source: bool = False) -> str:
        metric = self.parameters["plot_source_metric"] if from_source else self.parameters["plot_metric"]
        table_name = self.parameters.get("plot_source_table_name") if from_source else self.table_name
        query = f"select {metric} "
        if table_name is not None:
            query += f"from {table_name}"
            conditions = []
            date_condition = (
                self.parameters.get("plot_source_date_condition")
                if from_source
                else self.parameters.get("plot_date_condition")
            )
            if date_condition is not None:
                date_condition = date_condition.format(date=date)
            else:
                date_column = (
                    self.table_parameters["source_date_column"] if from_source else self.table_parameters["date_column"]
                )
                date_condition = f"{date_column} = '{date}'"
            conditions.append(date_condition)
            if "plot_source_condition" in self.parameters or "plot_condition" in self.parameters:
                conditions.append(
                    self.parameters["plot_source_condition"] if from_source else self.parameters["plot_condition"]
                )
            if conditions:
                query += " where " + " and ".join(conditions)
        query += " limit 1"
        query = query.replace("{date}", str(date))
        return query

    def _get_self_control_query(self, date: Union[str, dt.date]) -> str:
        if "query" in self.parameters:
            query_name = self.parameters["query"]
            checker_path = self.table_filesystem_representation.checker_path()
            query_path = checker_path.parent / query_name
            query = query_path.read_text()
            query = query.format(date=date)
        else:
            metric = self.parameters["metric"]
            query = f"select {metric} from {self.table_name}"
            conditions = []
            if not self.table_parameters["is_full"]:
                date_condition = self.parameters["date_condition"]
                if date_condition is not None:
                    date_condition = date_condition.format(date=date)
                else:
                    date_column = self.table_parameters["date_column"]
                    date_condition = f"{date_column} = '{date}'"
                conditions.append(date_condition)
            if "condition" in self.parameters:
                conditions.append(self.parameters["condition"])
            if conditions:
                query += " where " + " and ".join(conditions)
        query += " limit 1"
        query = query.replace("{date}", str(date))
        return query

    def _get_source_control_query(self, date: Union[str, dt.date]) -> str:
        source_metric = self.parameters["source_metric"]
        source_table_name = self.parameters["source_table_name"]
        query = f"select {source_metric} from {source_table_name}"
        conditions = []
        if not self.table_parameters["is_full"]:
            date_condition = self.parameters["source_date_condition"]
            if date_condition is not None:
                date_condition = date_condition.format(date=date)
            else:
                date_column = self.table_parameters["source_date_column"]
                date_condition = f"{date_column} = '{date}'"
            conditions.append(date_condition)
        if "source_condition" in self.parameters:
            conditions.append(self.parameters["source_condition"])
        if conditions:
            query += " where " + " and ".join(conditions)
        query += " limit 1"
        query = query.replace("{date}", str(date))
        return query

    def _controls_equal(
        self,
    ) -> bool:
        tolerance = self.parameters["tolerance"]
        equality = (
            self.self_control_value == self.source_control_value
            or abs(self.self_control_value - self.source_control_value)
            / max(self.self_control_value, self.source_control_value)
            <= tolerance
        )
        message = f"{self.self_control_value} vs {self.source_control_value}"
        if tolerance:
            message += f" (margin {tolerance})"
        if equality:
            logger.info("Success: " + message)
        else:
            logger.info("FAILURE: " + message)

        return equality

    def _get_standart_parameters(self, **raw_checker_parameters: Hashable) -> dict:
        checker_parameters = raw_checker_parameters.copy()

        for parname, parameter in checker_parameters.items():
            if isinstance(parameter, str) and "{" in parameter:
                checker_parameters[parname] = parameter.format(**raw_checker_parameters, **query_parameters())

        if "metric" in raw_checker_parameters:
            if "source_metric" not in raw_checker_parameters:
                checker_parameters["source_metric"] = raw_checker_parameters["metric"]

            if "source_table_name" not in raw_checker_parameters:
                checker_parameters["source_table_name"] = self.table_name
            if "source_db" not in raw_checker_parameters:
                source_table_name = checker_parameters["source_table_name"]
                assert isinstance(source_table_name, str)
                source_table = TableInFilesystem(source_table_name)
                checker_parameters["source_db"] = source_table.get_db()

            checker_parameters["source_date_condition"] = raw_checker_parameters.get("source_date_condition")

            checker_parameters["date_condition"] = raw_checker_parameters.get("date_condition")

        checker_parameters["tolerance"] = raw_checker_parameters.get("tolerance", 0)

        return checker_parameters
