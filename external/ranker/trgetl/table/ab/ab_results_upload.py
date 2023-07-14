import datetime as dt
import itertools
from typing import Dict

import pandas as pd

from ...database import Database


class AbReckoner:
    SLICE_COLUMNS = [
        "source",
        "metric",
        "jira_task",
        "platform",
        "project",
        "custom_slice",
        "user_id_type",
        "experiment_group_id_type",
    ]

    def __init__(
        self,
        table_name: str,
        db: Database,
    ) -> None:
        self.table_name = table_name
        self.db = db

    def calculate_and_upload(
        self,
        queries: Dict[str, str],
        date: dt.date,
    ) -> int:
        slices = self.db.read(queries["slices"])
        rownum = 0

        for index, slice_ in slices.fillna("").groupby(self.SLICE_COLUMNS):
            index = {colname: value for colname, value in zip(self.SLICE_COLUMNS, index)}

            test_group_ids = slice_.query("group_type == 'test'").experiment_group_id
            control_group_ids = slice_.query("group_type == 'control'").experiment_group_id
            assert len(test_group_ids) != 0, "no test groups"
            assert len(control_group_ids) != 0, "no control groups"

            if len(test_group_ids) > 1 or len(control_group_ids) > 1:
                all_test_group_ids = ",".join(test_group_ids.sort_values().astype(str))
                all_control_group_ids = ",".join(control_group_ids.sort_values().astype(str))
                result_data = self._ab_results_line(
                    metrics_query=queries["metrics"],
                    date=date,
                    constants=index,
                    test_group_id=all_test_group_ids,
                    control_group_id=all_control_group_ids,
                )
                rownum += self.db.insert(self.table_name, result_data)
                print(f"{rownum} rows inserted...")

            test_description = slice_.query("group_type == 'test'").description.iloc[0]
            control_description = slice_.query("group_type == 'control'").description.iloc[0]

            test_control_combinations = itertools.product(test_group_ids, control_group_ids)
            for test_group_id, control_group_id in test_control_combinations:
                result_data = self._ab_results_line(
                    metrics_query=queries["metrics"],
                    date=date,
                    constants=index,
                    test_group_id=test_group_id,
                    control_group_id=control_group_id,
                    test_description=test_description,
                    control_description=control_description,
                )
                rownum += self.db.insert(self.table_name, result_data)
                print(f"{rownum} rows inserted...")

        return rownum

    def _ab_results_line(
        self,
        metrics_query: str,
        date: dt.date,
        constants: dict,
        test_group_id: str,
        control_group_id: str,
        test_description: str = "",
        control_description: str = "",
    ) -> pd.DataFrame:
        from .statfunctions import paired_stat_results

        print("Calculating:", constants["metric"], test_group_id, control_group_id)
        test = self.db.read(
            metrics_query.format(
                metric=constants["metric"],
                user_id_type=constants["user_id_type"],
                experiment_group_id_type=constants["experiment_group_id_type"],
                experiment_ids=test_group_id,
            )
        )["value"]
        control = self.db.read(
            metrics_query.format(
                metric=constants["metric"],
                user_id_type=constants["user_id_type"],
                experiment_group_id_type=constants["experiment_group_id_type"],
                experiment_ids=control_group_id,
            )
        )["value"]

        stat_result: dict = paired_stat_results(test=test, control=control)

        result_data = constants.copy()
        result_data.update(stat_result)
        result_data["date"] = date
        result_data["test_group_id"] = test_group_id
        result_data["control_group_id"] = control_group_id
        result_data["test_description"] = test_description
        result_data["control_description"] = control_description

        result_data = pd.DataFrame(result_data, index=[0])
        result_data = result_data[self.db.columns(self.table_name)]
        return result_data
