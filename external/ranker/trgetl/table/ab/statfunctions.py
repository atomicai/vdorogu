from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import t


def paired_stat_results(control: pd.Series, test: pd.Series, alpha: float = 0.05) -> dict:
    size_cnt, size_tst, block_size = control.shape[0], test.shape[0], 1
    if (test == 0).all() and (control == 0).all():
        control_mean = test_mean = 0
        control_sum = test_sum = 0
        numerator = 0
        left_bound = right_bound = 0
        pvals = [0]
        mde = 0
    else:
        control_mean, test_mean = np.mean(control, axis=0), np.mean(test, axis=0)
        control_sum, test_sum = np.sum(control, axis=0), np.sum(test, axis=0)
        numerator = control_mean - test_mean
        var_control = np.var(control, axis=0, ddof=1)
        var_test = np.var(test, axis=0, ddof=1)
        denominator = np.sqrt(var_control / size_cnt + var_test / size_tst)
        ddof_num = (var_control / size_cnt + var_test / size_tst) ** 2
        ddof_den = (var_control / size_cnt) ** 2 / (size_cnt - 1) + (var_test / size_tst) ** 2 / (size_tst - 1)
        ddof = ddof_num / ddof_den
        pvals = (
            t.sf(
                np.abs(
                    (numerator / denominator).reshape(
                        block_size,
                    )
                ),
                ddof,
            )
            * 2
        )
        value = t.ppf(1 - alpha / 2, ddof) * denominator
        left_bound, right_bound = numerator - value, numerator + value
        mde = (t.ppf(1 - alpha / 2, ddof) + t.ppf(0.8, ddof)) * denominator
    return dict(
        control_size=size_cnt,
        test_size=size_tst,
        control_mean=control_mean,
        test_mean=test_mean,
        control_sum=control_sum,
        test_sum=test_sum,
        uplift=numerator,
        left_bound=left_bound,
        right_bound=right_bound,
        pval=pvals[0],
        mde=mde,
    )


def linearization(
    control_numerator: pd.Series,
    control_denominator: pd.Series,
    test_numerator: pd.Series,
    test_denominator: pd.Series,
) -> Dict[str, pd.Series]:
    numerator = np.sum(control_numerator, axis=1)
    denominator = np.sum(control_denominator, axis=1)
    k = numerator / denominator
    linearized_control = control_numerator - control_denominator * k
    linearized_test = test_numerator - test_denominator * k
    return dict(
        linearized_control=linearized_control,
        linearized_test=linearized_test,
    )
