from typing import Iterable, List, Union

import numpy as np
from catalyst import metrics


class Monitor:
    def __init__(self, classes):
        self.monitor = {key: metrics.AdditiveMetric(compute_on_call=False) for key in list(classes)}

    def update(self, key, value: Union[np.ndarray, List[np.ndarray]], n: int = 1):
        """
        key: the class name to update embedding(s) or single value(s) for
        value: embedding or any single value tensor
        """
        if isinstance(value, np.ndarray) or not isinstance(value, Iterable):
            self.monitor[key].update(value, n)
        else:
            for vec in value:
                self.monitor[key].update(vec, 1)
        return self

    def compute(self, key=None):
        response = {}
        if key is not None:
            return self.monitor[key].compute()[0]
        for key in self.monitor.keys():
            response[key] = self.monitor[key].compute()[0]
        return response


__all__ = ["Monitor"]
