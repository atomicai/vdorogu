import abc
import getpass
import os
from pathlib import Path
from typing import Union

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


class Ignitor(abc.ABC):
    def __init__(self):
        self.username = getpass.getuser()

    @abc.abstractmethod
    def connect(self):
        pass


class SparkIgnitor(Ignitor):
    def __init__(
        self,
        python_spark_path: Union[str, Path] = "/usr/bin/python3",
        executor_cores: int = 2,
        executor_instances: int = 20,
        executor_memory: str = "25g",
        executor_memory_overhead: str = "4g",
        app: str = "app",
    ):
        super().__init__()
        os.environ["HADOOP_USER_NAME"] = self.username
        os.environ["PYSPARK_PYTHON"] = str(python_spark_path)
        os.environ["PYSPARK_DRIVER_PYTHON"] = str(python_spark_path)

        self.python_spark_path = str(python_spark_path)
        self.executor_cores = executor_cores
        self.executor_instances = executor_instances
        self.executor_memory = executor_memory
        self.executor_memory_overhead = executor_memory_overhead
        self.app = app

        self.client = None

    def connect(self):
        conf = (
            SparkConf()
            .setMaster("yarn-client")
            .set("spark.yarn.queue", "dev.dm.perf.priority")
            .setAppName(self.app)
            .set("spark.executor.instances", str(self.executor_instances))
            .set("spark.executor.memory", self.executor_memory)
            .set("spark.executor.cores", str(self.executor_cores))
            .set("spark.executorEnv.PYSPARK_PYTHON", str(self.python_spark_path))
            .set("spark.executorEnv.PYSPARK_DRIVER_PYTHON", str(self.python_spark_path))
            .set("spark.yarn.executor.memoryOverhead", str(self.executor_memory_overhead))
        )

        sc = SparkContext(conf=conf)
        # sql_ctx = SQLContext(sc)
        client = SparkSession.builder.getOrCreate()

        self.handler = client.sparkContext

        return self.client


__all__ = ["Ignitor", "SparkIgnitor"]
