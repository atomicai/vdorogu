import datetime as dt
import getpass
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Hashable, Optional, Union

from pandas import DataFrame

from .base_database import BaseDatabase
from .hdfs import Hdfs

PYSPARK_2_PATHS = ['/usr/lib/spark/python', '/usr/lib/spark/python/lib/py4j-current-src.zip']
PYSPARK_3_PATHS = ['/usr/lib/spark3/python', '/usr/lib/spark3/python/lib/py4j-current-src.zip']

for pyspark_2_path in PYSPARK_2_PATHS:
    if pyspark_2_path in sys.path:
        sys.path.remove(pyspark_2_path)
for pyspark_3_path in PYSPARK_3_PATHS:
    if pyspark_3_path not in sys.path:
        sys.path.append(pyspark_3_path)

os.environ["PYSPARK_PYTHON"] = "./anaconda3.zip/anaconda3/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "./anaconda3.zip/anaconda3/bin/python"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--archives hdfs:///data/dm/dmdesc/env/anaconda3.zip pyspark-shell"  # noqa: E501
os.environ["PYTHONPATH"] = "/usr/lib/spark3/python/lib/py4j-current-src.zip:/usr/lib/spark3/python/:"  # noqa: E501
os.environ["SPARK_HOME"] = "/usr/lib/spark3/"
# os.environ["PATH"] += ":/usr/lib/spark3/:/usr/lib/spark3/python/lib/py4j-current-src.zip"
os.environ["PATH"] = f"{os.environ.get('PATH')}:/usr/lib/spark3/:/usr/lib/spark3/python/lib/py4j-current-src.zip"

from pyspark.conf import SparkConf  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402


class Spark(BaseDatabase):
    SYSTEM_USERS = ['airflow-trgetl', 'jenkins-trgan']

    def __init__(self, db: str = '', keep_session: bool = True, shuffle_partitions: int = 200, conf: dict = dict()):
        self.db = db
        self.hdfs = Hdfs()

        username = getpass.getuser()
        if username in self.SYSTEM_USERS:
            keep_session = False
        self.keep_session = keep_session
        self.shuffle_partitions = shuffle_partitions
        self.conf = conf

    def execute(self, query: str, appname: Optional[str] = None) -> None:
        with self.spark_session(appname=appname) as session:
            print(session.sql(query))

    def read(self, query: Union[str, tuple], appname: Optional[str] = None) -> DataFrame:
        if isinstance(query, str):
            with self.spark_session(appname=appname) as session:
                return session.sql(query).toPandas()
        elif isinstance(query, tuple):
            module, main_function, arguments = query
            assert isinstance(module, ModuleType), f"module type should be 'module', but got {type(module)}"
            try:
                main = getattr(module, main_function)
            except AttributeError as e:
                raise (e)
            return main(arguments)
        else:
            raise TypeError(f"Expected str or tuple query type, but got {type(query)}")

    def save_as_file(
        self,
        query: str,
        path: Optional[Union[str, Path]] = None,
        format_: str = 'orc',
        appname: Optional[str] = None,
        return_structure: bool = False,
    ) -> Optional[list]:
        if path is None:
            path = self.default_path() / str(dt.datetime.now())
        with self.spark_session(appname=appname) as session:
            df = session.sql(query)
            print('Query executed')
            write = getattr(df.write, format_)
            write(str(path))
            print(f'File saved at {path}')
            if return_structure:
                return df.dtypes
        return None

    def default_path(self) -> Path:
        # username = getpass.getuser()
        # if username in self.SYSTEM_USERS:
        #     return Path('/export/target')
        # else:
        return Path('/data/sandbox/trgetl')

    def insert(  # type: ignore
        self,
        path: Path,
        df: DataFrame,
        as_table: bool = False,
        mode: Optional[str] = None,
        format: str = 'csv',
        appname: Optional[str] = None,
        **options: Hashable,
    ) -> None:
        with self.spark_session(appname=appname) as session:
            spark_df = session.createDataFrame(df)
            spark_df = spark_df.repartition(1)
            writer = spark_df.write.format(format)
            if mode is not None:
                writer = writer.mode(mode)
            for option_name, value in options.items():
                writer = writer.option(option_name, value)
            if as_table:
                writer.saveAsTable(path)
            else:
                writer.save(path)

    def rm(self, path: Path) -> int:
        return self.hdfs.rm(path=path)

    def columns(self, table_name: str, return_dtypes: bool = False) -> list:
        columns = self.read(f'describe table {table_name}')
        columns = (
            columns.rename(columns={'col_name': 'name', 'date_type': 'type'})
            .query("name.str.startswith('#') == False", engine='python')
            .drop_duplicates()
        )
        if return_dtypes:
            return columns
        else:
            return list(columns.name)

    def now_query(self) -> str:
        return "date_trunc('Second', now())"

    @contextmanager
    def spark_session(self, appname: Optional[str]) -> SparkSession:
        session = self.get_session(appname=appname)
        try:
            yield session
        finally:
            if not self.keep_session:
                session.stop()
                print('Session stopped')

    def get_session(self, appname: Optional[str] = None) -> SparkSession:
        start_time = time.time()
        conf = self._get_spark_conf(appname=appname)
        session = SparkSession.builder.config(conf=conf).getOrCreate()

        print(f'Session established; {int(time.time() - start_time)} sec')
        return session

    def stop_session(self) -> None:
        session = self.get_session()
        session.stop()
        print('Session stopped')

    def _get_spark_conf(self, appname: Optional[str] = None) -> SparkConf:
        username = getpass.getuser()
        queue = "root.dev.trg.priority"
        if username in self.SYSTEM_USERS:
            queue = "root.prod.regular"
        if appname is None:
            appname = username

        conf = SparkConf()
        conf.setMaster("yarn")
        conf.setAppName(appname)
        conf.set("spark.yarn.queue", queue)

        conf.set("spark.executor.memory", "16gb")
        conf.set("spark.executor.cores", "4")
        conf.set("spark.driver.memory", "16g")
        conf.set("spark.driver.cores", "4")
        conf.set("spark.driver.maxResultSize", "0")

        conf.set("spark.dynamicAllocation.enabled", "true")
        conf.set("spark.shuffle.service.enabled", "true")
        conf.set("spark.executor.instances", "10")
        conf.set("spark.speculation", "false")
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        conf.set("spark.kryoserializer.buffer.max", "2000m")

        conf.set("hive.exec.dynamic.partition", "true")
        conf.set("hive.exec.dynamic.partition.mode", "nonstrict")

        conf.set("spark.sql.hive.convertMetastoreOrc", "false")
        conf.set("spark.sql.shuffle.partitions", self.shuffle_partitions)
        conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

        conf.set("spark.driver.allowMultipleContext", "true")

        conf.set("spark.sql.sources.parallelPartitionDiscovery.threshold", "0")
        conf.set("spark.sql.execution.arrow.enabled", "true")
        for param, value in self.conf.items():
            conf.set(param, value)

        return conf


def init_spark_session(appname: Optional[str] = None, shuffle_partitions: int = 200, conf: dict = dict()) -> SparkSession:
    spark = Spark(shuffle_partitions=shuffle_partitions, conf=conf)
    with spark.spark_session(appname=appname) as session:
        return session
