import abc
import getpass
import os
import sys

from pyspark import SparkConf, SparkContext, SQLContext

username = getpass.getuser()
os.environ['HADOOP_USER_NAME'] = username
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"

conf = (
    SparkConf()
    .setMaster("yarn-client")
    .set("spark.yarn.queue", 'dev.dm.perf.priority')
    .setAppName('TRG-104980')
    .set('spark.executor.instances', '100')
    .set("spark.executor.memory", "10g")
    .set("spark.executor.cores", "2")
    .set('spark.executorEnv.PYSPARK_PYTHON', '/usr/bin/python3')
    .set('spark.executorEnv.PYSPARK_DRIVER_PYTHON', '/usr/bin/python3')
    .set('spark.yarn.executor.memoryOverhead', '20g')
)
sc = SparkContext(conf=conf)
spark = SQLContext(sc)


import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


class SparkCommand:
    @abc.abstractclassmethod
    def run(self, **kwargs):
        pass


class IQuery(SparkCommand):
    def __init__(self):
        super().__init__()

    def run(self, subdb: str = None, db: str = None, start_date: str = None, end_date: str = None, **kwargs):
        return "\
        select p.search_phrase from ods_target_data.search_phrase_bannerd as p\
        join ods_targetdb_data.remarketing_search_phrases as sp on p.id = sp.id\
        where sp.dt='2023-04-24' and sp.status = 'ready' and p.dt='2023-04-24'\
        "


if __name__ == "__main__":
    pass
