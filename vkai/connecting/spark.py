import abc
import getpass
import os
import sys

from pyspark import SparkConf, SparkContext, SQLContext


class Ignitor(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def connect(self):
        pass


class SparkIgnitor(Ignitor):
    def __init__(self):
        username = getpass.getuser()
        os.environ["HADOOP_USER_NAME"] = username
        os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"


__all__ = ["Ignitor"]
