from typing import Any, Optional

from .base_database import BaseDatabase
from .clickhouse import Clickhouse
from .hdfs import Hdfs
from .mysql import Mysql
from .postgres import Postgres
from .presto import Presto
from .spark import Spark


class Database(BaseDatabase):
    DATABASES = {
        'ch': (Clickhouse, None),
        'hdfs': (Hdfs, None),
        'mysql': (Mysql, None),
        'pg': (Postgres, None),
        'presto': (Presto, None),
        'spark': (Spark, None),
        'ch_proxy': (Clickhouse, 'vketl'),
        'cx_hub': (Clickhouse, 'cx_hub'),
        'events': (Clickhouse, 'events'),
    }

    def __new__(cls, db_class: str, db: Optional[str] = None, *args: Any, **kwargs: Any) -> BaseDatabase:  # type: ignore
        DBClass, db_val = cls.DATABASES[db_class]
        return DBClass(db or db_val, *args, **kwargs)
