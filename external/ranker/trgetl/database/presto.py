import getpass
import re
import subprocess
from contextlib import contextmanager

import pandas as pd
import prestodb as presto

from .base_database import BaseDatabase


class Presto(BaseDatabase):
    CONN_PARAMETERS = {
        "trg": dict(
            http_scheme="https",
            host="rbhp-control5.rbdev.mail.ru",
            port=8093,
            user="{user}",
            catalog="hive",
        )
    }
    AUTH_PARAMETERS = {
        "trg": dict(
            config="/etc/krb5.conf",
            service_name="presto",
            principal="{principal}",
            ca_bundle="/etc/prestosql/prestosql_cacerts.pem",
        )
    }

    RETRY_ERRORS = {
        "Encountered too many errors talking to a worker node": "Worker node crashed",
    }

    def __init__(self, db=None, retries: int = 10, retry_sleep: int = 10):
        if db is None:
            db = "trg"

        assert db in self.CONN_PARAMETERS, f"Unknown db: {db}"
        self.db = db
        self.auth_parameters, self.conn_parameters = self._get_connection_parameters(db)
        self.retries = retries
        self.retry_sleep = retry_sleep

    def read(self, query: str) -> pd.DataFrame:
        with self._cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            colnames = [e[0] for e in cursor.description]
        df = pd.DataFrame(result, columns=colnames)
        return df

    def _get_connection_parameters(self, db):
        auth_parameters = self.AUTH_PARAMETERS[db].copy()
        conn_parameters = self.CONN_PARAMETERS[db].copy()
        user = getpass.getuser()

        klist = subprocess.run(["klist"], stdout=subprocess.PIPE).stdout.decode()
        principal = re.findall(r"Default principal: (\S*)\n", klist)
        assert len(principal) == 1
        principal = principal[0]

        for parameters in (auth_parameters, conn_parameters):
            for key, value in parameters.items():
                if isinstance(value, str) and "{" in value:
                    parameters[key] = value.format(user=user, principal=principal)
        return auth_parameters, conn_parameters

    @contextmanager
    def _cursor(self):
        auth = presto.auth.KerberosAuthentication(**self.auth_parameters)
        conn = presto.dbapi.connect(auth=auth, **self.conn_parameters)
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.cancel()
            conn.close()


class PrestoError(Exception):
    pass
