import pytest
import requests_kerberos


@pytest.fixture
def auth():
    return requests_kerberos.HTTPKerberosAuth(mutual_authentication=requests_kerberos.DISABLED)
