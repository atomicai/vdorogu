import pytest
import requests

from ...database import Clickhouse
from .fixture import auth

ch_urls = list(Clickhouse.DB_URLS.values())


@pytest.mark.parametrize('ch_url', ch_urls)
def test_ch(auth, ch_url):
    response = requests.get(ch_url + 'query=select+1', auth=auth)
    assert response.status_code == 200
    assert response.text == '1\n'
