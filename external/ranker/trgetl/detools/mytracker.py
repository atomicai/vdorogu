from typing import Any

try:
    from airflow.models import Variable
    from mytracker_export_api import MyTracker
except ImportError:
    pass


class MyTrackerClient:
    def __init__(self):
        self._client = MyTracker(
            api_user_id=Variable.get("trgetl_mytracker_api_user_id_secret"),
            api_secret_key=Variable.get("trgetl_mytracker_api_password_secret"),
            proxies=Variable.get("rbhp_proxies", deserialize_json=True),
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)
