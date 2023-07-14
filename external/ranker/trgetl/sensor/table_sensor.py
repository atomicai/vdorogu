from .base_sensor import BaseSensor
from .exceptions import SensorError, SensorNotReadyError


class TableSensor(BaseSensor):
    def __init__(self, table_name):
        super().__init__(table_name)
        self.query = self._get_query()

    def _get_query(self):
        try:
            return self.query_path.read_text().strip()
        except AttributeError:
            return None

    def run(self, date=None, date_correction=False):
        date = self._prepare_date(date, date_correction)
        print(f"Starting (date: {date}):\n{self}")
        query = self.query.format(date=date)
        print(f"Query:    {query}")
        try:
            response = self.db.read(query)
        except Exception as e:
            raise SensorError(f"ERROR: {e}")
        if response.shape[0] > 0:
            response = response.iloc[0, 0]
            if response:
                print(f"Success: {response}")
                return True
            else:
                raise SensorNotReadyError(f"FAILURE: Got negative response: {response} for query: {query}")
        else:
            raise SensorNotReadyError(f"ERROR: Empty response for query: {query}")
