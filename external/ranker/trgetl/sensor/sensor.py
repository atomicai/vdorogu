from ..filesystem import TableInFilesystem
from .base_sensor import BaseSensor
from .file_sensor import FileSensor
from .table_sensor import TableSensor


class Sensor(BaseSensor):
    SENSORS = {'file': FileSensor, 'table': TableSensor}

    def __new__(cls, name):
        sensor_type = cls._find_type(name)
        SensorClass = cls.SENSORS.get(sensor_type)
        if SensorClass:
            return SensorClass(name)
        else:
            return super(Sensor, cls).__new__(cls)

    @classmethod
    def _find_type(cls, table_name):
        filesystem_representation = TableInFilesystem(table_name)
        table_parameters = filesystem_representation.get_standart_parameters()
        if table_parameters.get('as_file'):
            return 'file'
        else:
            return 'table'
