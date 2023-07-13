from ..filesystem import FilesystemNotFoundError, TableInFilesystem
from ..sensor import Sensor
from .custom_table import CustomTable
from .daily_table import DailyTable
from .dump_table import DumpTable
from .transform_table import TransformTable


class Table(DailyTable):
    TABLES = {
        'custom': CustomTable,
        'dump': DumpTable,
        'dump-full': DumpTable,
        'transform': TransformTable,
        'transform-full': TransformTable,
        'sensor': Sensor,
    }

    TABLE_TYPE_DESCRIPTIONS = {
        'custom': 'Таблица со специальным запросом (заливка ежедневная)',
        'dump': 'Дамп таблицы с данными по дням (заливка ежедневная)',
        'dump-full': 'Полный дамп таблицы (заливка ежедневная, старые данные затираются)',
        'transform': 'Трансформация с данными по дням (заливка ежедневная)',
        'transform-full': 'Полная трансформация (заливка ежедневная, старые данные затираются)',
    }

    def __new__(cls, name):
        table_type = cls._find_type(name)
        TableClass = cls.TABLES.get(table_type)
        if TableClass:
            return TableClass(name)
        else:
            return super(Table, cls).__new__(cls)

    @classmethod
    def _find_type(cls, table_name):
        filesystem_representation = TableInFilesystem(table_name)
        table_type = filesystem_representation.get_type()
        try:
            return table_type
        except FilesystemNotFoundError:
            return None

    @classmethod
    def get_type_description(cls, table_name):
        filesystem_representation = TableInFilesystem(table_name)
        table_type = filesystem_representation.get_type()
        return cls.TABLE_TYPE_DESCRIPTIONS.get(table_type)
