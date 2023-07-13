import json
import sys
from pathlib import Path

LIB_PATH = Path(__file__).parent.parent
REPO_PATH = LIB_PATH.parent.parent
DWH_PATH = REPO_PATH / 'common' / 'trgetl'

DDL_PATH = DWH_PATH / 'ddl'
DATAFLOW_PATH = DWH_PATH / 'data-flow'
CHECKER_PATH = DWH_PATH / 'checker'
SENSOR_PATH = DWH_PATH / 'sensor'
DICT_PATH = DDL_PATH / 'ch' / 'dictionary'
MISC_PATH = DWH_PATH / 'misc'
REPORT_PATH = DWH_PATH / 'report'


if str(LIB_PATH.parent) not in sys.path:
    sys.path.append(str(LIB_PATH.parent))


class Filesystem:
    def __repr__(self):
        name = self.__class__.__name__
        object_name = getattr(self, 'name', None)
        if object_name:
            name += f': {object_name}'
        return name

    @classmethod
    def read_misc_query(cls, query_name):
        path = cls._find_query_path(query_name, MISC_PATH)
        return path.read_text()

    @classmethod
    def all_tables(cls, table_db=None):
        ddl_path = DDL_PATH
        if table_db is not None:
            ddl_path /= table_db
        single_query_ddls = [table.stem for table in ddl_path.glob('**/*.sql') if not table.stem.isnumeric()]
        ci_ddls = [
            table.parent.parent.name + '.' + table.parent.name
            for table in ddl_path.glob('**/*.sql')
            if table.stem.isnumeric()
        ]
        json_ddls = [
            table
            for file in ddl_path.glob('**/*.json')
            if not file.name.startswith('_')
            for table in json.loads(file.read_text())
        ]
        ddls = single_query_ddls + ci_ddls + json_ddls
        ddls = [ddl for ddl in ddls if '.' in ddl]
        return sorted(ddls)

    @classmethod
    def all_live_tables(cls, table_type=None, table_db=None):
        dataflow_path = DATAFLOW_PATH
        if table_type is not None:
            dataflow_path /= table_type
        sql_tables = {table.stem for table in dataflow_path.glob('**/*.sql')}
        py_tables = {table.stem for table in dataflow_path.glob('**/*.py')}
        json_tables = {
            table
            for file in dataflow_path.glob('**/*.json')
            if not file.name.startswith('_')
            for table in json.loads(file.read_text())
        }
        ddls = set(cls.all_tables(table_db=table_db))
        live_tables = ddls.intersection(sql_tables.union(json_tables).union(py_tables))
        return sorted(live_tables)

    @classmethod
    def all_dictionaries(cls):
        return [dictionary.stem for dictionary in DICT_PATH.glob('*.xml')]

    @classmethod
    def all_views(cls, table_db=None):
        ddl_path = DDL_PATH
        if table_db is not None:
            ddl_path /= table_db
        views = [table.stem for table in ddl_path.glob('**/view/*.sql') if not table.stem.isnumeric()] + [
            table.parent.parent.name + '.' + table.parent.name
            for table in ddl_path.glob('**/view/**/*.sql')
            if table.stem.isnumeric()
        ]
        return set(views)

    @classmethod
    def all_sensors(cls, table_type=None, table_db=None):
        sensor_path = SENSOR_PATH
        json_sensors = {sensor for file in sensor_path.glob('**/*.json') for sensor in json.loads(file.read_text())}
        if table_type is not None:
            sensor_path /= table_type
            if table_db is not None:
                sensor_path /= table_db
        sql_sensors = {query.stem for query in sensor_path.glob('**/*.sql')}
        return sql_sensors.union(json_sensors)

    @classmethod
    def all_reports(cls) -> set:
        reports = set()
        for report_path in REPORT_PATH.glob('*'):
            is_dir = report_path.is_dir()
            if not is_dir and report_path.name.endswith('.py'):
                reports.add(report_path.stem)
            elif is_dir and not report_path.name.startswith('__') and not report_path.name.startswith('.'):
                reports.add(report_path.name)
        return reports

    @classmethod
    def _find_query_path(cls, name, path, exception_dirs=tuple(), silent=False) -> Path:
        query_paths = {
            p
            for p in path.glob(f'**/{name}*')
            if p.name == name or p.stem == name and p.parent.name not in exception_dirs
        }
        query_separate_schema_paths = {
            p for p in path.glob('**/{}'.format(name.replace('.', '/'))) if p.parent.parent.name not in exception_dirs
        }
        aggregator_paths = {
            p for p in path.glob('**/*.json') if not p.name.startswith('_') and name in json.loads(p.read_text())
        }
        paths = query_paths.union(query_separate_schema_paths).union(aggregator_paths)

        if len(paths) == 0:
            if silent:
                return None
            else:
                raise FilesystemNotFoundError(f'Query not found for table {name} in {path}')
        if len(paths) > 1:
            raise FilesystemDuplicatedError(f'Several queries found for table {name}:\n{paths}')
        return paths.pop()


class FilesystemError(Exception):
    pass


class FilesystemNotFoundError(FilesystemError):
    pass


class FilesystemDuplicatedError(FilesystemError):
    pass


class FilesystemDataflowPathFound(FilesystemError):
    pass
