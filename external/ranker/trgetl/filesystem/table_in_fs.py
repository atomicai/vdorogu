import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .dictionary_in_fs import DictionaryInFilesystem
from .filesystem import (
    CHECKER_PATH,
    DATAFLOW_PATH,
    DDL_PATH,
    SENSOR_PATH,
    Filesystem,
    FilesystemDataflowPathFound,
    FilesystemError,
    FilesystemNotFoundError,
)

SCRIPT_CONF_FILE = "_conf.json"


class TableInFilesystem(Filesystem):
    def __init__(self, name: str):
        self.name = name

    def get_db(self) -> Tuple[str, Optional[str]]:
        ddl_path = self.ddl_path()
        if ddl_path.parent.name + "." + ddl_path.name == self.name:
            db_path, db_class_path = list(ddl_path.parents)[2:4]
        else:
            db_path, db_class_path = list(ddl_path.parents)[1:3]

        if db_class_path.parent == DDL_PATH:
            return (db_class_path.name, db_path.name)
        elif db_class_path == DDL_PATH:
            db_class_path = db_path
            return (db_class_path.name, None)
        else:
            raise FilesystemError("Wrong DDL structure")

    def get_source_db(self) -> Tuple[str, Optional[str]]:
        try:
            metadata_path: Path = self.metadata_path()
        except FilesystemNotFoundError:
            try:
                metadata_path = self.sensor_path()
            except FilesystemNotFoundError:
                return ("", None)
        parents = list(reversed(metadata_path.parents))
        if metadata_path.suffix == ".json":
            parents.append(metadata_path)
        dataflow_root_passed = 0
        db_class, db = "", None
        start_dir = 0
        for parent in parents:
            if dataflow_root_passed == start_dir + 1:
                db_class = parent.stem
            elif dataflow_root_passed == start_dir + 2:
                db = parent.stem
            if parent == DATAFLOW_PATH or parent == SENSOR_PATH or dataflow_root_passed > 0:
                if parent == DATAFLOW_PATH:
                    start_dir += 1
                dataflow_root_passed += 1
        return (db_class, db)

    def get_sensor_db(self) -> Tuple[Optional[str], Optional[str]]:
        if self.is_view():
            source_schema = self.get_db()
        else:
            source_schema = self.get_source_db()
        return source_schema

    def get_type(self) -> Optional[str]:
        if self.is_view():
            return "view"
        if self.is_sensor():
            return "sensor"
        try:
            metadata_path: Path = self.metadata_path()
        except FilesystemNotFoundError:
            return None
        dataflow_root_passed = 0
        for parent in reversed(metadata_path.parents):
            if dataflow_root_passed == 1:
                return parent.name
            if parent == DATAFLOW_PATH or dataflow_root_passed:
                dataflow_root_passed += 1
        raise FilesystemError("Wrong dataflow structure")

    def get_ddl_type(self) -> str:
        ddl_path = self.ddl_path()
        ddl_root_passed = 0
        for parent in reversed(ddl_path.parents):
            if ddl_root_passed == 2:
                return parent.name
            if parent == DDL_PATH or ddl_root_passed:
                ddl_root_passed += 1
        raise FilesystemError("Wrong ddl structure")

    def get_parameters(self) -> Dict[str, Any]:
        try:
            metadata_path: Path = self.metadata_path()
        except FilesystemNotFoundError:
            try:
                metadata_path = self.sensor_path()
            except FilesystemNotFoundError:
                return {}
        if metadata_path.suffix == ".sql":
            query = metadata_path.read_text()
            parameter_strings = re.findall(r"/\*parameter:\s*([^=]+?=[^=]+?)\s*\*/", query)
            parameters = {par.split("=")[0]: par.split("=")[1] for par in parameter_strings}
            parameters = {key: json.loads(value) for key, value in parameters.items()}
        elif metadata_path.suffix == ".json":
            parameters = json.loads(metadata_path.read_text())[self.name]
        elif metadata_path.suffix == ".py":
            parameters = json.loads((metadata_path.parent / SCRIPT_CONF_FILE).read_text())[self.name]
        else:
            raise FilesystemError(f"Wrong dataflow extension: {metadata_path.suffix}")
        return parameters

    def extract_all_dependencies(self, pass_over_views: bool = True) -> Set[str]:
        parameters = self.get_parameters()
        if "source" in parameters:
            return {parameters["source"]}

        dependencies: Set[str] = set()
        try:
            dataflow_path = self.dataflow_path()

            if dataflow_path.is_dir():
                dependencies = set()
                for path in dataflow_path.glob("*.sql"):
                    query = path.read_text()
                    dependencies = dependencies.union(self._dependencies_in_query(query))

            elif dataflow_path.suffix == ".sql":
                query = dataflow_path.read_text()
                dependencies = self._dependencies_in_query(query)
                if parameters.get("dependencies"):
                    deps = parameters["dependencies"]
                    if isinstance(deps, (list, set, tuple)):
                        dependencies = dependencies.union(deps)
                    else:
                        dependencies = dependencies.union({deps})

            elif dataflow_path.suffix == ".py":
                parameters = json.loads((dataflow_path.parent / SCRIPT_CONF_FILE).read_text())[self.name]
                dependencies = set(parameters.get("dependencies", set()))

        except FilesystemNotFoundError:
            pass

        custom_dependencies_raw: Union[str, List[str]] = parameters.get("custom_dependencies", [])
        if isinstance(custom_dependencies_raw, str):
            custom_dependencies = {custom_dependencies_raw}
        else:
            custom_dependencies = set(custom_dependencies_raw)
        dependencies = dependencies.union(custom_dependencies)

        if pass_over_views:
            all_views = self.all_views()
            for table_name in dependencies.copy():
                if table_name in all_views:
                    table = TableInFilesystem(table_name)
                    view_dependencies = table.extract_all_dependencies()
                    dependencies = dependencies.union(view_dependencies)

        return dependencies

    def extract_live_dependencies(self, pass_over_views: bool = True) -> Set[str]:
        parameters = self.get_parameters()
        dependency_free = parameters.get("dependency_free", False)
        if dependency_free:
            return set()
        dependencies = self.extract_all_dependencies(pass_over_views=pass_over_views)
        all_live_tables = set(self.all_live_tables())
        live_dependencies = dependencies.intersection(all_live_tables)
        if self.name in live_dependencies:
            live_dependencies.remove(self.name)
        if not self.is_sensor():
            all_sensors = self.all_sensors()
            live_dependencies = live_dependencies.union({sensor for sensor in dependencies.intersection(all_sensors)})
        return live_dependencies

    def ddl_path(self) -> Path:
        return self._find_query_path(self.name, DDL_PATH, exception_dirs=["index"])

    def metadata_path(self) -> Path:
        if self.is_view():
            ddl_path = self.ddl_path()
            if ddl_path.is_dir():
                ddl_path = sorted(ddl_path.glob("*.sql"))[-1]
            return ddl_path
        else:
            return self._find_query_path(self.name, DATAFLOW_PATH)

    def dataflow_path(self) -> Path:
        metadata_path = self.metadata_path()
        parameters = self.get_parameters()
        if "query" not in parameters:
            return metadata_path
        else:
            dataflow_path = metadata_path.parent / parameters["query"]
            return dataflow_path

    def checker_path(self) -> Path:
        return self._find_query_path(self.name, CHECKER_PATH, silent=True)

    def sensor_path(self) -> Path:
        if self.dataflow_path_exists():
            raise FilesystemDataflowPathFound(f"Table already exists {self.dataflow_path()}")
        return self._find_query_path(self.name, SENSOR_PATH)

    def get_checkers(self) -> List[str]:
        checker_path = self.checker_path()
        if checker_path is None:
            return []
        checkers = json.loads(checker_path.read_text())[self.name]
        return checkers

    def ddl_path_exists(self) -> bool:
        try:
            self.ddl_path()
            return True
        except FilesystemNotFoundError:
            return False

    def dataflow_path_exists(self) -> bool:
        try:
            self.dataflow_path()
            return True
        except FilesystemNotFoundError:
            return False

    def sensor_path_exists(self) -> bool:
        if self.dataflow_path_exists():
            return False
        try:
            return self.sensor_path() is not None
        except FilesystemNotFoundError:
            return False

    def get_dictionaries(self) -> List[DictionaryInFilesystem]:
        dictionaries = []
        db_class, db = self.get_db()
        if db_class == "ch":
            for dictionary_name in self.all_dictionaries():
                dictionary = DictionaryInFilesystem(dictionary_name)
                if dictionary.table_name() == self.name:
                    dictionaries.append(dictionary)
        return dictionaries

    def is_dictionary(self) -> bool:
        db_class, db = self.get_db()
        if db_class == "ch":
            for dictionary_name in self.all_dictionaries():
                dictionary = DictionaryInFilesystem(dictionary_name)
                if dictionary.table_name() == self.name:
                    return True
        return False

    def is_view(self) -> bool:
        try:
            ddl_type = self.get_ddl_type()
            return ddl_type == "view"
        except FilesystemNotFoundError:
            return False

    def is_sensor(self) -> bool:
        if self.dataflow_path_exists():
            return False
        try:
            sensor_path = self.sensor_path()
            if sensor_path is None:
                return False
            if SENSOR_PATH in sensor_path.parents:
                return True
            raise FilesystemError("Wrong sensor structure")
        except FilesystemNotFoundError:
            return False

    def get_standart_parameters(self) -> Dict[str, Any]:
        raw_parameters = self.get_parameters()
        parameters = raw_parameters.copy()

        parameters["schedule"] = raw_parameters.get("schedule", "day")
        parameters["date_column"] = raw_parameters.get("date_column", "date")
        parameters["source_date_column"] = raw_parameters.get("source_date_column") or parameters["date_column"]
        for field in ("date_column", "source_date_column"):
            if parameters[field] != "date" and not parameters[field].startswith("cast("):
                parameters[field] = f"cast({parameters[field]} as date)"

        parameters["query_parameters"] = raw_parameters.get("query_parameters", {})
        parameters["allow_zero"] = raw_parameters.get("allow_zero", False)
        parameters["datelag"] = raw_parameters.get("datelag", 0)
        parameters["as_file"] = raw_parameters.get("as_file", False)

        table_type = self.get_type()
        if table_type in ["transform", "dump"]:
            parameters["is_full"] = False
        elif table_type in ["transform-full", "dump-full", "file"]:
            parameters["is_full"] = True
        else:
            parameters["is_full"] = raw_parameters.get("is_full", False)

        return parameters

    def _dependencies_in_query(self, query: str) -> Set[str]:
        dependencies = self._tables_in_query(query)
        if self.get_source_db() == ("ch", "olap"):
            dictionaries = self._dicts_in_query(query)
            dict_tables = {DictionaryInFilesystem(dictionary).table_name() for dictionary in dictionaries}
            dependencies = dependencies.union(dict_tables)
        return dependencies

    @staticmethod
    def _tables_in_query(sql_str: str) -> Set[str]:
        # remove the /* */ comments
        q = re.sub(r"/\*[^*]*\*+(?:[^*/][^*]*\*+)*/", "", sql_str)

        # remove whole line -- and # comments
        lines = [line for line in q.splitlines() if not re.match(r"^\s*(--|#)", line)]

        # remove trailing -- and # comments
        q = " ".join([re.split(r"--|#", line)[0] for line in lines])

        # split on blanks, parens and semicolons
        tokens = re.split(r"[\s)(;]+", q)

        # scan the tokens. if we see a FROM or JOIN, we set the get_next
        # flag, and grab the next one (unless it's SELECT).

        result = set()
        get_next = False
        for tok in tokens:
            if get_next:
                if tok.lower() not in ["", "select", "with"] and "." in tok:
                    result.add(tok)
                get_next = False
            get_next = tok.lower() in ["from", "join"]

        return result

    @staticmethod
    def _dicts_in_query(sql_str: str) -> Set[str]:
        q = re.sub(r"/\*[^*]*\*+(?:[^*/][^*]*\*+)*/", "", sql_str)
        lines = [line for line in q.splitlines() if not re.match(r"^\s*(--|#)", line)]
        q = " ".join([re.split(r"--|#", line)[0] for line in lines])

        tokens = re.split(r"[\s)(;',]+", q)

        result = set()
        get_next = False
        for tok in tokens:
            if get_next:
                result.add(tok)
                get_next = False
            get_next = bool(re.match("dictget[a-z0-9]*", tok.lower()))

        return result
