import datetime as dt
import io
import json
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from snakebite.client import AutoConfigClient

from ..filesystem import TableInFilesystem
from .base_database import BaseDatabase


class Hdfs(BaseDatabase):
    RAW_MODE = True

    def __init__(self, db=None):
        if db is None:
            db = 'trg'
        assert db in ['trg'], f'Unknown db: {db}'
        self.db = db

    def ls(self, path, return_directories=False, recursive=False):
        command = ['hdfs', 'dfs', '-ls', '-C', str(path)]
        if return_directories:
            command.insert(-1, '-d')
        if recursive:
            command.insert(-1, '-R')
        stdout = subprocess.run(command, stdout=subprocess.PIPE).stdout
        stdout = stdout.decode().strip()
        if stdout == '':
            return []
        else:
            return stdout.split('\n')

    def delete(self, table_name: str, date: Optional[dt.date], date_column: str = '') -> int:
        parameters = self._get_parameters_from_ddl(table_name)
        path = parameters['source']
        path = path + '/dt=' + str(date)
        return self.rm(path)

    def _get_parameters_from_ddl(self, table_name: str) -> Dict[str, Any]:
        filesystem_representation = TableInFilesystem(table_name)
        ddl_path = filesystem_representation.ddl_path()
        return json.loads(ddl_path.read_text())[table_name]

    def rm(self, path):
        files = self.ls(path, return_directories=True)
        if len(files) == 0:
            print(f'Nothing to remove: {path}')
        else:
            command = ['hdfs', 'dfs', '-rm', '-r', str(path)]
            subprocess.run(command, stdout=subprocess.PIPE).stdout.decode()
            print(f'Removed: {path}')
        return len(files)

    def read(self, path, parse_dates=None) -> pd.DataFrame:
        if parse_dates is None:
            parse_dates = True

        format = path.split('.')[-1]
        if format == 'csv':
            sep = ','
        elif format == 'tsv':
            sep = '\t'
        else:
            raise AssertionError('Unknown format: {format}')

        data = self.read_string(path)
        df = pd.read_csv(io.StringIO(data), sep=sep, parse_dates=parse_dates)
        return df

    def read_string(self, path):
        assert isinstance(path, str), 'path should be string for read_string'
        client = AutoConfigClient()
        files = self._fixed_iterator(
            client.text,
            path,
        )
        assert len(files) == 1
        file = files[0]
        return file.decode()

    def _fixed_iterator(self, callable, path, **kwargs) -> List[str]:
        if not isinstance(path, list):
            path = [path]
        result = []
        try:
            for i in callable(path, **kwargs):
                result.append(i)
        except RuntimeError as e:
            if str(e) != 'generator raised StopIteration':
                raise e
        return result

    def insert_file(
        self,
        table_name: str,
        data: Union[str, bytes],
        date: Optional[Union[str, dt.date]],
        return_filenum: bool = True,
        format: str = None,
    ) -> Optional[int]:
        parameters = self._get_parameters_from_ddl(table_name)
        path = parameters['source'] + '/dt=' + str(date)
        self.mkdir(path)
        filename = parameters.get('filename') or table_name
        return self.put(path, filename, data, format)

    def mkdir(self, path: str) -> int:
        command = ['hdfs', 'dfs', '-mkdir', path]
        returncode = subprocess.run(command).returncode
        if returncode:
            print(f'Failed to make a directory {path}')
        else:
            print(f'Success to make a directory {path}')
        return returncode

    def put(
        self,
        path: str,
        filename: str,
        data: Union[str, bytes],
        format: str = None,
    ) -> int:
        if isinstance(data, str):
            data = data.encode('utf-8')
        tmp_file = NamedTemporaryFile()
        tmp_file.file.write(data)
        tmp_file.file.seek(0)
        format = format or 'csv'
        fullpath = path + '/' + filename + '.' + format.lower()
        command = ['hdfs', 'dfs', '-put', tmp_file.name, fullpath]
        returncode = subprocess.run(command).returncode
        if returncode:
            raise Exception(f'Something gone wrong while putting file to {fullpath}')
        else:
            self._touch_success_file(path)
            print(f'Successfully put file to {path}')
        return len(self.ls(path, return_directories=True))

    def _touch_success_file(self, path: str) -> None:
        command = ['hdfs', 'dfs', '-touchz', path + '/_SUCCESS']
        returncode = subprocess.run(command).returncode
        if returncode:
            raise Exception(f'Something gone wrong while putting file to {path}')
