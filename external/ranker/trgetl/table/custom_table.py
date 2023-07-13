import datetime as dt
from typing import Dict

from .ab import AbReckoner
from .transform_table import TransformTable


class CustomTable(TransformTable):
    INTERNAL_CALLABLES = ['recursive', 'ab_results']
    EXTERNAL_CALLABLES = ['multiquery']

    def _get_query(self):
        dataflow_path = self.filesystem_representation.dataflow_path()
        assert dataflow_path.exists(), f'Dataflow path does not exist for table {self.name}'
        if dataflow_path.is_dir():
            return {path.stem: path.read_text() for path in dataflow_path.glob('*.sql')}
        else:
            return {'default': dataflow_path.read_text()}

    def _prepare_query(self, queries, date):
        return {key: super(CustomTable, self)._prepare_query(query_, date) for key, query_ in queries.items()}

    def _external_upload(self, date, queries):
        custom_upload_name = self.parameters['callable']
        assert custom_upload_name in self.EXTERNAL_CALLABLES, f'unknown external callable {custom_upload_name}'
        return self._upload(date, queries, custom_upload_name)

    def _internal_upload(self, date, queries):
        custom_upload_name = self.parameters['callable']
        assert custom_upload_name in self.INTERNAL_CALLABLES, f'unknown internal callable {custom_upload_name}'
        return self._upload(date, queries, custom_upload_name)

    def _upload(self, date, queries, custom_upload_name):
        custom_upload_callable = getattr(self, 'custom_upload__' + custom_upload_name)
        self._clear(date)
        rownum = custom_upload_callable(queries, date)
        return rownum

    def custom_upload__recursive(self, queries, date):
        rownum = 0
        for step in range(100):
            if step == 0:
                current_step_query = queries['base']
            else:
                current_step_query = queries['step'].format(step=step)
            current_step_query = f'insert into {self.name}' + current_step_query

            current_step_rownum = self.source_db.execute(current_step_query, return_rownum=True)
            if current_step_rownum == 0:
                break

            rownum += current_step_rownum
            print(f'Step {step}, total of {rownum} rows inserted')

        return rownum

    def custom_upload__multiquery(
        self,
        queries: Dict[str, str],
        date: dt.date,
    ):
        rownum = 0
        for subquery_name, subquery in queries.items():
            data = self._read(subquery)
            rownum += self._insert(data)
            print(f'Table {self.name}: subquery {subquery_name}: ' f'total of {rownum} rows inserted')
        return rownum

    def custom_upload__ab_results(
        self,
        queries: Dict[str, str],
        date: dt.date,
    ) -> int:
        ab_reckoner = AbReckoner(self.name, self.db)
        return ab_reckoner.calculate_and_upload(queries, date)
