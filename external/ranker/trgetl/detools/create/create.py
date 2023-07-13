from typing import Optional, Tuple

from ...autodoc import TableInDocumentation
from ...database import Clickhouse, Postgres
from ...database.postgres import PostgresError
from ...filesystem import DATAFLOW_PATH, DDL_PATH, TableInFilesystem
from ..helpers import get_tables_to_import


def table_in_public(table_name: str, drop: bool = True) -> None:
    filesystem_representation = TableInFilesystem(table_name)
    table_name = table_name.split('.')[1]
    ddl_path = filesystem_representation.ddl_path()
    olap = Clickhouse()
    try:
        for path in sorted(list(ddl_path.glob('*.sql'))):
            ddls = path.read_text()
            for ddl in ddls.split(';\n'):
                if ddl.strip() != '':
                    assert '${dbname}.${tablename}' in ddl, 'Not found: ${dbname}.${tablename}'
                    ddl = ddl.replace('Replicated', '')
                    ddl = ddl.replace('${tablename}', table_name)
                    ddl = ddl.replace('${dbname}', 'public')
                    olap.execute(ddl)
            print(f'Success: {path.name}')
        olap.show_create_table(f'public.{table_name}')
    finally:
        if drop:
            olap.execute(f'drop table if exists public.{table_name}')
            print('Deleted')


def ddl(
    table_name: str,
    count: int = 1,
    content: Optional[str] = None,
) -> None:
    filesystem_representation = TableInFilesystem(table_name)
    ddl_path = filesystem_representation.ddl_path()
    ddls = [int(file.stem) for file in ddl_path.iterdir()]
    first_new_ddl = max(ddls) + 1
    for new_ddl_num in range(first_new_ddl, first_new_ddl + count):
        new_ddl = str(new_ddl_num)
        new_ddl = '0' * (4 - len(new_ddl)) + new_ddl + '.sql'
        new_ddl_path = ddl_path / new_ddl
        new_ddl_path.touch()
        print(f'Created: {new_ddl_path}')
    if count == 1 and content is not None:
        new_ddl_path.write_text(content)
        print(f'Written: {new_ddl_path}')


def new_ddl(
    table_name: str,
    db: str = 'ch',
    table_type: str = 'table',
) -> None:
    path = DDL_PATH / db / table_type
    assert path.exists(), f'{path} does not exist'
    if db == 'ch':
        schema, table_name = table_name.split('.')
        path = path / schema / table_name
        path.mkdir(exist_ok=True, parents=True)
        table_name = '0001'
    path = path / (table_name + '.sql')
    assert not path.exists(), f'{path} already exists'
    path.touch()
    print(f'Created: {path}')


def new_dataflow(
    table_name: str,
    dataflow_type: str = 'transform',
    db: Tuple[str, str] = ('ch', 'events'),
) -> None:
    if isinstance(db, str):
        db = [db]
    path = DATAFLOW_PATH / dataflow_type
    for db_subpath in db:
        path = path / db_subpath
    assert path.exists(), f'{path} does not exist'
    path = path / (table_name + '.sql')
    assert not path.exists(), f'{path} already exists'
    path.touch()
    print(f'Created: {path}')


def recreate_view(table_name: str) -> None:
    filesystem_representation = TableInFilesystem(table_name)
    assert filesystem_representation.get_type() == 'view'
    ddl_path = filesystem_representation.ddl_path()
    ddls = [int(file.stem) for file in ddl_path.iterdir()]

    drop_ddl_file = str(max(ddls) + 1)
    drop_ddl_file = '0' * (4 - len(drop_ddl_file)) + drop_ddl_file + '.sql'
    drop_path = ddl_path / drop_ddl_file
    drop_text = 'DROP TABLE ${dbname}.${tablename}\n'
    drop_path.write_text(drop_text)
    print(f'Created: {drop_path}')

    recreate_ddl_file = str(max(ddls) + 2)
    recreate_ddl_file = '0' * (4 - len(recreate_ddl_file)) + recreate_ddl_file + '.sql'
    recreate_path = ddl_path / recreate_ddl_file
    recreate_text = (ddl_path / '0001.sql').read_text()
    recreate_path.write_text(recreate_text)
    print(f'Created: {recreate_path}')


def postgres_tables_from(schema: str = 'ch_olap') -> list:
    print('Start dropping foreign tables...')
    _drop_foreign_tables(db='postgres', schema=schema)
    print('Start importing foreign tables...')
    _import_foreign_tables(db='postgres', schema=schema)
    print('Start creating views on foreign tables...')
    _create_views_on_foreign_tables(db='postgres', schema=schema)
    return Postgres().search_tables(schema)


def _drop_foreign_tables(db: str = 'postgres', schema: str = 'ch_olap') -> None:
    if db == 'postgres':
        drop_queries = []
        foreign_tables = Postgres().tables()
        for table in [t for t in foreign_tables if t.startswith(schema + '.')]:
            query = f'DROP FOREIGN TABLE {table} cascade;'
            drop_queries.append(query)
            print('QUERY:', query)
            try:
                Postgres().execute(query)
            except PostgresError as e:
                raise e
    else:
        raise NotImplementedError


def _import_foreign_tables(db: str = 'postgres', schema: str = 'ch_olap') -> None:
    if db == 'postgres':
        foreign_schema = {
            'ch_olap': 'default',
            'ch_events': 'target',
        }[schema]
        foreign_server = {
            'ch_olap': 'clickhouse_trg1_fdw_svr',
            'ch_events': 'clickhouse_fwd_pg_to_trg_srv',
        }[schema]
        import_query = """IMPORT FOREIGN SCHEMA "{0}" LIMIT TO ({1}) FROM SERVER {2} INTO {3};"""
        import_query = import_query.format(foreign_schema, ', '.join(get_tables_to_import()), foreign_server, schema)
        print('QUERY:', import_query)
        try:
            Postgres().execute(import_query)
        except PostgresError as e:
            raise e
    else:
        raise NotImplementedError


def _create_views_on_foreign_tables(db: str = 'postgres', schema: str = 'ch_olap') -> None:
    if db == 'postgres':
        view_queries = []
        foreign_tables = Postgres().tables()
        for table in [t for t in foreign_tables if t.startswith(schema + '.')]:
            emart_tab = 'emart.' + table.replace('.', '_')
            if not emart_tab.endswith('_v'):
                emart_tab += '_v'
            query = f'CREATE OR REPLACE VIEW {emart_tab} AS ( SELECT * FROM {table} );'
            view_queries.append(query)
            print('QUERY:', query)
            try:
                Postgres().execute(query)
            except PostgresError as e:
                raise e
    else:
        raise NotImplementedError


def comments_from_manual(table_name: str) -> None:
    manual_comments = TableInDocumentation(table_name).manual_comments()
    alter_content = ''
    for idx, row in manual_comments.iterrows():
        colname = row['Поле']
        comment = row['Ручное описание']
        alter_content += "ALTER TABLE ${dbname}.${tablename} " f"COMMENT COLUMN {colname} '{comment}';\n"
    ddl(table_name, content=alter_content)
