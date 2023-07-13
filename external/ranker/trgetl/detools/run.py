from ..table import Table


def table_in_public(table_name, date=None):
    table = Table(table_name)
    schema_name, table_name = table_name.split('.')
    table.name = f'public.{table_name}'
    table.run(date)
