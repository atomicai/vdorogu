try:
    from tqdm.notebook import tqdm
except ImportError:
    tqdm = iter

from ..database import Clickhouse
from ..filesystem import Filesystem, TableInFilesystem


def table(table_name: str) -> None:
    filesystem_representation = TableInFilesystem(table_name)
    ddl_path = filesystem_representation.ddl_path()
    ddls = [int(file.stem) for file in ddl_path.iterdir()]
    drop_ddl_file = str(max(ddls) + 1)
    drop_ddl_file = "0" * (4 - len(drop_ddl_file)) + drop_ddl_file + ".sql"
    drop_path = ddl_path / drop_ddl_file
    text = "DROP TABLE ${dbname}.${tablename} NO DELAY\n"
    drop_path.write_text(text)
    print(f"Created: {drop_path}")


def deleted_tables() -> None:
    for table_name in tqdm(set(Filesystem.all_tables())):
        filesystem_representation = TableInFilesystem(table_name)
        if filesystem_representation.get_db()[0] == "ch":
            ddl_path = filesystem_representation.ddl_path()
            ddls = [file.name for file in ddl_path.iterdir()]
            latest_ddl_file = max(ddls)
            latest_ddl = (ddl_path / latest_ddl_file).read_text()
            if "drop table ${dbname}.${tablename}" in latest_ddl.lower():
                if table_name not in Clickhouse("olap").tables():
                    for ddl in ddls:
                        (ddl_path / ddl).unlink()
                    print(f"Deleted {ddl_path}")


def recreated_tables() -> None:
    for table_name in tqdm(set(Filesystem.all_tables())):
        filesystem_representation = TableInFilesystem(table_name)
        if filesystem_representation.get_db()[0] == "ch":
            ddl_path = filesystem_representation.ddl_path()
            ddls = sorted([file.name for file in ddl_path.iterdir()])
            if len(ddls) >= 2:
                drop_ddl = (ddl_path / ddls[-2]).read_text()
                recreate_ddl = (ddl_path / ddls[-1]).read_text()
                if "drop table ${dbname}.${tablename}" in drop_ddl.lower() and (
                    "create view ${dbname}.${tablename}" in recreate_ddl.lower()
                    or "create table ${dbname}.${tablename}" in recreate_ddl.lower()
                ):
                    for ddl in ddls:
                        (ddl_path / ddl).unlink()
                        print(f"Deleted {ddl_path / ddl}")
                    new_ddl = ddl_path / "0001.sql"
                    new_ddl.write_text(recreate_ddl)
                    print(f"Created: {new_ddl}")
