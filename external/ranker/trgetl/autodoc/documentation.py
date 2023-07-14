from .confluence import Confluence


class Documentation:
    CH_AUTODOC_PAGE = 697262719
    SCHEMA_PREFIX = "Схема "

    def __init__(self, confluence_client: Confluence = None):
        if confluence_client is None:
            confluence_client = Confluence()
        self.confluence = confluence_client

    def all_tables(self, return_ids=False):
        tables = []
        schemas = self.all_schemas(return_ids=True)
        for schema_pageid, schema_name in schemas:
            new_tables = self.confluence.get_children(page_id=schema_pageid)
            if not return_ids:
                new_tables = [table_name for page_id, table_name in new_tables]
            tables += new_tables
        return set(tables)

    def all_schemas(self, return_ids=False):
        schemas = self.confluence.get_children(page_id=self.CH_AUTODOC_PAGE)
        schemas = [(page_id, schema_name.replace(self.SCHEMA_PREFIX, "")) for page_id, schema_name in schemas]
        if not return_ids:
            schemas = [schema_name for page_id, schema_name in schemas]
        return set(schemas)

    def table_page_exists(self, table_name):
        return table_name in self.all_tables()

    def get_schema_pageid(self, schema):
        if schema not in self.all_schemas():
            raise DocumentationError(f"schema {schema} not in documentation")
        schema_title = self.SCHEMA_PREFIX + schema
        return self.confluence.find(schema_title)

    def create_schema(self, schema):
        schema_title = self.SCHEMA_PREFIX + schema
        page_id = self.confluence.create(title=schema_title, parent_page_id=self.CH_AUTODOC_PAGE)
        schema_content = (
            '<p><ac:structured-macro ac:macro-id="041edf0b-5cac-4d92-9c0e-a7ef20daa9df" '
            'ac:name="pagetree" ac:schema-version="1"><ac:parameter ac:name="root">'
            '<ac:link><ri:page ri:content-title="@self"></ri:page></ac:link></ac:parameter>'
            '<ac:parameter ac:name="searchBox">true</ac:parameter></ac:structured-macro></p>'
        )
        self.confluence.write(page_id, schema_content)


class DocumentationError(Exception):
    pass
