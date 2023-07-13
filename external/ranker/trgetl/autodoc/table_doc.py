import os
import re

import pandas as pd

from ..database import Clickhouse, Database
from ..filesystem import REPO_PATH, FilesystemNotFoundError, TableInFilesystem
from ..table import Table
from .confluence import Confluence
from .documentation import Documentation, DocumentationError

SQL_KEY_WORDS = [
    'add',
    'all',
    'alter',
    'and',
    'as',
    'asc',
    'asof',
    'between',
    'by',
    'case',
    'cast',
    'column',
    'comment',
    'create',
    'delete',
    'desc',
    'distinct',
    'drop',
    'else',
    'end',
    'engine',
    'from',
    'full',
    'global',
    'group',
    'if',
    'in',
    'inner',
    'is',
    'join',
    'key',
    'left',
    'like',
    'limit',
    'modify',
    'multiif',
    'not',
    'null',
    'on',
    'or',
    'order',
    'outer',
    'over',
    'partition',
    'primary',
    'rename',
    'right',
    'row',
    'sample',
    'select',
    'settings',
    'table',
    'then',
    'ttl',
    'union',
    'update',
    'using',
    'values',
    'view',
    'where',
    'when',
    'with',
]
SQL_DTYPES = [
    'String',
    'Date',
    'DateTime',
    'Int[0-9]+',
    'UInt[0-9]+',
    'Float[0-9]+',
]


class TableInDocumentation:
    SCHEMA = {
        'prefix': None,
        'manualdoc': 'Ручное описание',
        'autodoc': 'Автодокументация',
    }
    AUTODOC_SCHEMA = {'structure': 'Структура данных'}

    def __init__(self, name, confluence_client=None):
        self.name = name

        if confluence_client is None:
            confluence_client = Confluence()
        self.confluence = confluence_client
        self.db = Clickhouse('olap')

        self.page_id = self.confluence.find(self.name)
        if self.page_id is not None:
            self._assert_has_correct_structure()

    def link(self):
        return self._get_confluence_link(page_id=self.page_id)

    def create(self):
        if self.page_id is not None:
            print(f'{self.name} already exists at {self.page_id}, passing')
        else:
            schema = self.name.split('.')[0]
            documentation = Documentation(confluence_client=self.confluence)
            schema_pageid = documentation.get_schema_pageid(schema)
            self.page_id = self.confluence.create(title=self.name, parent_page_id=schema_pageid)
            self._write(self._empty_page())
            print(f'{self.name} created')

    def run(self):
        self._assert_exists()
        content = self.read()
        content = self._update_structure_table(content)
        content = self._update_metadata(content)
        content = self._update_dataflow(content)
        content = self._update_ddl(content)
        content = self._update_users(content)
        self._write(content)
        print(f'{self.name} updated')

    def read(self, parse=True):
        self._assert_exists()
        return self.confluence.read(self.page_id, parse=parse)

    def delete(self, force=False):
        self._assert_exists()
        if force or not self._has_manual_info():
            self.confluence.delete(self.page_id)
            print(f'page {self.name} ({self.page_id}) deleted ')
        else:
            print(f'page {self.name} contains manual info, marking to delete')
            self._mark_to_delete()

    def manual_comments(self, content=None):
        if content is None:
            content = self.read()
        content.select('table.structure')[0]
        table = content.select('table.structure')[0]
        df = self.confluence.soup_to_df(table)
        manual_comments = df[['Поле', 'Ручное описание']].dropna()
        return manual_comments

    def _assert_exists(self):
        if self.page_id is None:
            raise DocumentationError(f'Table not found in counfluence: {self.name}')

    def _assert_has_correct_structure(self):
        content = self.read()
        autodoc_mark = content.select('p.mark.autodoc')
        if not autodoc_mark:
            raise DocumentationError(f'doc content does not contain autodoc_mark ({self.name}, {self.page_id})')

    def _write(self, content):
        self.confluence.write(self.page_id, content)

    def _has_manual_info(self):
        content = self.read()

        manual_paragraphs = content.select('p.description')
        has_manual_info = any(paragraph.string is not None for paragraph in manual_paragraphs)

        if not has_manual_info:
            structure_table = self._select_one_element(content, 'table.structure')
            structure_table = self.confluence.soup_to_df(structure_table)
            has_manual_info = structure_table['Ручное описание'].notna().any()

        return has_manual_info

    def _mark_to_delete(self):
        content = self.read()
        delete_mark = content.select('p.mark.delete')
        if delete_mark:
            print(f'page {self.name} already contains delete mark, passing\n{self.link()}')
        else:
            delete_mark = '<p class="mark delete" style="color:red"><strong>MARKED TO DELETE</strong></p>'
            delete_mark = self.confluence.str_to_soup(delete_mark)
            content.insert(0, delete_mark)
            self._write(content)
            print(f'page {self.name} is marked to delete\n{self.link()}')

    @staticmethod
    def _empty_page():
        autodoc_mark = '<p class="mark autodoc">[autodoc]</p>'
        table_of_contents = (
            '<p class="table_of_contents">'
            '<ac:structured-macro ac:macro-id="26a40cf7-e109-4b24-9b0b-5bf0aa3a0e76"'
            ' ac:name="toc" ac:schema-version="1">'
            '</ac:structured-macro>'
            '</p>'
        )
        description_header = '<h1 class="description">Ручное описание</h1>'
        description_body = '<p class="description"><br/></p>'
        structure_header = '<h1 class="structure">Структура данных</h1>'
        structure_table = pd.DataFrame(
            {
                'Поле': [''],
                'Тип': [''],
                'Описание': [''],
                'Ручное описание': [''],
            }
        )
        structure_table = Confluence.df_to_soup(structure_table, classes='structure')
        meta_header = '<h1 class="meta">Метаданные</h1>'
        meta_db = '<p class="meta db">Таблица находится в базе:</p>'
        meta_source_db = '<p class="meta source_db">Таблица загружается из базы:</p>'
        meta_dependencies = '<p class="meta dependencies">Таблица зависит от:</p>'
        meta_dependencies_list = '<ul class="meta dependencies"><li> </li></ul>'
        meta_dataflow_type = '<p class="meta dataflow_type">Описание загрузки:</p>'
        meta_runtime = '<p class="meta runtime">Среднее время загрузки за последнюю неделю:</p>'
        meta_parameters = '<p class="meta parameters">Параметры заливатора:</p>'
        dataflow_header = '<h1 class="dataflow">Запрос для заливки данных</h1>'
        dataflow_link = '<p class="dataflow link">Ссылка на gitlab</p>'
        dataflow_body = '<pre class="dataflow query">[dataflow_query]</pre>'
        ddl_header = '<h1 class="ddl">DDL</h1>'
        ddl_link = '<p class="ddl link">Ссылка на gitlab</p>'
        ddl_body = '<pre class="ddl query">[ddl_query]</pre>'
        users_header = '<h1 class="users">Пользователи данных</h1>'
        users_table = pd.DataFrame(
            {
                'Пользователь': [''],
                'Количество обращений за последние 30 дней': [''],
                'Дата последнего обращения': [''],
            }
        )
        users_table = Confluence.df_to_soup(users_table, classes='users')
        empty_paragraph = '<p></p>'
        return (
            autodoc_mark
            + table_of_contents
            + description_header
            + description_body
            + structure_header
            + str(structure_table)
            + meta_header
            + meta_db
            + meta_source_db
            + meta_dependencies
            + meta_dependencies_list
            + meta_dataflow_type
            + meta_runtime
            + meta_parameters
            + dataflow_header
            + dataflow_link
            + dataflow_body
            + ddl_header
            + ddl_link
            + ddl_body
            + users_header
            + str(users_table)
            + empty_paragraph
        )

    def _update_structure_table(self, content):
        manual_comments = self.manual_comments(content)

        new_metadata = self.db.columns(self.name, return_dtypes=True)
        new_metadata = new_metadata[['name', 'type', 'comment']]
        new_metadata.columns = ['Поле', 'Тип', 'Описание']

        new_df = new_metadata.merge(manual_comments, how='left')
        new_df['Описание'] = new_df['Описание'].apply(self._wrap_tables_in_links)
        dict_examples = self._get_dictionary_examples()
        if dict_examples is not None:
            new_df = new_df.merge(dict_examples, how='left')

        content = self._replace_table_with(content, 'table.structure', new_df, escape=False)
        return content

    def _get_dictionary_examples(self):
        dictionaries = TableInFilesystem(self.name).get_dictionaries()
        if dictionaries == []:
            return None

        dict_examples: pd.DataFrame = pd.concat(
            [
                pd.DataFrame(dictionary.usage_examples().items(), columns=['Поле', 'Обращение к словарю'])
                for dictionary in dictionaries
            ]
        )
        dict_examples = dict_examples.drop_duplicates('Поле')
        return dict_examples

    def _update_metadata(self, content):
        content = self._update_metadata_db(content)
        content = self._update_metadata_source_db(content)
        content = self._update_metadata_dependencies(content)
        content = self._update_metadata_dataflow_type(content)
        content = self._update_metadata_runtime(content)
        content = self._update_metadata_parameters(content)
        return content

    def _update_metadata_db(self, content):
        db = TableInFilesystem(self.name).get_db()
        db = Database(*db).hunam_readable_title()
        db = f'Таблица находится в базе: {db}'
        content = self._replace_element_with(content, 'p.meta.db', db)
        return content

    def _update_metadata_source_db(self, content):
        source_db = TableInFilesystem(self.name).get_source_db()
        if source_db == (None, None):
            source_db = '(нет)'
        else:
            source_db = Database(*source_db).hunam_readable_title()
        source_db = f'Таблица загружается из базы: {source_db}'
        content = self._replace_element_with(content, 'p.meta.source_db', source_db)
        return content

    def _update_metadata_dependencies(self, content):
        dependencies = TableInFilesystem(self.name).extract_all_dependencies()
        dependencies = sorted(dependencies)
        dependencies_html = ''
        for dependency in dependencies:
            link = self._get_confluence_link(dependency)
            if link is not None:
                dependency = f'<a href="{link}">{dependency}</a>'
            dependencies_html += f'<li>{dependency}</li>'
        if dependencies_html == '':
            dependencies_html = '<li> </li>'
        content = self._replace_element_with(content, 'ul.meta.dependencies', dependencies_html)
        return content

    def _update_metadata_dataflow_type(self, content):
        dataflow_description = Table.get_type_description(self.name)
        dataflow_description = f'Описание загрузки: {dataflow_description}'
        content = self._replace_element_with(content, 'p.meta.dataflow_type', dataflow_description)
        return content

    def _update_metadata_runtime(self, content):
        runtime = self.db.read(
            f'''
            select round(avg(execution_time_sec)) as runtime
            from dwh.process_log
            where date >= today() - 7
            and target = '{self.name}'
            and error = ''
        '''
        ).iloc[0, 0]
        runtime_text = f'Среднее время загрузки за последнюю неделю: {runtime} сек'
        content = self._replace_element_with(content, 'p.meta.runtime', runtime_text)
        return content

    def _update_metadata_parameters(self, content):
        parameters = TableInFilesystem(self.name).get_parameters()
        parameters_text = f'Параметры заливатора: {parameters}'
        content = self._replace_element_with(content, 'p.meta.parameters', parameters_text)
        return content

    def _update_dataflow(self, content):
        dataflow_link = 'Ссылка на gitlab'
        dataflow_query = 'Отсутствует'

        try:
            table_type = TableInFilesystem(self.name).get_type()
            if table_type != 'view':
                dataflow_path = TableInFilesystem(self.name).metadata_path()
                dataflow_link = self._get_gitlab_link(dataflow_path)
                dataflow_link = f'<a href="{dataflow_link}">Ссылка на gitlab</a>'
                dataflow_query = Table(self.name).query
                if isinstance(dataflow_query, dict):
                    dataflow_query = ';\n\n\n'.join(dataflow_query.values())
                dataflow_query = self._replace_special_symbols(dataflow_query)
                dataflow_query = self._highlight_syntax(dataflow_query)
                dataflow_query = self._add_dependencies_links(dataflow_query)
        except FilesystemNotFoundError:
            pass

        content = self._replace_element_with(content, 'p.dataflow.link', dataflow_link)
        content = self._replace_element_with(content, 'pre.dataflow.query', dataflow_query)
        return content

    def _update_ddl(self, content):
        ddl_path = TableInFilesystem(self.name).ddl_path()
        ddl_link = self._get_gitlab_link(ddl_path)
        ddl_link = f'<a href="{ddl_link}">Ссылка на gitlab</a>'
        content = self._replace_element_with(content, 'p.ddl.link', ddl_link)

        ddl_query = self.db.show_create_table(self.name, return_result=True)
        ddl_query = self._replace_special_symbols(ddl_query)
        ddl_query = self._highlight_syntax(ddl_query)
        if TableInFilesystem(self.name).is_view():
            ddl_query = self._add_dependencies_links(ddl_query)
        content = self._replace_element_with(content, 'pre.ddl.query', ddl_query)
        return content

    def _update_users(self, content):
        table_requests = self.db.read(
            f'''
            select
                user as "Пользователь",
                count() as "Количество обращений за последние 30 дней",
                max(event_date) as "Дата последнего обращения"
            from system.query_log
            where event_date >= today() - 30
                and type = 'QueryStart'
                and query_kind = 'Select'
                and has(tables, '{self.name}')
                and user not like 'jenkins%'
                and user not like 'airflow%'
            group by user
            order by count() desc
            limit 100
        '''
        )
        content = self._replace_table_with(content, 'table.users', table_requests)
        return content

    def _replace_element_with(self, content, element: str, new_content: str):
        tag = element.split('.')[0]
        classes = element.split('.')[1:]
        classes = ' '.join(classes)

        element_in_content = self._select_one_element(content, element)

        new_content = f'<{tag} class="{classes}">' + str(new_content) + f'</{tag}>'
        new_content = self.confluence.str_to_soup(new_content)
        element_in_content.replace_with(new_content)
        return content

    def _replace_table_with(self, content, element, df, escape=True):
        assert element.startswith('table.'), f'element {element} is not a table'
        classes = element.split('.')[1:]

        element_in_content = self._select_one_element(content, element)

        new_table = self.confluence.df_to_soup(df, classes=classes, escape=escape)
        element_in_content.replace_with(new_table)
        return content

    def _select_one_element(self, content, element: str):
        elements_in_content = content.select(element)
        if len(elements_in_content) != 1:
            print(content)
            raise DocumentationError(
                'There should be exactly one element '
                f'of type "{element}" '
                f'in page {self.name}, '
                f'found {len(elements_in_content)}\n'
            )
        return elements_in_content[0]

    def _get_gitlab_link(self, path):
        relative_path = os.path.relpath(path, start=REPO_PATH)
        link = 'https://gitlab.corp.mail.ru/dwh/target-etl/-/tree/master/' + relative_path
        return link

    def _get_confluence_link(self, table_name=None, page_id=None):
        assert table_name is not None or page_id is not None, 'either table_name or page_id should be passed'
        if table_name is not None:
            try:
                doc = TableInDocumentation(table_name, confluence_client=self.confluence)
                page_id = doc.page_id
            except DocumentationError:
                page_id = None
            if page_id is None:
                return None
        link = f'https://confluence.vk.team/pages/viewpage.action?pageId={page_id}'
        return link

    def _replace_special_symbols(self, query):
        query = query.replace('>', '&gt;')
        query = query.replace('<', '&lt;')
        query = query.replace('=', '&#61;')
        return query

    def _add_dependencies_links(self, query):
        dependencies = TableInFilesystem(self.name).extract_all_dependencies()
        for dependency in dependencies:
            if dependency in query:
                query = self._replace_table_name_with_link(query, dependency)
        return query

    def _wrap_tables_in_links(self, text):
        if isinstance(text, str):
            tables_in_text = re.findall(r'(?:^|\W)(\w+\.\w+)(?:$|\W)', text)
            for table in tables_in_text:
                text = self._replace_table_name_with_link(text, table)
        return text

    def _replace_table_name_with_link(self, text, table_name):
        link = self._get_confluence_link(table_name)
        if link is not None:
            link = f'<a href="{link}">{table_name}</a>'
            table_name = r'\b' + table_name.replace('.', r'\.') + r'\b'
            text = re.sub(table_name, link, text)
        return text

    def _highlight_syntax(self, query):
        key_word_pattern = (
            r'\b(' + '|'.join(SQL_KEY_WORDS + SQL_DTYPES + [r'(?<!#)[0-9]+(\.0-9]+)?'] + [r'\b\w+(?=\()']) + r')\b'
        )
        query = re.sub(key_word_pattern, r'<b>\g<1></b>', query, flags=re.IGNORECASE)
        query = re.sub(
            r'/\*.+?\*/',
            r'<i style="color:silver">\g<0></i>',
            query,
        )
        query = re.sub(
            r"'.+?'",
            r'<font color="#8FBC8F">\g<0></font>',
            query,
        )
        query = re.sub(
            r'\(|\)|\*|!|&gt;|&lt;|&#61;',
            r'<b>\g<0></b>',
            query,
        )
        return query
