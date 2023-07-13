from bs4 import BeautifulSoup

try:
    from tqdm.notebook import tqdm
except ImportError:
    tqdm = iter

from ..autodoc import Confluence, Documentation, TableInDocumentation


def autodoc_paragraph(*paragraphs, after: str, tables: list = None):
    documentation = Documentation()
    tables_not_updated = []
    if tables is None:
        tables = documentation.all_tables()

    for table_name in tqdm(tables):
        print(f'{table_name} starting')

        update = True
        table_doc = TableInDocumentation(table_name)
        content = table_doc.read()
        tag_to_insert_after = table_doc._select_one_element(content, after)

        for paragraph in reversed(paragraphs):
            if isinstance(paragraph, str):
                new_tag = Confluence.str_to_soup(paragraph)
            elif isinstance(paragraph, BeautifulSoup):
                new_tag = paragraph
            else:
                raise TypeError(
                    'paragraphs should be either strings or beautiful soup objects, ' f'{type(paragraph)} found'
                )

            tag = (list(new_tag.children)[0]).name
            classes = (list(new_tag.children)[0]).attrs['class']
            classes.insert(0, tag)
            element = '.'.join(classes)
            if len(content.select(element)) > 0:
                print(f'WARNING: element {element} already in {table_name}; passing')
                update = False

            tag_to_insert_after.insert_after(new_tag)

        if update:
            table_doc._write(content)
        else:
            tables_not_updated.append(table_name)

        print(f'{table_name} done')

    print(f'NOT UPDATED: {len(tables_not_updated)}')
    return tables_not_updated
