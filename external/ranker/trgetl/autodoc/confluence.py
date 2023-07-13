from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from ..filesystem import FilesystemError
from .confluence_connector import ConfluenceConnector, ConfluenceOAuthConnector, ConfluencePATConnector


class Confluence:
    DEFAULT_PROXY = 'http://rbhp-proxy.i:3128'
    DEFAULT_SPACE = 'TRG'

    __object = None

    def __init__(
        self,
        consumer_key: Optional[str] = None,
        access_token: Optional[str] = None,
        rsa_privatekey: Optional[str] = None,
        pat_token: Optional[str] = None,
        proxy: Optional[str] = None,
        space_key: Optional[str] = None,
    ):
        if proxy is None:
            proxy = self.DEFAULT_PROXY
        if space_key is None:
            space_key = self.DEFAULT_SPACE

        is_oauth_client = not (consumer_key is None or access_token is None or rsa_privatekey is None)
        is_pat_client = pat_token is not None

        if not is_oauth_client and not is_pat_client:
            raise FilesystemError('Confluence tokens not found; pass them explisitly')

        if is_oauth_client:
            self.client: ConfluenceConnector = ConfluenceOAuthConnector(
                consumer_key=consumer_key,
                access_token=access_token,
                rsa_key=rsa_privatekey,
                http_proxy=proxy,
                https_proxy=proxy,
                space_key=space_key,
                alert_users=False,
            )
        elif is_pat_client:
            assert pat_token is not None
            self.client = ConfluencePATConnector(
                token=pat_token,
                http_proxy=proxy,
                https_proxy=proxy,
                space_key=space_key,
                alert_users=False,
            )

    def get_children(self, page_id):
        return self.client.get_child_pages(page_id)

    def create(self, title, parent_page_id):
        return self.client.create_page(title, parent_page_id)

    def delete(self, page_id):
        return self.client.delete_page(page_id)

    def find(self, title):
        return self.client.find_page(title)

    def write(self, page_id, content):
        content = str(content)
        return self.client.update_page(page_id, content)

    def read(self, page_id, parse=True):
        html = self.client.get_page_content(page_id)
        if not parse:
            return html
        else:
            content = self.str_to_soup(html)
            return content

    @staticmethod
    def str_to_soup(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, 'html.parser')

    @staticmethod
    def soup_to_df(table_html):
        df = pd.read_html(str(table_html))[0]
        return df

    @classmethod
    def df_to_soup(cls, df, classes=None, colwidth=None, escape=True):
        table_html = df.to_html(index=False, na_rep='', escape=escape)
        table_soup = cls.str_to_soup(table_html)

        if classes:
            if isinstance(classes, str):
                classes = [classes]
            classes = ['fixed-table'] + classes
        table_soup.find('table')['class'] = classes

        TABLE_WIDTH = 1000
        colnum = len(df.columns)
        if colwidth is None:
            colwidth = int(TABLE_WIDTH / colnum)
        if isinstance(colwidth, int):
            colwidth = [colwidth] * colnum
        assert len(colwidth) == colnum, 'length of colwidth is not equal to number of columns'
        colwidth_html = [f'<col style="width: {width}.0px;"/>' for width in colwidth]
        colwidth_html = '<colgroup>' + ''.join(colwidth_html) + '</colgroup>'
        colwidth_tag = BeautifulSoup(colwidth_html, 'html.parser')
        table_soup.find('table').insert(0, colwidth_tag)
        return table_soup
