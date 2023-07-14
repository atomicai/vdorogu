import logging
import time
from datetime import date
from typing import Optional
from urllib.parse import urlencode

import dateutil.parser
import dateutil.tz
import requests

logger = logging.getLogger(__name__)


class ConfluenceUnableToSaveChangesError(Exception):
    pass


class ConfluenceConnector(object):
    """Confluence API.

    This class defines API calls, including page creation, removal, and search.
    """

    SLEEP_TIME = 0.5
    RETRY_COUNT = 5
    CONTENT_TYPES = {"editor", "view", "export_view", "styled_view", "storage", "anonymous_export_view"}
    __BASE_URL = "https://confluence.vk.team/rest/api/content"
    __URL_BY_PAGE_ID = "https://confluence.vk.team/pages/viewpage.action?pageId={}"
    __LIMIT = "500"
    TIMEZONE = dateutil.tz.tzlocal()

    def __init__(self, login, password, http_proxy=None, https_proxy=None, space_key=None, alert_users=True):
        """Initialize a :class:`ConfluenceConnector`.

        :param str login: login name
        :param str password: password
        :param str http_proxy: proxy parameters for HTTP
        :param str https_proxy: proxy parameters for HTTPS
        :param str space_key: the key of the Confluence space that the API operations work with
        """
        self._auth = (login, password)
        self._headers = None
        self.space_key = space_key
        self.proxies = (
            {
                "http": http_proxy,
                "https": https_proxy,
            }
            if http_proxy and https_proxy
            else None
        )
        self.alert_users = alert_users

    def current_date(self):
        """Get the current date."""
        return date.today()

    def __make_request(self, method, path=None, params=None, data=None):
        """Send a request through the API and get a response.

        :param str method: one of 'put', 'get', 'post', or 'delete'
        :param str path: a suffix of the URI (without request parameters)
        :param params: request parameters
        :type: dict(str,str)
        :param data: a payload for POST and PUT methods
        :type data: dict(any)
        :return: response in the JSON format
        :rtype: :class:`request.Response`
        """

        # формируем путь как BASE_URL+PATH+PARAMS
        url = self.__BASE_URL
        if path:
            url += path
        if params:
            url += "?{}".format(urlencode(params))
        logger.info("request get to {}".format(url))

        # получаем ответ на запрос или валимся по exception
        try:
            response = requests.request(
                method, url, auth=self._auth, headers=self._headers, proxies=self.proxies, json=data
            )
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("Can not connect to {}".format(url))

        if response.status_code == 501:
            # noinspection PyBroadException
            try:
                msg = response.json()["message"]
            except Exception:
                msg = ""

            if msg and msg.endswith("Refreshing the page should fix this."):
                raise ConfluenceUnableToSaveChangesError(msg)

        if not response.ok:
            raise ValueError("Can not connect to {}, {}".format(url, response.text))

        return response

    def __make_get(self, path=None, params=None):
        """Prepare a GET request.

        :param str path: the URI suffix (without query parameters)
        :param params: query parameters
        :type: dict(str,str)
        :return: response in the JSON format
        :rtype: dict(Any)
        """
        return self.__make_request("get", path=path, params=params).json()

    def __make_post(self, data):
        """Prepare a POST request.

        :param data: a payload in the JSON format
        :type: dict(Any)
        :return: response in the JSON format
        :rtype: dict(Any)
        """
        return self.__make_request("post", data=data).json()

    def __make_put(self, path, data):
        """Prepare a PUT request.

        :param str path: the URI suffix (without query parameters)
        :param data: a payload in the JSON format
        :type: dict(Any)
        :return: response in the JSON format
        :rtype: dict(Any)
        """
        return self.__make_request("put", path=path, data=data).json()

    def __make_delete(self, path):
        """Prepare a DELETE request.

        :param str path: the URI suffix (without query parameters)
        :return: :class:`request.Response`
        """
        return self.__make_request("delete", path=path)

    def __get_page(self, page_id, return_content=True, content_type="view", return_response=False):
        """Prepare a request for getting a page.

        :param str page_id: a page ID
        :param bool return_content: True if the method must return the content (by default is True)
        :param str content_type: content type (by default is 'view')
        :param bool return_response: True if the method must return the response (by default is False)
        :return: version, title, content (if required), and response (if required)
        :rtype: tuple(str)
        """
        expand = "version,body.{}".format(content_type) if return_content else "version"
        params = {
            "spaceKey": self.space_key,
            "status": "current",
            "expand": expand,
        }

        response = self.__make_get(path="/{}".format(page_id), params=params)
        version = response["version"]["number"]
        title = response["title"]
        out = [version, title]
        if return_content:
            out.append(response["body"][content_type]["value"])
        if return_response:
            out.append(response)
        return tuple(out)

    def get_page_content(self, page_id, content_type="storage"):
        """Get a page content.

        :param str page_id: a page ID whose content the method must return
        :param str content_type: a content type according to :const:`CONTENT_TYPES` (by default is 'storage')
        :return: the page content
        """
        assert content_type in self.CONTENT_TYPES
        logger.info("get content page %s. Type: %s", page_id, content_type)
        return self.__get_page(page_id, return_content=True, content_type=content_type)[2]

    def update_page(self, page_id, new_content, check_date=False):
        """Update a page content.

        :param str page_id: the page ID
        :param new_content: new content
        :param bool check_date: True if the method must update only those data that were not updated
            during the current day.
        :return: None
        """
        logger.info("update page {}".format(page_id))
        version, title, response = self.__get_page(page_id, return_content=False, return_response=True)
        if check_date:
            str_dt = response["version"]["when"]
            dt = dateutil.parser.isoparse(str_dt).astimezone(self.TIMEZONE).date()
            if dt == self.current_date() and version > 1:
                logger.info("The page has already been updated today. Last update at %s", str_dt)
                return

        logger.info("page {} with version {}".format(page_id, version))
        page = {
            "id": page_id,
            "type": "page",
            "title": title,
            "space": {"key": self.space_key},
            "version": {"number": version + 1, "minorEdit": not self.alert_users},
            "body": {"storage": {"value": new_content, "representation": "storage"}},
        }
        for i in range(1, self.RETRY_COUNT):
            try:
                self.__make_put("/" + str(page_id), page)
                break
            except ConfluenceUnableToSaveChangesError:
                logger.info("Retry update (%s)", i)
                time.sleep(self.SLEEP_TIME)
                self.get_html(page_id)
                time.sleep(self.SLEEP_TIME)
        else:
            self.__make_put("/" + str(page_id), page)

    def get_html(self, page_id):
        """Get a page in the HTML format.

        :param str page_id: a page ID
        :return: str: the page content in the HTML format
        """
        url = self.__URL_BY_PAGE_ID.format(page_id)
        try:
            response = requests.get(url, auth=self._auth, proxies=self.proxies)
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("Can not connect to {}".format(url))

        if not response.ok:
            raise ValueError("Can not connect to {}, {}".format(url, response.text))

        return response.text

    def create_page(self, title, parent_page_id=None):
        """Create a page.

        :param str title: a page name
        :param str parent_page_id: the parent page ID
        :return: the created page ID
        :rtype: str
        """
        logger.info(
            "Create page {} {}".format(title, "as a child of {}".format(parent_page_id) if parent_page_id else "")
        )

        page = {
            "type": "page",
            "title": title,
            "space": {"key": self.space_key},
            "body": {"storage": {"value": "<p>auto generated page</p>", "representation": "storage"}},
        }

        if parent_page_id is not None:
            page["ancestors"] = [{"id": parent_page_id}]

        return self.__make_post(page)["id"]

    def find_page(self, title):
        """Find the ID of a page by its title.

        :param str title: a page title
        :return: the ID of the first found page
        :rtype: str
        """
        data = self.__make_get(params={"title": title, "spaceKey": self.space_key})

        if data["size"] == 0:
            return

        if data["size"] > 1:
            raise ValueError("Found {} pages for title {}".format(data["size"], title))

        return data["results"][0]["id"]

    def get_child_pages(self, page_id):
        """Get IDs of all pages by a given parent ID.

        :param str page_id: a parent page ID
        :return: a list of page and page ID pairs
        :rtype: list(tuple(str,str))
        """
        params = {"limit": self.__LIMIT}

        data = self.__make_get(path="/{}/child/page".format(page_id), params=params)

        children = [(page["id"], page["title"]) for page in data["results"]]
        while "next" in data["_links"]:
            data = self.__make_get(path=data["_links"]["next"].replace("/rest/api/content", ""))
            for page in data["results"]:
                children.append((page["id"], page["title"]))

        return children

    def delete_page(self, page_id):
        """Delete a page by its ID.

        :param str page_id: a page ID
        :return: an HTTP response
        :rtype: :class:`requests.Response`
        """
        logger.info("Delete page: %s", page_id)
        return self.__make_delete(path="/{}".format(page_id))

    def add_labels(self, page_id, labels, prefix="global"):
        """Add given labels with a specified prefix to a page.

        :param str page_id: a page ID
        :param labels: a list of labels
        :type: tuple(str) or list(str)
        """
        logger.info("Add labels: %s (prefix: %s) for page: %s", labels, prefix, page_id)
        if not labels:
            return
        return self.__make_request(
            method="post", path="/{}/label".format(page_id), data=[{"prefix": prefix, "name": label} for label in labels]
        )

    def delete_label(self, page_id, label):
        """Delete a label from a page.

        :param str page_id: a page ID
        :param str label: a label
        :return: an HTTP response
        :rtype: :class:`requests.Response`
        """
        logger.info("Delete label: %s for page: %s", label, page_id)
        return self.__make_delete(path="/{}/label/{}".format(page_id, label))

    def get_labels(self, page_id):
        """Get all labels in a page.

        :param str page_id: a page ID
        :return: a list of labels
        :rtype: list(str)
        """
        logger.info("Get labels for page: %s", page_id)
        params = {"limit": self.__LIMIT}

        data = self.__make_get(path="/{}/label".format(page_id), params=params)

        labels = [label["name"] for label in data["results"]]
        while "next" in data["_links"]:
            data = self.__make_get(path=data["_links"]["next"].replace("/rest/api/content", ""))
            labels.extend([label["name"] for label in data["results"]])

        return labels


class ConfluenceOAuthConnector(ConfluenceConnector):
    """Connector to Confluence API based on the OAuth1 authentication.

    :param str consumer_key: an OAuth consumer key
    :param str access_token: an OAuth access token
    :param str rsa_key: a public RSA key
    :param str http_proxy: a URL of the HTTP proxy
    :param str https_proxy: a URL of the HTTPS proxy
    :param str a Confluence: space key
    :param bool alert_users: True if the Confluence server must send alert to page watchers
    """

    # noinspection PyPackageRequirements
    # noinspection PyMissingConstructor
    def __init__(
        self, consumer_key, access_token, rsa_key, http_proxy=None, https_proxy=None, space_key=None, alert_users=True
    ):
        from oauthlib.oauth1 import SIGNATURE_RSA
        from requests_oauthlib import OAuth1

        self._auth = OAuth1(
            client_key=consumer_key, resource_owner_key=access_token, signature_method=SIGNATURE_RSA, rsa_key=rsa_key
        )
        self._headers = None
        self.space_key = space_key
        self.proxies = (
            {
                "http": http_proxy,
                "https": https_proxy,
            }
            if http_proxy and https_proxy
            else None
        )
        self.alert_users = alert_users


class ConfluencePATConnector(ConfluenceConnector):
    def __init__(
        self,
        token: str,
        http_proxy: Optional[str] = None,
        https_proxy: Optional[str] = None,
        space_key: Optional[str] = None,
        alert_users: bool = True,
    ):
        self._auth = None
        self._headers = {"Authorization": "Bearer {}".format(token)}
        self.space_key = space_key
        self.proxies = (
            {
                "http": http_proxy,
                "https": https_proxy,
            }
            if http_proxy and https_proxy
            else None
        )
        self.alert_users = alert_users
