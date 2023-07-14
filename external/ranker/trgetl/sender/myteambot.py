import getpass
import io
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, Sequence, Union

import requests

from .. import tokens
from ..filesystem import FilesystemError
from .exceptions import MyteamBotError


class MyteamBot:
    API_URL_BASE = "https://api.internal.myteam.mail.ru/bot/v1"

    def __init__(self, token: Optional[str] = None):
        if token is None:
            token = tokens.MYTEAMBOT_TOKEN
            if token is None:
                raise FilesystemError("Myteam token not found; pass it explisitly")

        self.token = token

    def send(
        self,
        content: str,
        receivers: Union[str, Sequence[str]] = None,
        *,
        subject: Optional[str] = None,
        attachments: Optional[Iterable[Union[io.BytesIO, Path, str]]] = None,
        markup: Optional[str] = None,
    ) -> None:
        if receivers is None:
            receivers = getpass.getuser() + "@corp.mail.ru"

        if isinstance(receivers, (list, tuple, set)):
            for receiver in receivers:
                self.send(content, receiver, attachments=attachments, markup=markup)
            return

        if content:
            self._send_text(receivers, content, markup)
        if isinstance(attachments, (list, tuple, set)):
            for attachment in attachments:
                if isinstance(attachment, io.BytesIO):
                    attachment.seek(0)
                    self._send_file(receivers, attachment, "", markup)
                elif isinstance(attachment, (str, Path)):
                    path = Path(attachment)
                    assert path.is_file(), "Attachment not found: {}".format(attachment)
                    with path.open("rb") as f:
                        self._send_file(receivers, f, "", markup)
                else:
                    raise ValueError(f"Wrong type of argument 'attachment': {type(attachment).__name__}")
        elif attachments is not None:
            raise ValueError(
                f"Wrong type of argument 'attachments': {type(attachments).__name__}, " + "expected list, set or tuple"
            )

    def _send_text(
        self,
        chat_id: Union[str, Sequence[str]],
        text: str,
        markup: Optional[str] = None,
    ) -> str:
        return self._send_object(chat_id, text, None, markup)

    def _send_file(
        self,
        chat_id: Union[str, Sequence[str]],
        file: Union[io.BytesIO, BinaryIO],
        caption: str = None,
        markup: Optional[str] = None,
    ) -> str:
        return self._send_object(chat_id, file, caption, markup)

    def _send_object(
        self,
        chat_id: Union[str, Sequence[str]],
        obj: Union[str, io.BytesIO, BinaryIO],
        caption: str = None,
        markup: Optional[str] = None,
    ) -> str:
        url = self.API_URL_BASE
        files = dict()
        parameters = {
            "token": self.token,
            "chatId": chat_id,
        }

        if isinstance(obj, str):
            url += "/messages/sendText"
            parameters["text"] = obj
        else:
            url += "/messages/sendFile"
            if caption is not None:
                parameters["caption"] = caption
            files["file"] = obj

        if markup is not None:
            assert markup in ("MarkdownV2", "HTML"), f"Wrong parse_mode {markup}"
            parameters["parseMode"] = markup

        response = requests.post(url, params=parameters, files=files)
        self._raise_request_errors(response)
        return response.text

    def _raise_request_errors(self, response: requests.Response) -> None:
        response.raise_for_status()
        if not response.json()["ok"]:
            raise MyteamBotError(response.json()["description"])
