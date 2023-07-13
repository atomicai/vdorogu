import io
import re
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from .. import tokens
from ..filesystem import FilesystemError
from .exceptions import EmailError


class Email:
    def __init__(self, send_from: Optional[str] = None, server: str = 'localhost'):
        if send_from is None:
            send_from = tokens.EMAIL_ADDRESS
            if send_from is None:
                raise FilesystemError('send_from email address not found; pass it explisitly')
        if self._is_valid_email(send_from):
            self.send_from = send_from
        else:
            raise EmailError(f'email address {send_from} is not valid')
        self.server = server

    def send(
        self,
        content: str,
        receivers: Union[str, Sequence[str]],
        *,
        subject: str,
        attachments: Optional[Iterable[Union[io.BytesIO, Path, str]]] = None,
        markup: Optional[str] = None,
    ) -> None:
        if not self._is_valid_email(receivers):
            raise EmailError(f'email address {receivers} is not valid')
        message = self._form_message(content, receivers, subject, attachments, markup)
        smtp_sender = smtplib.SMTP(self.server)
        try:
            smtp_sender.sendmail(self.send_from, receivers, message.as_string())
        finally:
            smtp_sender.close()

    def _is_valid_email(self, address: Union[str, Sequence[str]]) -> bool:
        """Simple check for email address format"""
        if isinstance(address, str):
            return len(re.findall(r'[^@]+@[^@]+\.[^@]+', address)) > 0
        if isinstance(address, (list, tuple, set)):
            return all([len(re.findall(r'[^@]+@[^@]+\.[^@]+', e)) > 0] for e in address)
        return False

    def _to_collection(
        self, obj: Union[io.BytesIO, Path, str, Iterable[Union[io.BytesIO, Path, str]]]
    ) -> Iterable[Union[io.BytesIO, Path, str]]:
        """Single value to collection"""
        if isinstance(obj, (io.BytesIO, Path, str)):
            return [obj]
        return obj

    @staticmethod
    def _prepare_attachment(name: str, content: Union[str, bytes]) -> MIMEApplication:
        part = MIMEApplication(content, Name=name)
        part['Content-Disposition'] = f'attachment; filename="{name}"'
        return part

    def _form_message(
        self,
        content: str,
        receivers: Union[str, Sequence[str]],
        subject: str,
        attachments: Optional[Iterable[Union[io.BytesIO, Path, str]]] = None,
        markup: Optional[str] = None,
    ) -> MIMEMultipart:
        if markup is None:
            markup = 'plain'
        message = MIMEMultipart()
        message['From'] = self.send_from
        message['To'] = ', '.join(map(str, self._to_collection(receivers)))
        message['Date'] = formatdate(localtime=True)
        message['Subject'] = subject
        message['Content-Type'] = "text/html; charset=utf-8"
        message.preamble = 'This is a multi-part message in MIME format.'
        message.attach(MIMEText(content, markup, 'utf-8'))
        if attachments:
            attachments = self._to_collection(attachments)
            for attachment in attachments:
                if isinstance(attachment, io.BytesIO):
                    if 'name' in dir(attachment):
                        part = self._prepare_attachment(attachment.name, attachment.read())
                    else:
                        part = self._prepare_attachment('noname.png', attachment.read())
                elif Path(attachment).is_file():
                    with open(attachment, 'rb') as attachment_file:
                        part = self._prepare_attachment(Path(attachment).name, attachment_file.read())
                else:
                    continue
                message.attach(part)
        return message
