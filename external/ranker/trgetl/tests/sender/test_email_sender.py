import getpass
import io
from pathlib import Path
from random import sample
from textwrap import dedent
from typing import Optional, Sequence, Union

import pytest
from _pytest.fixtures import SubRequest
from matplotlib import pyplot as plt
from mock import patch  # type: ignore

from ...sender import Email, EmailError

ENABLE_MOCKING = True
DEFAULT_KEYS_PATH = Path(__file__).parent.parent.parent / '__tokens__'
LOCALHOST = 'localhost'
SEND_FROM = 'rbhp-robots@we.mail.ru'
TEST_AUTHOR = 'd.kulemin'
TEST_RECEIVER = getpass.getuser() + '@corp.mail.ru'


class MailObject:
    def __init__(self, name: str, send_from: str, server: str, valid: bool):
        self.name = name
        self.send_from = send_from
        self.server = server
        self.valid = valid


MAILS = [
    MailObject('simple_mailer', SEND_FROM, LOCALHOST, True),
    MailObject('mailer_with_wrong_address', 'wrong@address', LOCALHOST, False),
]


def id_mail(mails: Sequence[MailObject]) -> Sequence[str]:
    return [mail.name for mail in mails]


@pytest.fixture(params=MAILS, ids=id_mail(MAILS))
def etalon_mail(request: SubRequest) -> MailObject:
    return request.param


@pytest.mark.skipif(getpass.getuser() != TEST_AUTHOR, reason='can not guarantee that token path exists')
def test_token_path_exists() -> None:
    assert DEFAULT_KEYS_PATH.exists(), 'there are should be a path to token files'


def test_import() -> None:
    from lib.trgetl.sender import MyteamBot  # noqa: F401
    from lib.trgetl.sender import __all__  # noqa: F401
    from lib.trgetl.sender import Email, EmailError  # noqa: F401


def test_email_initialization_v1(etalon_mail: MailObject) -> None:
    if not etalon_mail.valid:
        with pytest.raises(EmailError):
            Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
    else:
        mailer = Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
        assert mailer.send_from == etalon_mail.send_from, f'should be {etalon_mail.send_from}, but got {mailer.send_from}'
        assert mailer.server == etalon_mail.server, f'should be {etalon_mail.server}, but got {mailer.server}'


@pytest.mark.parametrize(
    'send_from, server, valid',
    [
        pytest.param(SEND_FROM, LOCALHOST, True, id=SEND_FROM),
        pytest.param('wrong@address', LOCALHOST, False, id='wrong@address'),
        pytest.param(
            None,
            LOCALHOST,
            True,
            marks=[
                pytest.mark.skipif(getpass.getuser() != TEST_AUTHOR, reason='can not guarantee that token path exists')
            ],
            id='NoneAddress',
        ),
    ],
)
def test_email_initialization_v2(send_from: str, server: str, valid: bool) -> None:
    if not valid:
        with pytest.raises(EmailError):
            Email(send_from=send_from, server=server)
    else:
        mailer = Email(send_from=send_from, server=server)
        assert mailer.send_from == send_from or SEND_FROM, f'should be {send_from}, but got {mailer.send_from}'
        assert mailer.server == server, f'should be {server}, but got {mailer.server}'


@pytest.mark.parametrize(
    'email, etalon_response',
    [
        pytest.param(SEND_FROM, True, id=SEND_FROM),
        pytest.param('abc@mail.com', True, id='abc@mail.com'),
        pytest.param('abc_def@mail.com', True, id='abc_def@mail.com'),
        pytest.param('abc-d@mail.com', True, id='abc-d@mail.com'),
        pytest.param('abc.def@mail.com', True, id='abc.def@mail.com'),
        pytest.param('abc.def@mail', False, id='abc.def@mail'),
        pytest.param('abc.def@', False, id='abc.def@'),
    ],
)
def test_check_if_email_address_is_valid(etalon_mail: MailObject, email: str, etalon_response: bool) -> None:
    if not etalon_mail.valid:
        with pytest.raises(EmailError):
            Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
    else:
        mailer = Email(send_from=etalon_mail.send_from)
        result = mailer._is_valid_email(email)
        assert result == etalon_response, f'should be {etalon_response}, but got {result}'


@pytest.mark.parametrize(
    'receivers, etalon_response',
    [
        pytest.param(SEND_FROM, [SEND_FROM], id='single_receiver'),
        pytest.param([SEND_FROM, TEST_RECEIVER], [SEND_FROM, TEST_RECEIVER], id='list_receivers'),
        pytest.param({SEND_FROM, TEST_RECEIVER}, {SEND_FROM, TEST_RECEIVER}, id='set_receivers'),
        pytest.param((SEND_FROM, TEST_RECEIVER), (SEND_FROM, TEST_RECEIVER), id='tuple_receivers'),
    ],
)
def test_to_collection(
    etalon_mail: MailObject, receivers: Union[str, Sequence[str]], etalon_response: Union[str, Sequence[str]]
) -> None:
    if not etalon_mail.valid:
        with pytest.raises(EmailError):
            Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
    else:
        mailer = Email(send_from=etalon_mail.send_from)
        result = mailer._to_collection(receivers)
        assert result == etalon_response, f'should be {etalon_response}, but got {result}'


@pytest.mark.parametrize(
    'content, receivers, subject, attach_name, markup, etalon_response',
    [
        pytest.param(
            'TEST text',
            TEST_RECEIVER,
            'simple',
            None,
            None,
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0'''
            ),
            id='simple',
        ),
        pytest.param(
            'TEST text',
            TEST_RECEIVER,
            'simple plain text',
            None,
            'plain',
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple plain text
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0'''
            ),
            id='simple_plain_text',
        ),
        pytest.param(
            dedent(
                """\
                <html>
                    <head></head>
                    <body>
                        <p>TEST text</p>
                    </body>
                </html>
            """
            ),
            TEST_RECEIVER,
            'simple html text',
            None,
            'html',
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple html text
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/html; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                PGh0bWw+CiAgICA8aGVhZD48L2hlYWQ+CiAgICA8Ym9keT4KICAgICAgICA8cD5URVNUIHRleHQ8
                L3A+CiAgICA8L2JvZHk+CjwvaHRtbD4K'''
            ),
            id='simple_html_text',
        ),
        pytest.param(
            'TEST text',
            ['a@mail.com', 'b@mail.com', 'c@mail.com'],
            'multiple receivers',
            None,
            None,
            dedent(
                '''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: a@mail.com, b@mail.com, c@mail.com
                Subject: multiple receivers
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0'''
            ),
            id='multiple_receivers',
        ),
        pytest.param(
            'TEST text',
            TEST_RECEIVER,
            'simple with attach',
            'attachment.png',
            None,
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple with attach
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0
                Content-Type: application/octet-stream; Name="attachment.png"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                Content-Disposition: attachment; filename="attachment.png"'''
            ),
            id='simple_with_attachment',
        ),
        pytest.param(
            'TEST text',
            TEST_RECEIVER,
            'simple with multiple attach',
            ['att1.png', 'att2.png'],
            None,
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple with multiple attach
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0
                Content-Type: application/octet-stream; Name="att1.png"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                Content-Disposition: attachment; filename="att1.png"
                Content-Type: application/octet-stream; Name="att2.png"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                Content-Disposition: attachment; filename="att2.png"'''
            ),
            id='multiple_attachments',
        ),
    ],
)
def test_form_message_v1(
    etalon_mail: MailObject,
    content: str,
    receivers: Union[str, Sequence[str]],
    subject: str,
    attach_name: Optional[Union[str, list, set, tuple]],
    markup: Optional[str],
    etalon_response: str,
) -> None:
    if not etalon_mail.valid:
        with pytest.raises(EmailError):
            Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
    else:
        mailer = Email(send_from=etalon_mail.send_from)
        if attach_name:
            if not isinstance(attach_name, (list, set, tuple)):
                attach_name = [attach_name]
            attach = []
            for att in attach_name:
                tmp = io.BytesIO()
                tmp.name = att
                attach.append(tmp)
        else:
            attach = None
        markup = markup or 'plain'
        message = mailer._form_message(content, receivers, subject, attach, markup).as_string()
        result_list = []
        for line in message.split('\n'):
            line = line.split('; boundary')[0]
            if line and all(token not in line for token in ['==', 'Date', 'From']):
                result_list.append(line)
        result = '\n'.join(result_list)
        assert result == etalon_response, f'should be {etalon_response}, but got {result}'


@pytest.mark.parametrize(
    'content, receivers, subject, attach_name, markup, etalon_response',
    [
        pytest.param(
            'TEST text',
            TEST_RECEIVER,
            'simple',
            None,
            None,
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0'''
            ),
            id='simple',
        ),
        pytest.param(
            'TEST text',
            TEST_RECEIVER,
            'simple plain text',
            None,
            'plain',
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple plain text
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0'''
            ),
            id='simple_plain_text',
        ),
        pytest.param(
            dedent(
                """\
                <html>
                    <head></head>
                    <body>
                        <p>TEST text</p>
                    </body>
                </html>
            """
            ),
            TEST_RECEIVER,
            'simple html text',
            None,
            'html',
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple html text
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/html; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                PGh0bWw+CiAgICA8aGVhZD48L2hlYWQ+CiAgICA8Ym9keT4KICAgICAgICA8cD5URVNUIHRleHQ8
                L3A+CiAgICA8L2JvZHk+CjwvaHRtbD4K'''
            ),
            id='simple_html_text',
        ),
        pytest.param(
            'TEST text',
            ['a@mail.com', 'b@mail.com', 'c@mail.com'],
            'multiple receivers',
            None,
            None,
            dedent(
                '''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: a@mail.com, b@mail.com, c@mail.com
                Subject: multiple receivers
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0'''
            ),
            id='multiple_receivers',
        ),
        pytest.param(
            'TEST text',
            TEST_RECEIVER,
            'simple with attach',
            'attachment.png',
            None,
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple with attach
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0
                Content-Type: application/octet-stream; Name="attachment.png"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                Content-Disposition: attachment; filename="attachment.png"'''
            ),
            id='simple_with_attachment',
        ),
        pytest.param(
            'TEST text',
            TEST_RECEIVER,
            'simple with multiple attach',
            ['att1.png', 'att2.png'],
            None,
            dedent(
                f'''\
                Content-Type: multipart/mixed
                MIME-Version: 1.0
                To: {TEST_RECEIVER}
                Subject: simple with multiple attach
                Content-Type: multipart/mixed
                This is a multi-part message in MIME format.
                Content-Type: text/plain; charset="utf-8"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                VEVTVCB0ZXh0
                Content-Type: application/octet-stream; Name="att1.png"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                Content-Disposition: attachment; filename="att1.png"
                Content-Type: application/octet-stream; Name="att2.png"
                MIME-Version: 1.0
                Content-Transfer-Encoding: base64
                Content-Disposition: attachment; filename="att2.png"'''
            ),
            id='multiple_attachments',
        ),
    ],
)
def test_form_message_v2(
    tmp_path: Path,
    etalon_mail: MailObject,
    content: str,
    receivers: Union[str, Sequence[str]],
    subject: str,
    attach_name: Optional[Union[str, list, set, tuple]],
    markup: Optional[str],
    etalon_response: str,
) -> None:
    if not etalon_mail.valid:
        with pytest.raises(EmailError):
            Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
    else:
        mailer = Email(send_from=etalon_mail.send_from)
        if attach_name:
            if not isinstance(attach_name, (list, set, tuple)):
                attach_name = [attach_name]
            attach = []
            for att in attach_name:
                tmp = tmp_path / att
                tmp.write_bytes(io.BytesIO().read())
                attach.append(tmp)
        else:
            attach = None
        markup = markup or 'plain'
        message = mailer._form_message(content, receivers, subject, attach, markup).as_string()
        result_list = []
        for line in message.split('\n'):
            line = line.split('; boundary')[0]
            if line and all(token not in line for token in ['==', 'Date', 'From']):
                result_list.append(line)
        result = '\n'.join(result_list)
        assert result == etalon_response, f'should be {etalon_response}, but got {result}'


@pytest.mark.parametrize(
    'content, encoded_text, receivers, subject, attach_name, markup, should_raise',
    [
        pytest.param('TEST text', 'VEVTVCB0ZXh0', TEST_RECEIVER, 'simple', None, None, False, id='simple'),
        pytest.param(
            'TEST text', 'VEVTVCB0ZXh0', 'wrong@address', 'wrong_receiver', None, None, True, id='wrong_receiver'
        ),
        pytest.param(
            'TEST text',
            'VEVTVCB0ZXh0',
            TEST_RECEIVER,
            'simple with attach',
            'attachment.png',
            None,
            False,
            id='simple_with_attachment',
        ),
        pytest.param(
            'TEST text',
            'VEVTVCB0ZXh0',
            TEST_RECEIVER,
            'simple with multiple attach',
            ['att1.png', 'att2.png'],
            None,
            False,
            id='multiple_attachments',
        ),
    ],
)
def test_send(
    tmp_path: Path,
    etalon_mail: MailObject,
    content: str,
    encoded_text: str,
    receivers: Union[str, Sequence[str]],
    subject: str,
    attach_name: Optional[Union[str, Sequence[str]]],
    markup: str,
    should_raise: bool,
) -> None:
    if not etalon_mail.valid:
        with pytest.raises(EmailError):
            Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
    elif should_raise:
        with pytest.raises(EmailError):
            mailer = Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
            if ENABLE_MOCKING:
                with patch('smtplib.SMTP', autospec=True) as mock_smtp:
                    mailer.send(content, receivers, subject=subject, markup=markup)
            else:
                mailer.send(content, receivers, subject=subject, markup=markup)
    else:
        mailer = Email(send_from=etalon_mail.send_from, server=etalon_mail.server)
        if attach_name:
            if isinstance(attach_name, str):
                attach_name = [attach_name]
            attach = []
            for att in attach_name:
                tmp = tmp_path / att
                plot_object = io.BytesIO()
                plt.plot(sorted(sample(range(100), 10)), sample(range(10), 10))
                plt.savefig(plot_object)
                plt.close()
                plot_object.seek(0)
                tmp.write_bytes(plot_object.read())
                attach.append(tmp)
        else:
            attach = None
        if ENABLE_MOCKING:
            with patch('smtplib.SMTP', autospec=True) as mock_smtp:
                mailer.send(content, receivers, subject=subject, attachments=attach, markup=markup)
                mock_smtp.assert_called()
                name, args, kwargs = mock_smtp.method_calls.pop(0)
                assert name == '().sendmail' and {} == kwargs, 'sendmail() method was not called'
                from_, to_, body_ = args
                assert etalon_mail.send_from == from_, f'should be {etalon_mail.send_from}, but got {from_}'
                assert receivers == to_, f'should be {receivers}, but got {to_}'
                assert encoded_text in body_, f'{encoded_text} should be in {body_}'
        else:
            mailer.send(content, receivers, subject=subject, attachments=attach, markup=markup)
