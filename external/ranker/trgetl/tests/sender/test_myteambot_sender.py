import getpass
import io
from pathlib import Path
from random import sample, uniform
from typing import List, Optional, Sequence, Union

import pytest
from _pytest.fixtures import SubRequest
from matplotlib import pyplot as plt
from mock import patch  # type: ignore

from ... import tokens
from ...sender import MyteamBot

ENABLE_MOCKING = True
DEFAULT_KEYS_PATH = Path(__file__).parent.parent.parent / "__tokens__"
TEST_AUTHOR = "d.kulemin"
TEST_RECEIVER = getpass.getuser() + "@corp.mail.ru"


class BotObject:
    def __init__(self, name: str, token: Optional[str], valid: bool):
        self.name = name
        self.token = token
        self.valid = valid


BOTS = [
    BotObject("normal_bot", None, True),
]


def id_bot(bots: Sequence[BotObject]) -> Sequence[str]:
    return [bot.name for bot in bots]


@pytest.fixture(params=BOTS, ids=id_bot(BOTS))
def etalon_bot(request: SubRequest) -> BotObject:
    return request.param


@pytest.mark.skipif(getpass.getuser() != TEST_AUTHOR, reason="can not guarantee that token path exists")
def test_token_path_exists() -> None:
    assert DEFAULT_KEYS_PATH.exists(), "there are should be a path to token files"


def test_import() -> None:
    from lib.trgetl.sender import MyteamBot  # noqa: F401, F811
    from lib.trgetl.sender import __all__  # noqa: F401


@pytest.mark.skipif(getpass.getuser() != TEST_AUTHOR, reason="can not guarantee that token path exists")
def test_myteambot_initialization(etalon_bot: BotObject) -> None:
    bot = MyteamBot(etalon_bot.token)
    assert bot.token == etalon_bot.token or tokens.MYTEAMBOT_TOKEN, f"token should not be None, but got {bot.token}"


@pytest.mark.skipif(getpass.getuser() != TEST_AUTHOR, reason="can not guarantee that token path exists")
@pytest.mark.parametrize(
    "content, receivers, attach_name, markup",
    [
        pytest.param("simple message", TEST_RECEIVER, None, None, id="simple_message"),
        pytest.param(
            "simple message to multiple receivers", [TEST_RECEIVER, TEST_RECEIVER], None, None, id="multiple_receivers"
        ),
        pytest.param("", TEST_RECEIVER, "attachment.png", None, id="only_one_attachment_raises"),
        pytest.param(
            "", [TEST_RECEIVER, TEST_RECEIVER], "attachment.png", None, id="multiple_receivers_attachment_raises"
        ),
        pytest.param(
            "message and attachment_raises", TEST_RECEIVER, "attachment.png", None, id="message_and_attachment_raises"
        ),
        pytest.param("", TEST_RECEIVER, ["attachment.png"], None, id="only_one_attachment"),
        pytest.param("", [TEST_RECEIVER, TEST_RECEIVER], ["attachment.png"], None, id="multiple_receivers_attachment"),
        pytest.param("message and attachment", TEST_RECEIVER, ["attachment.png"], None, id="message_and_attachment"),
        pytest.param(
            "message and multiple attachments",
            TEST_RECEIVER,
            ["attach1.png", "attach2.png"],
            None,
            id="message_and_multiple_attachment",
        ),
        pytest.param(
            "message and multiple attachments to multiple receivers",
            [TEST_RECEIVER, TEST_RECEIVER],
            ["attach1.png", "attach2.png"],
            None,
            id="message_and_multiple_attachment_to_multiple_receivers",
        ),
    ],
)
def test_send(
    tmp_path: Path,
    etalon_bot: BotObject,
    content: str,
    receivers: Union[str, Sequence[str]],
    attach_name: Optional[Union[str, Sequence[str]]],
    markup: Optional[str],
) -> None:
    def prepare_attachment(name: str, path: Path) -> Union[Path, io.BytesIO]:
        tmp = path / name
        plot_object = io.BytesIO()
        plt.plot(sorted(sample(range(100), 10)), sample(range(10), 10))
        plt.savefig(plot_object)
        plt.close()
        plot_object.seek(0)
        tmp.write_bytes(plot_object.read())
        return tmp if uniform(0, 1) > 0.5 else plot_object

    def actual_send(bot: MyteamBot, kwargs: dict) -> None:
        if ENABLE_MOCKING:
            with patch("requests.post", autospec=True) as mock_post:
                bot.send(
                    kwargs["content"],
                    kwargs["receivers"],
                    attachments=kwargs["attachments"],
                    markup=kwargs["markup"],
                )
                mock_post.assert_called()
        else:
            bot.send(
                kwargs["content"],
                kwargs["receivers"],
                attachments=kwargs["attachments"],
                markup=kwargs["markup"],
            )

    bot = MyteamBot(etalon_bot.token)
    if attach_name:
        if isinstance(attach_name, str):
            with pytest.raises(ValueError):
                actual_send(
                    bot,
                    dict(
                        content=content,
                        receivers=receivers,
                        attachments=prepare_attachment(attach_name, tmp_path),
                        markup=markup,
                    ),
                )
        else:
            attach: List[Union[io.BytesIO, Path, str]] = []
            for att in attach_name:
                attach.append(prepare_attachment(att, tmp_path))
            actual_send(
                bot,
                dict(
                    content=content,
                    receivers=receivers,
                    attachments=attach,
                    markup=markup,
                ),
            )
    else:
        actual_send(
            bot,
            dict(
                content=content,
                receivers=receivers,
                attachments=attach_name,
                markup=markup,
            ),
        )
