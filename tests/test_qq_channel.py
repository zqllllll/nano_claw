from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.qq import QQChannel
from nanobot.channels.qq import QQConfig


class _FakeApi:
    def __init__(self) -> None:
        self.c2c_calls: list[dict] = []
        self.group_calls: list[dict] = []

    async def post_c2c_message(self, **kwargs) -> None:
        self.c2c_calls.append(kwargs)

    async def post_group_message(self, **kwargs) -> None:
        self.group_calls.append(kwargs)


class _FakeClient:
    def __init__(self) -> None:
        self.api = _FakeApi()


@pytest.mark.asyncio
async def test_on_group_message_routes_to_group_chat_id() -> None:
    channel = QQChannel(QQConfig(app_id="app", secret="secret", allow_from=["user1"]), MessageBus())

    data = SimpleNamespace(
        id="msg1",
        content="hello",
        group_openid="group123",
        author=SimpleNamespace(member_openid="user1"),
    )

    await channel._on_message(data, is_group=True)

    msg = await channel.bus.consume_inbound()
    assert msg.sender_id == "user1"
    assert msg.chat_id == "group123"


@pytest.mark.asyncio
async def test_send_group_message_uses_plain_text_group_api_with_msg_seq() -> None:
    channel = QQChannel(QQConfig(app_id="app", secret="secret", allow_from=["*"]), MessageBus())
    channel._client = _FakeClient()
    channel._chat_type_cache["group123"] = "group"

    await channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="group123",
            content="hello",
            metadata={"message_id": "msg1"},
        )
    )

    assert len(channel._client.api.group_calls) == 1
    call = channel._client.api.group_calls[0]
    assert call == {
        "group_openid": "group123",
        "msg_type": 0,
        "content": "hello",
        "msg_id": "msg1",
        "msg_seq": 2,
    }
    assert not channel._client.api.c2c_calls


@pytest.mark.asyncio
async def test_send_c2c_message_uses_plain_text_c2c_api_with_msg_seq() -> None:
    channel = QQChannel(QQConfig(app_id="app", secret="secret", allow_from=["*"]), MessageBus())
    channel._client = _FakeClient()

    await channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="user123",
            content="hello",
            metadata={"message_id": "msg1"},
        )
    )

    assert len(channel._client.api.c2c_calls) == 1
    call = channel._client.api.c2c_calls[0]
    assert call == {
        "openid": "user123",
        "msg_type": 0,
        "content": "hello",
        "msg_id": "msg1",
        "msg_seq": 2,
    }
    assert not channel._client.api.group_calls


@pytest.mark.asyncio
async def test_send_group_message_uses_markdown_when_configured() -> None:
    channel = QQChannel(
        QQConfig(app_id="app", secret="secret", allow_from=["*"], msg_format="markdown"),
        MessageBus(),
    )
    channel._client = _FakeClient()
    channel._chat_type_cache["group123"] = "group"

    await channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="group123",
            content="**hello**",
            metadata={"message_id": "msg1"},
        )
    )

    assert len(channel._client.api.group_calls) == 1
    call = channel._client.api.group_calls[0]
    assert call == {
        "group_openid": "group123",
        "msg_type": 2,
        "markdown": {"content": "**hello**"},
        "msg_id": "msg1",
        "msg_seq": 2,
    }


@pytest.mark.asyncio
async def test_on_message_accepts_media_only_message(monkeypatch) -> None:
    channel = QQChannel(QQConfig(app_id="app", secret="secret", allow_from=["*"]), MessageBus())

    async def _fake_collect(_data, _sender):
        return ["/tmp/pic.png"], ["[image: /tmp/pic.png]"]

    monkeypatch.setattr(channel, "_collect_inbound_media", _fake_collect)

    data = SimpleNamespace(
        id="msg_media",
        content="",
        author=SimpleNamespace(id="user123", user_openid="user123"),
    )

    await channel._on_message(data, is_group=False)

    msg = await channel.bus.consume_inbound()
    assert msg.chat_id == "user123"
    assert msg.media == ["/tmp/pic.png"]
    assert "[image: /tmp/pic.png]" in msg.content


@pytest.mark.asyncio
async def test_collect_inbound_media_downloads_attachments(monkeypatch) -> None:
    channel = QQChannel(QQConfig(app_id="app", secret="secret", allow_from=["*"]), MessageBus())

    async def _fake_download(url, sender_id, filename=None, content_type=None):
        assert url == "https://cdn.example.com/a.png"
        assert sender_id == "user123"
        assert filename == "a.png"
        assert content_type == "image/png"
        return "/tmp/a.png"

    monkeypatch.setattr(channel, "_download_attachment", _fake_download)

    data = SimpleNamespace(
        attachments=[
            SimpleNamespace(
                url="https://cdn.example.com/a.png",
                filename="a.png",
                content_type="image/png",
            )
        ]
    )

    paths, tags = await channel._collect_inbound_media(data, "user123")
    assert paths == ["/tmp/a.png"]
    assert tags == ["[image: /tmp/a.png]"]
