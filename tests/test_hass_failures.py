"""HA websocket deadlines, response pairing, and redaction."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from wyoming_whisper_cpp.hass_api import HomeAssistant, HomeAssistantError


@pytest.mark.parametrize("mode", ["silent", "auth-invalid", "wrong-id"])
async def test_bad_server_is_bounded_and_does_not_echo_token(mode):
    token = "sensitive-token-do-not-log"

    async def respond(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if mode == "silent":
            await ws.receive()
            return ws
        await ws.send_json({"type": "auth_required"})
        await ws.receive_json()
        if mode == "auth-invalid":
            await ws.send_json({"type": "auth_invalid", "message": token})
            return ws
        await ws.send_json({"type": "auth_ok"})
        await ws.receive_json()
        await ws.send_json({"id": 999, "type": "result", "success": True, "result": {}})
        return ws

    app = web.Application()
    app.router.add_get("/api/websocket", respond)
    async with TestServer(app) as server:
        hass = HomeAssistant(token, str(server.make_url("/api")), timeout=0.05)
        with pytest.raises(HomeAssistantError) as error:
            await asyncio.wait_for(hass.get_context(), 1)
        assert token not in str(error.value)
