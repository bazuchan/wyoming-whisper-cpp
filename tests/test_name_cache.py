"""Refresh coalescing, deadlines, prompt limits, and stale-data fallback."""

import asyncio

import pytest

from wyoming_whisper_cpp.hass_api import HomeAssistantError
from wyoming_whisper_cpp.name_cache import HassNameCache
from wyoming_whisper_cpp.vocabulary import RecognitionContext, estimate_tokens


class FakeHass:
    def __init__(self):
        self.calls = 0
        self.context = RecognitionContext(
            priority_entities=["Office Lamp", "Desk Lamp"]
        )
        self.fail = False
        self.gate = asyncio.Event()
        self.gate.set()

    async def get_context(self):
        self.calls += 1
        await self.gate.wait()
        if self.fail:
            raise HomeAssistantError("Unavailable")
        return self.context


async def test_background_refresh_coalesces_and_throttles():
    hass = FakeHass()
    cache = HassNameCache(hass, prefix="Names:", refresh_seconds=60)
    try:
        for _ in range(10):
            cache.schedule_refresh()
        assert await cache.initial_prompt() == "Names: Office Lamp, Desk Lamp."
        cache.schedule_refresh()
        await cache.initial_prompt()
        assert hass.calls == 1
    finally:
        await cache.close()


async def test_stale_snapshot_survives_failure():
    hass = FakeHass()
    cache = HassNameCache(hass)
    assert await cache.refresh()
    hass.fail = True
    cache.schedule_refresh()
    try:
        assert await cache.initial_prompt() == "Office Lamp, Desk Lamp."
    finally:
        await cache.close()


async def test_timeout_does_not_cancel_refresh():
    hass = FakeHass()
    cache = HassNameCache(hass, wait_timeout=0.01)
    await cache.refresh()
    hass.gate.clear()
    hass.context = RecognitionContext(used_areas=["Kitchen"])
    cache.schedule_refresh()
    try:
        assert await cache.initial_prompt() == "Office Lamp, Desk Lamp."
        assert not cache._task.done()
        hass.gate.set()
        await cache._task
        assert await cache.initial_prompt() == "Kitchen."
    finally:
        await cache.close()


async def test_streaming_never_waits_for_network():
    hass = FakeHass()
    hass.gate.clear()
    cache = HassNameCache(hass, prefix="Names:")
    cache.schedule_refresh()
    try:
        assert await asyncio.wait_for(cache.initial_prompt(wait=False), 0.1) == "Names:"
    finally:
        await cache.close()
    assert cache._task.done()


async def test_same_snapshot_reuses_prompt_cache():
    hass = FakeHass()
    cache = HassNameCache(hass)
    await cache.refresh()
    context = cache._context
    hass.context = RecognitionContext(priority_entities=["Office Lamp", "Desk Lamp"])
    await cache.refresh()
    assert cache._context is context


async def test_long_prefix_and_multibyte_names_fit_budget():
    # Escapes keep this source ASCII while exercising real Cyrillic input.
    name = "\u041a\u0443\u0445\u043d\u044f"
    hass = FakeHass()
    hass.context = RecognitionContext(used_areas=[name] * 100)
    cache = HassNameCache(hass, prefix=name * 100, max_tokens=40)
    assert estimate_tokens(await cache.initial_prompt()) <= 40
    await cache.refresh()
    assert estimate_tokens(await cache.initial_prompt()) <= 40


@pytest.mark.parametrize("budget", [0, -1, 224])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        HassNameCache(FakeHass(), max_tokens=budget)
