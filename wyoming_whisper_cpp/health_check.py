"""Check Wyoming Describe/Info instead of merely opening the TCP port."""

import argparse
import asyncio

from wyoming.client import AsyncClient
from wyoming.info import Describe, Info


async def check(uri: str) -> None:
    async with AsyncClient.from_uri(uri) as client:
        await client.write_event(Describe().event())
        event = await client.read_event()
        if (
            event is None
            or not Info.is_type(event.type)
            or not Info.from_event(event).asr
        ):
            raise RuntimeError("Server did not advertise an ASR program")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://127.0.0.1:10300")
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(check(args.uri), timeout=5))


if __name__ == "__main__":
    run()
