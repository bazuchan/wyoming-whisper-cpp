"""Resolve GGML files without relying on the binding's implicit downloads."""

import os
import platform
import re
import tempfile
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"


def resolve_name(name: str) -> str:
    """Keep auto/int8 CLI conventions, but use whisper.cpp quantization names."""
    if name == "auto":
        machine = platform.machine().lower()
        return "tiny-q8_0" if "arm" in machine or "aarch" in machine else "base-q8_0"
    return re.sub(r"[.-]int8$", "-q8_0", name)


def resolve_model(
    name: str,
    data_dirs: list[Path],
    download_dir: Path,
    local_files_only: bool = False,
) -> Path:
    """Search all data directories before downloading atomically into the cache."""
    direct = Path(name).expanduser()
    if direct.is_file():
        return direct.resolve()
    if direct.suffix == ".bin" or "/" in name or "\\" in name:
        raise FileNotFoundError(f"Model file does not exist: {name}")

    name = resolve_name(name)
    filename = f"ggml-{name}.bin"
    for directory in [*data_dirs, download_dir]:
        candidate = directory / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    if local_files_only:
        raise FileNotFoundError(f"Model {filename} not found; downloads are disabled")

    from pywhispercpp.constants import AVAILABLE_MODELS

    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {name}. Use a supported name or a local .bin file"
        )
    download_dir.mkdir(parents=True, exist_ok=True)
    destination = download_dir / filename
    temporary = None
    try:
        with urlopen(MODEL_URL.format(name=name), timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=download_dir, delete=False) as output:
                temporary = Path(output.name)
                size = 0
                while block := response.read(1024 * 1024):
                    output.write(block)
                    size += len(block)
                expected = response.headers.get("Content-Length")
                if not size or (expected is not None and size != int(expected)):
                    raise OSError("Incomplete model download")
                output.flush()
                os.fsync(output.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination.resolve()
