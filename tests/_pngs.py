"""Tiny but valid PNG generator for tests.

python-docx verifies PNG chunk CRCs when embedding images, so any test
that exercises the docx renderer can't use hand-rolled fake bytes.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def make_minimal_png(width: int = 1, height: int = 1) -> bytes:
    """Build a valid white-pixel RGB PNG using only stdlib (zlib + struct)."""

    def chunk(type_: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(type_ + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + type_ + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),  # 8-bit RGB
    )
    # Each scanline = 1 filter byte (0 = None) + width * 3 bytes (RGB white).
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def write_fake_png(path: Path) -> None:
    """Write a tiny but valid PNG to *path* (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_minimal_png())
