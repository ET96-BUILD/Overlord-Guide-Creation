"""Bundle the per-run output directory into a single sop_bundle.zip.

The zip wraps everything the user needs in one downloadable file:
sop.json, sop.docx, and the images/ folder. It's both a CLI artifact
(written next to the other outputs) and the body of the
GET /v1/jobs/<job_id>/zip endpoint.
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Name of the bundle file itself — kept as a constant so the recursion
# guard and any external code can agree on what to skip.
BUNDLE_FILENAME = "sop_bundle.zip"


# ── Public API ──────────────────────────────────────────────────────────


def write_zip(output_dir: Path, zip_path: Path) -> Path:
    """Bundle every file under *output_dir* into *zip_path*.

    The recursion guard skips any file already named ``sop_bundle.zip``
    so a re-run that overwrites an existing bundle never includes the
    previous bundle inside itself.

    Members are stored with forward-slash arcnames so the archive is
    portable across platforms.
    """
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)

    if not output_dir.is_dir():
        raise FileNotFoundError(f"output_dir does not exist: {output_dir}")

    # Collect the file list BEFORE opening the zip for write. This way,
    # if zip_path lives inside output_dir, the in-progress zip file
    # cannot accidentally enter the listing.
    members: list[Path] = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_dir():
            continue
        if p.name == BUNDLE_FILENAME:
            continue
        members.append(p)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            arcname = p.relative_to(output_dir).as_posix()
            zf.write(p, arcname=arcname)

    logger.info(
        "ZIP saved → %s (%d members)", zip_path, len(members)
    )
    return zip_path


# ── Slug helper ─────────────────────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(s: str | None, *, max_len: int = 60) -> str:
    """Lowercase + hyphenate + collapse runs + strip + cap.

    Returns ``"sop"`` if the input slugifies to an empty string, so
    callers can use the result as a filename without a separate
    fallback.
    """
    if not s:
        return "sop"
    out = _NON_ALNUM.sub("-", s.lower()).strip("-")
    if len(out) > max_len:
        out = out[:max_len].rstrip("-")
    return out or "sop"
