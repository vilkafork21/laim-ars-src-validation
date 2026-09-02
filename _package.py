"""Детерминированная упаковка исполнимого ядра validation."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import tarfile
from pathlib import Path


EXCLUDED_DIRECTORIES = frozenset({".git", ".venv", "__pycache__"})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})


def _source_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not EXCLUDED_DIRECTORIES.intersection(path.relative_to(root).parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    )


def build_archive(root: Path) -> bytes:
    """Возвращает tar.gz с каноническими метаданными.

    Одинаковый набор исходников обязан давать один checksum на каждом запуске:
    этот checksum идентифицирует реализацию отчёта в SberDS и DataLab.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"каталог validation не найден: {root}")
    files = _source_files(root)
    if not files or not (root / "validation.py").is_file():
        raise FileNotFoundError(
            f"в {root} нет полного ядра validation с entrypoint validation.py"
        )

    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", fileobj=compressed, mode="wb", compresslevel=9, mtime=0
    ) as gzip_stream:
        with tarfile.open(fileobj=gzip_stream, mode="w") as archive:
            for path in files:
                payload = path.read_bytes()
                member = tarfile.TarInfo(path.relative_to(root).as_posix())
                member.size = len(payload)
                member.mode = 0o644
                member.mtime = 0
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                archive.addfile(member, io.BytesIO(payload))
    return compressed.getvalue()


def encode_archive(raw: bytes) -> tuple[str, str]:
    return (
        base64.b64encode(raw).decode("ascii"),
        hashlib.sha256(raw).hexdigest(),
    )
