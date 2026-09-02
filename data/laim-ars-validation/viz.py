"""Рендеринг графиков Altair в самодостаточный HTML-отчёт."""

from __future__ import annotations

from base64 import b64encode
from importlib.util import find_spec
from pathlib import Path, PurePath
from typing import Any


class Html:
    @staticmethod
    def image(title: str, encoded: str) -> str:
        if not encoded:
            return ""
        return (
            '<div class="fig"><img '
            f'alt="{title}" src="data:image/png;base64,{encoded}"></div>'
        )


def img_to_base64(path: PurePath | None) -> str:
    source = Path(path) if path is not None else None
    return (
        b64encode(source.read_bytes()).decode("utf-8")
        if source and source.exists()
        else ""
    )


def save_chart(chart: Any, path: PurePath) -> PurePath | None:
    """Сохраняет график в PNG, если доступен конвертер Vega-Lite."""
    if find_spec("vl_convert") is None:
        return None
    target = Path(path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    chart.save(target.as_posix(), scale_factor=1.5)
    return target
