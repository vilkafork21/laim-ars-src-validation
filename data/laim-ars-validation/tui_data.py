"""Цветовая схема консольного вывода валидации."""

from __future__ import annotations

from dataclasses import dataclass

from tui import ColorSchemeDataScience
from tui_core import AnsiCodes


@dataclass(frozen=True)
class ColorSchemeDataScienceCold(ColorSchemeDataScience):
    header: str = (
        AnsiCodes.bg_true(10, 25, 47)
        + AnsiCodes.fg_true(255, 255, 255)
        + AnsiCodes.bold
    )
    subheader: str = (
        AnsiCodes.fg_true(0, 180, 216) + AnsiCodes.bold + AnsiCodes.underline
    )
    perf_metric: str = (
        AnsiCodes.bg_true(255, 215, 0) + AnsiCodes.fg_true(0, 0, 0) + AnsiCodes.bold
    )
