"""Короткий консольный вывод хода валидации."""

from __future__ import annotations

from dataclasses import dataclass
from shutil import get_terminal_size
from typing import Any

from tui_core import AnsiCodes, ColorScheme


def sprint(*args: Any, style_code: str = "", sep: str = " ", end: str = "\n") -> None:
    print(f"{style_code}{sep.join(map(str, args))}{AnsiCodes.reset}", sep="", end=end)


@dataclass(frozen=True)
class ColorSchemeSystem(ColorScheme):
    subheader: str = ""


@dataclass(frozen=True)
class ColorSchemeDataScience(ColorSchemeSystem):
    perf_metric: str = ""

    def print_section(self, title: str) -> None:
        sprint(Layout.section(self, title), style_code=self.header)

    def print_subsection(self, title: str) -> None:
        sprint(Layout.subsection(self, title), style_code=self.subheader)

    def print_performance_metrics(
        self,
        label: str,
        elapsed: float,
        cpu_peak: float | None = None,
        gpu_peak: float | None = None,
    ) -> None:
        sprint(
            Layout.performance_line(label, elapsed, cpu_peak, gpu_peak),
            style_code=self.perf_metric,
        )


class Layout:
    @staticmethod
    def _width(default: int = 100) -> int:
        return get_terminal_size((default, 24)).columns

    @staticmethod
    def section(scheme: ColorSchemeSystem, title: str) -> str:
        pad = max(Layout._width() - len(title) - 10, 0)
        return f"\n{scheme.header}\t{title}{' ' * pad}"

    @staticmethod
    def subsection(scheme: ColorSchemeSystem, title: str) -> str:
        pad = max(Layout._width() - len(title), 0)
        return f"\n{scheme.subheader}{title}{' ' * pad}"

    @staticmethod
    def performance_line(
        label: str,
        elapsed: float,
        cpu_peak: float | None,
        gpu_peak: float | None,
    ) -> str:
        memory = (
            f"пик RAM {cpu_peak:.2f} MB"
            if cpu_peak is not None
            else "пик RAM недоступен"
        )
        video = (
            f"пик VRAM {gpu_peak:.2f} MB"
            if gpu_peak is not None
            else "пик VRAM недоступен"
        )
        return f">> {label}: {memory}; {video}; {elapsed:.5f} сек"
