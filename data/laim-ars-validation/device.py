"""Настройка движка Polars для расчёта валидации."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Device:
    name: str = "cpu"

    @staticmethod
    def of(value: str = "", fallback: "Device | None" = None) -> "Device":
        """Выбирает устройство из параметра или переменной ARS_DEVICE."""
        from os import environ

        selected = str(value).strip() or environ.get("ARS_DEVICE", "").strip()
        return Device(selected.lower()) if selected else (fallback or Device())

    @property
    def engine(self) -> Literal["gpu", "streaming"]:
        return "gpu" if self.name == "gpu" else "streaming"
