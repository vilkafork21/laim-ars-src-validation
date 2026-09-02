"""Замер времени и пикового потребления памяти расчёта."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from shutil import which
from subprocess import DEVNULL, PIPE, run
from time import sleep, time
from typing import Any, Callable

from psutil import Process


def _probe(args: tuple[str, ...]) -> str | None:
    completed = (
        run(args, stdout=PIPE, stderr=DEVNULL, text=True, check=False)
        if which(args[0])
        else None
    )
    return (
        completed.stdout
        if completed is not None and completed.returncode == 0
        else None
    )


@dataclass(frozen=True)
class Hardware:
    gpu: bool = _probe(("nvidia-smi",)) is not None


def _cpu_memory_mb() -> float:
    return Process().memory_info().rss / (1024 * 1024)


def _gpu_memory_mb() -> float | None:
    output = _probe(
        ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
    )
    return float(output.strip().splitlines()[0]) if output is not None else None


def measure_during_call(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    interval_sec: float = 0.05,
    prec: int = 5,
) -> tuple[Any, float, float | None, float | None]:
    """Исполняет функцию и возвращает результат, время и пиковые RAM/VRAM."""

    def read() -> tuple[float, float | None]:
        gpu = _gpu_memory_mb() if Hardware.gpu else None
        return round(_cpu_memory_mb(), prec), (
            round(gpu, prec) if gpu is not None else None
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        started = time()
        pending = pool.submit(func, *args, **kwargs)
        samples: list[tuple[float, float | None]] = []
        while not pending.done():
            sleep(interval_sec)
            samples.append(read())
        result = pending.result()
        elapsed = time() - started

    readings = samples or [read()]
    peak_cpu = max(sample[0] for sample in readings)
    gpu_readings = [sample[1] for sample in readings if sample[1] is not None]
    peak_gpu = max(gpu_readings) if gpu_readings else None
    return result, elapsed, peak_cpu, peak_gpu
