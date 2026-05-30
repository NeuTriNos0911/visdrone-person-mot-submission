from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class FpsMeter:
    frames: int = 0
    elapsed: float = 0.0

    def start(self) -> float:
        return perf_counter()

    def stop(self, start_time: float) -> None:
        self.frames += 1
        self.elapsed += perf_counter() - start_time

    @property
    def fps(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.frames / self.elapsed


