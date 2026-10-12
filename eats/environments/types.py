"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class EnvironmentSpec:
    key: str
    name: str
    means_tk: np.ndarray
    arm_labels: tuple[str, ...] | None = None
    metadata: dict[str, Any] | None = None
