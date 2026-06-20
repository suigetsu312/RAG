from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationResult:
    content: str
    metadata: dict[str, Any]