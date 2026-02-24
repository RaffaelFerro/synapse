from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def isoformat_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_dt(value: str) -> datetime:
    # Aceita ISO8601 com 'Z' ou offset.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def json_loads(text: str) -> Any:
    return json.loads(text)


def get_lock_owner() -> str:
    return f"pid:{os.getpid()}:{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class Timer:
    start: datetime

    @classmethod
    def start_new(cls) -> "Timer":
        return cls(start=utc_now())

    def ms(self) -> float:
        return (utc_now() - self.start).total_seconds() * 1000.0

