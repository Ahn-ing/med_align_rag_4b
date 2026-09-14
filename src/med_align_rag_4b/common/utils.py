"""Small cross-subsystem helpers that do not warrant a dedicated module."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in a stable ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def require_non_empty_string(value: object, field_name: str) -> str:
    """Return a stripped string or raise an actionable validation error."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
