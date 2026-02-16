from __future__ import annotations


def fmt_time_label(value: str | None) -> str:
    if not value:
        return "n/a"
    return str(value)


__all__ = ["fmt_time_label"]
