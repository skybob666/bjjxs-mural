"""把用户明确要求保存的API配置写入项目本机 .env。"""

from __future__ import annotations

import os
from pathlib import Path


def default_env_path() -> Path:
    return Path(__file__).resolve().parent / ".env"


def save_env_values(values: dict[str, str], path: Path | None = None) -> Path:
    """更新指定键并保留其他配置；不在输出中打印值。"""
    target = path or default_env_path()
    existing = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
    updated: list[str] = []
    handled: set[str] = set()
    for line in existing:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped and not stripped.startswith("#") else ""
        if key in values:
            updated.append(f"{key}={values[key]}")
            handled.add(key)
        else:
            updated.append(line)
    for key, value in values.items():
        if key not in handled:
            updated.append(f"{key}={value}")
    target.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
    for key, value in values.items():
        os.environ[key] = value
    return target


def remove_env_values(keys: list[str], path: Path | None = None) -> Path:
    target = path or default_env_path()
    if target.exists():
        existing = target.read_text(encoding="utf-8").splitlines()
        kept = []
        for line in existing:
            stripped = line.strip()
            key = stripped.split("=", 1)[0].strip() if "=" in stripped and not stripped.startswith("#") else ""
            if key not in keys:
                kept.append(line)
        target.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
    for key in keys:
        os.environ.pop(key, None)
    return target
