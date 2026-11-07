"""Helpers for syncing local artifacts with the shared Modal volume."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_VOLUME_NAME = "acousticbench-kimi-rad-data"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text or "")


def _parse_modal_json_output(raw_output: str) -> Any:
    cleaned = _strip_ansi(raw_output).strip()
    if not cleaned:
        return None
    start_candidates = [index for index in (cleaned.find("["), cleaned.find("{")) if index >= 0]
    if start_candidates:
        start = min(start_candidates)
        cleaned = cleaned[start:]
    end_candidates = [index for index in (cleaned.rfind("]"), cleaned.rfind("}")) if index >= 0]
    if end_candidates:
        end = max(end_candidates)
        cleaned = cleaned[: end + 1]
    return json.loads(cleaned)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "CLICOLOR": "0",
        "CLICOLOR_FORCE": "0",
        "TERM": "dumb",
    }
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        )
    return proc


def volume_put(
    local_path: str | Path,
    remote_parent: str,
    *,
    modal_env: str,
    volume_name: str = DEFAULT_VOLUME_NAME,
    force: bool = True,
) -> str:
    cmd = ["modal", "volume", "put", "--env", modal_env]
    if force:
        cmd.append("--force")
    cmd.extend([volume_name, str(local_path), remote_parent])
    proc = _run(cmd)
    return (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")


def volume_rm(
    remote_path: str,
    *,
    modal_env: str,
    volume_name: str = DEFAULT_VOLUME_NAME,
    recursive: bool = False,
    check: bool = True,
) -> str:
    cmd = ["modal", "volume", "rm", "--env", modal_env]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([volume_name, remote_path])
    proc = _run(cmd, check=check)
    return (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")


def volume_get(
    remote_path: str,
    dest_dir: str | Path,
    *,
    modal_env: str,
    volume_name: str = DEFAULT_VOLUME_NAME,
    check: bool = True,
) -> Path | None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            "modal",
            "volume",
            "get",
            "--env",
            modal_env,
            volume_name,
            remote_path,
            str(dest) + "/",
        ],
        check=check,
    )
    if proc.returncode != 0:
        return None
    output = dest / Path(remote_path.rstrip("/")).name
    if output.exists():
        return output
    return None


def remote_path_exists(
    remote_path: str,
    *,
    modal_env: str,
    volume_name: str = DEFAULT_VOLUME_NAME,
) -> bool:
    with tempfile.TemporaryDirectory(prefix="modal-volume-exists-") as tmp_dir:
        path = volume_get(
            remote_path,
            tmp_dir,
            modal_env=modal_env,
            volume_name=volume_name,
            check=False,
        )
        return path is not None


def volume_ls(
    remote_path: str,
    *,
    modal_env: str,
    volume_name: str = DEFAULT_VOLUME_NAME,
) -> list[dict[str, Any]]:
    proc = _run(
        [
            "modal",
            "volume",
            "ls",
            "--env",
            modal_env,
            "--json",
            volume_name,
            remote_path,
        ]
    )
    payload = _parse_modal_json_output(proc.stdout or "[]")
    if not isinstance(payload, list):
        raise TypeError(f"Expected list from modal volume ls, got {type(payload).__name__}")
    return payload


def append_remote_jsonl(
    remote_path: str,
    row: dict[str, Any],
    *,
    modal_env: str,
    volume_name: str = DEFAULT_VOLUME_NAME,
) -> str:
    with tempfile.TemporaryDirectory(prefix="modal-append-jsonl-") as tmp_dir:
        local_dir = Path(tmp_dir)
        local_file = local_dir / Path(remote_path).name
        existing = volume_get(
            remote_path,
            local_dir,
            modal_env=modal_env,
            volume_name=volume_name,
            check=False,
        )
        if existing is not None and existing.is_file():
            local_file = existing
        with local_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        return volume_put(
            local_file,
            remote_path,
            modal_env=modal_env,
            volume_name=volume_name,
            force=True,
        )


def read_remote_json(
    remote_path: str,
    *,
    modal_env: str,
    volume_name: str = DEFAULT_VOLUME_NAME,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="modal-read-json-") as tmp_dir:
        local_path = volume_get(
            remote_path,
            tmp_dir,
            modal_env=modal_env,
            volume_name=volume_name,
            check=True,
        )
        if local_path is None:
            raise FileNotFoundError(remote_path)
        return json.loads(local_path.read_text(encoding="utf-8"))


def read_remote_jsonl(
    remote_path: str,
    *,
    modal_env: str,
    volume_name: str = DEFAULT_VOLUME_NAME,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="modal-read-jsonl-") as tmp_dir:
        local_path = volume_get(
            remote_path,
            tmp_dir,
            modal_env=modal_env,
            volume_name=volume_name,
            check=True,
        )
        if local_path is None:
            raise FileNotFoundError(remote_path)
        rows: list[dict[str, Any]] = []
        with local_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
