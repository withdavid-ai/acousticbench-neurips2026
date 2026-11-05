from __future__ import annotations

from typing import Any, Callable


def normalize_content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        text = str(content).strip()
        return [{"type": "text", "text": text}] if text else []
    if not isinstance(content, list):
        return []

    normalized: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            text = str(part).strip()
            if text:
                normalized.append({"type": "text", "text": text})
            continue
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "").strip()
        if part_type == "audio":
            normalized.append({"type": "audio", "audio_index": int(part.get("audio_index", 0))})
            continue
        if part_type == "audio_file":
            normalized.append({"type": "audio", "audio_index": int(part.get("audio_index", 0))})
            continue
        text = str(part.get("text") or "").strip()
        if text:
            normalized.append({"type": "text", "text": text})
    return normalized


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return normalized
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip() or "user"
        normalized.append({"role": role, "content": normalize_content_parts(message.get("content"))})
    return normalized


def legacy_messages(prompt: str, audio_count: int) -> list[dict[str, Any]]:
    content = [{"type": "audio", "audio_index": idx} for idx in range(max(0, int(audio_count)))]
    text = str(prompt or "").strip()
    if text:
        content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def ensure_messages(msg: dict, *, audio_count: int) -> list[dict[str, Any]]:
    normalized = normalize_messages(msg.get("messages"))
    if normalized:
        return normalized
    return legacy_messages(str(msg.get("text", "") or ""), audio_count)


def flatten_messages_to_text(
    messages: Any,
    *,
    audio_placeholder: str | Callable[[int], str] = "[audio]",
    include_roles: bool = False,
) -> str:
    rendered_messages: list[str] = []
    for message in normalize_messages(messages):
        rendered_parts: list[str] = []
        for part in message["content"]:
            if part.get("type") == "audio":
                placeholder = (
                    audio_placeholder(int(part["audio_index"]))
                    if callable(audio_placeholder)
                    else str(audio_placeholder)
                )
                placeholder = str(placeholder or "").strip()
                if placeholder:
                    rendered_parts.append(placeholder)
                continue
            text = str(part.get("text") or "").strip()
            if text:
                rendered_parts.append(text)
        rendered = "\n".join(rendered_parts).strip()
        if not rendered:
            continue
        if include_roles:
            rendered_messages.append(f"{message['role'].upper()}:\n{rendered}")
        else:
            rendered_messages.append(rendered)
    return "\n\n".join(rendered_messages).strip()


def last_user_text(messages: Any, *, audio_placeholder: str | Callable[[int], str] = "[audio]") -> str:
    normalized = normalize_messages(messages)
    for message in reversed(normalized):
        if message["role"] != "user":
            continue
        return flatten_messages_to_text(
            [message],
            audio_placeholder=audio_placeholder,
            include_roles=False,
        )
    return ""


def render_messages(
    messages: Any,
    audio_values: list[Any],
    *,
    build_text_part: Callable[[str, str], Any],
    build_audio_part: Callable[[Any, int, str], Any],
    drop_empty_messages: bool = True,
) -> list[dict[str, Any]]:
    rendered_messages: list[dict[str, Any]] = []
    for message in normalize_messages(messages):
        rendered_content: list[Any] = []
        role = message["role"]
        for part in message["content"]:
            if part.get("type") == "audio":
                audio_index = int(part["audio_index"])
                built = build_audio_part(audio_values[audio_index], audio_index, role)
            else:
                built = build_text_part(str(part.get("text") or ""), role)
            if built is None:
                continue
            if isinstance(built, list):
                rendered_content.extend(item for item in built if item is not None)
            else:
                rendered_content.append(built)
        if drop_empty_messages and not rendered_content:
            continue
        rendered_messages.append({"role": role, "content": rendered_content})
    return rendered_messages
