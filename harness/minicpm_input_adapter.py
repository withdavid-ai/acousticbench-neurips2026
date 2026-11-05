from __future__ import annotations

import copy
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from .message_utils import flatten_messages_to_text


class MiniCPMConcatAdapter:
    """Merge paired audio clips for MiniCPM-o's single-audio chat interface."""

    ADAPTER_ID = "minicpm-concat-v1"
    SAMPLE_RATE = 16000
    GAP_S = 1.0

    @classmethod
    def _render_prompt(cls, messages: Any) -> str:
        return flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: f"[Audio {idx + 1}]",
            include_roles=True,
        )

    @classmethod
    def _single_turn_messages(cls, prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if str(prompt).strip():
            content.append({"type": "text", "text": prompt})
        content.append({"type": "audio", "audio_index": 0})
        return [{"role": "user", "content": content}]

    @classmethod
    def _merge_audio_paths(cls, audio_paths: list[str]) -> str:
        if len(audio_paths) <= 1:
            return audio_paths[0]

        import librosa
        import soundfile as sf

        clips = [
            librosa.load(path, sr=cls.SAMPLE_RATE, mono=True)[0]
            for path in audio_paths
        ]
        gap = np.zeros(int(cls.SAMPLE_RATE * cls.GAP_S), dtype=np.float32)
        segments: list[np.ndarray] = []
        for idx, clip in enumerate(clips):
            segments.append(np.asarray(clip, dtype=np.float32))
            if idx < len(clips) - 1:
                segments.append(gap)
        merged = np.concatenate(segments)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = handle.name
        sf.write(output_path, merged, cls.SAMPLE_RATE)
        return output_path

    @staticmethod
    def _normalize_audio_paths(audio: Any) -> list[str]:
        if isinstance(audio, str):
            return [audio] if audio.strip() else []
        if isinstance(audio, list):
            return [path for item in audio if (path := str(item).strip())]
        return []

    def transform_runtime_message(
        self,
        msg: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        adapted = copy.deepcopy(msg)
        audio_paths = self._normalize_audio_paths(adapted.get("audio"))
        if len(audio_paths) <= 1:
            return adapted, []

        prompt = self._render_prompt(adapted.get("messages"))
        merged_path = self._merge_audio_paths(audio_paths)
        adapted["audio"] = [merged_path]
        adapted["messages"] = self._single_turn_messages(prompt)
        adapted["text"] = prompt

        meta = dict(adapted.get("meta") or {})
        meta["_input_adapter_audio_merge"] = {
            "strategy": "concat-silence-gap",
            "sample_rate": self.SAMPLE_RATE,
            "gap_s": self.GAP_S,
            "source_clip_count": len(audio_paths),
        }
        adapted["meta"] = meta
        return adapted, [merged_path]


@contextmanager
def minicpm_runtime_adapter_context(
    msg: dict[str, Any],
    adapters: tuple[MiniCPMConcatAdapter, ...] | list[MiniCPMConcatAdapter],
) -> Iterator[dict[str, Any]]:
    adapted = copy.deepcopy(msg)
    cleanup_paths: list[str] = []
    try:
        for adapter in adapters:
            adapted, adapter_cleanup = adapter.transform_runtime_message(adapted)
            meta = adapted.setdefault("meta", {})
            applied = meta.get("_input_adapters")
            if not isinstance(applied, list):
                applied = []
            if adapter.ADAPTER_ID not in applied:
                applied.append(adapter.ADAPTER_ID)
            meta["_input_adapters"] = applied
            cleanup_paths.extend(adapter_cleanup)
        yield adapted
    finally:
        for path in cleanup_paths:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
