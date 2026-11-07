import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from .base import BaseModel
from ..message_utils import ensure_messages, flatten_messages_to_text, last_user_text

_MULTIPLE_CHOICE_RE = re.compile(r"^\s*[\(\[]?([A-Z])[\)\].:]\s+.+$", re.MULTILINE)



class OpenAICascadeTranscribeGPT54(BaseModel):
    """OpenAI-only cascade baseline: ASR first, then text-only reasoning."""

    NAME = "Cascade-GPT4oTranscribe-GPT5.4"
    # OpenAI publishes no dated snapshot for gpt-4o-transcribe, so this alias
    # is the most specific identifier available (paper Table 5).
    DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-transcribe"
    # Dated snapshot from paper Table 5.
    DEFAULT_ANSWER_MODEL = "gpt-5.4-2026-03-05"

    def __init__(
        self,
        transcribe_model_name: str = DEFAULT_TRANSCRIBE_MODEL,
        answer_model_name: str = DEFAULT_ANSWER_MODEL,
        **kwargs,
    ):
        from openai import OpenAI

        super().__init__(**kwargs)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.transcribe_model_name = transcribe_model_name
        self.answer_model_name = answer_model_name

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 600) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)

    @staticmethod
    def _choice_labels(question: str) -> set[str]:
        return {match.group(1) for match in _MULTIPLE_CHOICE_RE.finditer(question or "")}

    @classmethod
    def _looks_multiple_choice(cls, question: str) -> bool:
        labels = cls._choice_labels(question)
        return len(labels) >= 2

    @staticmethod
    def _is_new_api_model(model_name: str) -> bool:
        lowered = str(model_name or "").lower()
        return any(tag in lowered for tag in ("gpt-5", "o1", "o3", "o4"))

    def _transcribe_audio(self, audio_path: str) -> tuple[str, dict[str, Any]]:
        request_meta = {
            "model": self.transcribe_model_name,
            "filename": Path(audio_path).name,
        }
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.transcribe_model_name,
                file=audio_file,
            )

        if isinstance(response, str):
            transcript = response.strip()
            response_payload = {"text": transcript}
        else:
            transcript = str(getattr(response, "text", "") or "").strip()
            if hasattr(response, "model_dump"):
                response_payload = response.model_dump()
            else:
                response_payload = {"text": transcript}
        return transcript, {
            "request": request_meta,
            "response": response_payload,
        }

    def _build_chat_request(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.answer_model_name,
            "messages": messages,
        }
        if self._is_new_api_model(self.answer_model_name):
            request_kwargs["max_completion_tokens"] = 200
        else:
            request_kwargs["max_tokens"] = 200
            if self.benchmark_sampling_enabled():
                if self.temperature is not None:
                    request_kwargs["temperature"] = self.temperature
            else:
                request_kwargs["temperature"] = 0.0
            if self.top_p is not None:
                request_kwargs["top_p"] = self.top_p
            if self.seed is not None:
                request_kwargs["seed"] = self.seed
        return request_kwargs

    @staticmethod
    def _extract_structured_choice(raw_text: str, valid_labels: set[str]) -> str | None:
        if not raw_text:
            return None

        raw = str(raw_text).strip()
        candidates = [raw]
        if raw.startswith("```"):
            stripped = raw.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
            candidates.append(stripped)
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidates.append(raw[start : end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            for key in ("selected_label", "label", "answer", "choice"):
                value = payload.get(key)
                if not isinstance(value, str):
                    continue
                label = value.strip().upper()
                if label in valid_labels:
                    return label

        cleaned = raw.upper().strip()
        if cleaned in valid_labels:
            return cleaned
        return None

    def _build_reasoning_prompts(
        self,
        transcript_prompt: str,
        *,
        use_structured_choice: bool,
        valid_labels: set[str],
    ) -> tuple[str, str]:
        if use_structured_choice:
            labels_text = ", ".join(sorted(valid_labels))
            system_prompt = (
                "You are answering a question about audio using only ASR transcripts. "
                "You cannot hear tone, prosody, timing, emphasis, or speaker affect beyond "
                "what the words imply. Select exactly one option from the provided choices. "
                f'Return exactly one JSON object with key "selected_label" and a value from: {labels_text}.'
            )
        else:
            system_prompt = (
                "You are answering a question about audio using only ASR transcripts. "
                "You cannot hear tone, prosody, timing, emphasis, or speaker affect beyond "
                "what the words imply. Answer based only on the transcript."
            )
        return system_prompt, transcript_prompt

    def generate_inner(self, msg: dict):
        audio_paths = self.normalize_audio_paths(msg["audio"])
        messages = ensure_messages(msg, audio_count=len(audio_paths))
        prompt = last_user_text(messages)
        request_payload: dict[str, Any] = {
            "provider": "openai",
            "transcribe_model_name": self.transcribe_model_name,
            "answer_model_name": self.answer_model_name,
            "audio_paths": audio_paths,
            "messages": messages,
            "sampling": {
                "mode": self.sampling_mode,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            },
        }
        response_payload: dict[str, Any] = {
            "transcriptions": [],
        }

        try:
            transcripts = []
            for audio_path in audio_paths:
                transcript_text, transcript_payload = self._transcribe_audio(audio_path)
                transcripts.append(transcript_text)
                response_payload["transcriptions"].append(transcript_payload)

            transcript_prompt = flatten_messages_to_text(
                messages,
                audio_placeholder=lambda idx: transcripts[idx],
                include_roles=True,
            )
            valid_labels = self._choice_labels(transcript_prompt)
            use_structured_choice = self._looks_multiple_choice(transcript_prompt)
            system_prompt, user_prompt = self._build_reasoning_prompts(
                transcript_prompt,
                use_structured_choice=use_structured_choice,
                valid_labels=valid_labels,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            request_kwargs = self._build_chat_request(messages)
            if use_structured_choice:
                request_kwargs["response_format"] = {"type": "json_object"}
            request_payload["system_prompt"] = system_prompt
            request_payload["prompt"] = user_prompt
            request_payload["answer_request_kwargs"] = {
                key: value
                for key, value in request_kwargs.items()
                if key != "messages"
            }

            answer_response = self.client.chat.completions.create(**request_kwargs)
            if hasattr(answer_response, "model_dump"):
                response_payload["answer"] = answer_response.model_dump()
            else:
                response_payload["answer"] = {
                    "id": getattr(answer_response, "id", None),
                    "model": getattr(answer_response, "model", None),
                }
            raw_answer_text = None
            if getattr(answer_response, "choices", None):
                raw_answer_text = answer_response.choices[0].message.content
            response_payload["raw_answer_text"] = raw_answer_text
            parsed_selected_label = None
            if use_structured_choice:
                parsed_selected_label = self._extract_structured_choice(
                    raw_answer_text or "",
                    valid_labels,
                )
            response_payload["parsed_selected_label"] = parsed_selected_label
            answer_text = parsed_selected_label or raw_answer_text

            return {
                "prompt": f"[cascade-openai] {system_prompt}\n{user_prompt}",
                "response_text": answer_text,
                "request_payload": request_payload,
                "response_payload": response_payload,
                "submission_payload": {
                    "provider": "openai",
                    "transport": "cascade-transcribe-then-chat",
                    "transcribe_model_name": self.transcribe_model_name,
                    "answer_model_name": self.answer_model_name,
                    "transcriptions": list(response_payload.get("transcriptions") or []),
                    "answer_messages": messages,
                    "answer_request_kwargs": request_payload.get("answer_request_kwargs"),
                },
                "submission_trace": {
                    "exact_submission_available": True,
                    "builder": f"{self.__class__.__name__}.generate_inner",
                },
                "response_status": "ok",
                "response_error": None,
            }
        except Exception as exc:
            logger.error(f"OpenAI cascade error: {exc}")
            return {
                "prompt": prompt,
                "response_text": None,
                "request_payload": request_payload,
                "response_payload": response_payload,
                "submission_payload": {
                    "provider": "openai",
                    "transport": "cascade-transcribe-then-chat",
                    "transcribe_model_name": self.transcribe_model_name,
                    "answer_model_name": self.answer_model_name,
                    "transcriptions": list(response_payload.get("transcriptions") or []),
                    "answer_messages": messages if "messages" in locals() else None,
                    "answer_request_kwargs": request_payload.get("answer_request_kwargs"),
                },
                "submission_trace": {
                    "exact_submission_available": False,
                    "builder": f"{self.__class__.__name__}.generate_inner",
                    "error": str(exc),
                },
                "response_status": "error",
                "response_error": str(exc),
            }
