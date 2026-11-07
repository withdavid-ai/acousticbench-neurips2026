"""Base class for models served via Docker containers with FastAPI endpoints.

Each open-source model is packaged as a Docker image exposing a FastAPI
server on port 8000. The adapter here is a thin HTTP client: it base64-encodes
the audio, POSTs to the container's /generate endpoint, and returns the
response.

Subclasses define NAME and optionally override DOCKER_ENDPOINT or
REQUEST_TIMEOUT_S. The endpoint URL is resolved from:
  1. Constructor kwarg ``endpoint``
  2. Environment variable ``ACOUSTICBENCH_{NAME_UPPER}_ENDPOINT``
     (where NAME_UPPER is NAME with hyphens/dots replaced by underscores)
  3. Class-level DOCKER_ENDPOINT default

All model-specific prompt formatting and inference logic lives server-side
in the Docker container's app.py.
"""

from __future__ import annotations

import base64
import os
import time

from loguru import logger

from .base import BaseModel
from ..message_utils import ensure_messages, last_user_text


class DockerRemoteModel(BaseModel):

    NAME = None
    DOCKER_ENDPOINT = None
    GENERATE_PATH = "/generate"
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2
    RETRY_BACKOFF_S = 3.0

    def __init__(self, **kwargs):
        endpoint = kwargs.pop("endpoint", None)
        super().__init__(**kwargs)

        if endpoint is None:
            env_key = self._endpoint_env_key()
            endpoint = os.environ.get(env_key)
        if endpoint is None:
            endpoint = self.DOCKER_ENDPOINT
        if endpoint is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an endpoint URL. "
                f"Set {self._endpoint_env_key()} or pass endpoint= to constructor."
            )
        self._endpoint = endpoint.rstrip("/")

    def _endpoint_env_key(self) -> str:
        name = self.NAME or self.__class__.__name__
        safe = name.upper().replace("-", "_").replace(".", "_")
        return f"ACOUSTICBENCH_{safe}_ENDPOINT"

    @staticmethod
    def _encode_audio(audio_path: str) -> tuple[str, str]:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        ext = audio_path.rsplit(".", 1)[-1].lower()
        fmt = {"mp3": "mp3", "flac": "flac", "ogg": "ogg"}.get(ext, "wav")
        return audio_b64, fmt

    @classmethod
    def _encode_audios(cls, audio_paths: str | list[str]) -> list[dict[str, str]]:
        return [
            {"data": audio_b64, "format": audio_fmt}
            for audio_b64, audio_fmt in (
                cls._encode_audio(audio_path)
                for audio_path in cls.normalize_audio_paths(audio_paths)
            )
        ]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        import requests

        if isinstance(exc, requests.exceptions.ConnectionError):
            return True
        if isinstance(exc, requests.exceptions.Timeout):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            return exc.response is not None and exc.response.status_code >= 500
        return False

    def generate_inner(self, msg: dict):
        import requests

        audio_paths = self.normalize_audio_paths(msg["audio"])
        audios = self._encode_audios(audio_paths)
        payload_meta = dict(msg.get("meta", {}))
        payload_meta.update(self.sampling_request_meta())
        messages = ensure_messages(msg, audio_count=len(audio_paths))
        prompt_text = last_user_text(messages)

        payload = {
            "provider": "docker",
            "audio_paths": audio_paths,
            "audio_b64": audios[0]["data"] if audios else "",
            "audio_format": audios[0]["format"] if audios else "",
            "audios": audios,
            "messages": messages,
            "meta": payload_meta,
        }

        url = f"{self._endpoint}{self.GENERATE_PATH}"
        max_attempts = max(1, int(self.MAX_RETRIES) + 1)
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT_S,
                )
                resp.raise_for_status()
                data = resp.json()
                wall_s = time.perf_counter() - t0

                submission_payload = data.get("submission_payload")
                submission_trace = data.get("submission_trace")
                if submission_trace is None:
                    submission_trace = {
                        "transport": "docker-http",
                        "exact_submission_available": True,
                        "builder": f"{self.__class__.__name__}.generate_inner",
                    }
                return {
                    "prompt": data.get("prompt", prompt_text),
                    "response_text": data.get("response"),
                    "request_payload": payload,
                    "response_payload": data,
                    "submission_payload": submission_payload,
                    "submission_trace": submission_trace,
                    "response_status": "ok",
                    "response_error": None,
                    "wall_time_s": float(wall_s),
                }
            except Exception as e:
                last_error = e
                if isinstance(e, ValueError):
                    raise
                if attempt >= max_attempts or not self._is_retryable(e):
                    break
                logger.warning(
                    "Retrying Docker call for {} ({}/{}): {}",
                    self.NAME or self.__class__.__name__,
                    attempt,
                    max_attempts - 1,
                    e,
                )
                time.sleep(self.RETRY_BACKOFF_S)

        raise RuntimeError(
            f"Docker endpoint {url} failed after {max_attempts} attempts: {last_error}"
        )
