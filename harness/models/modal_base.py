"""Base class for models served via Modal serverless GPU endpoints.

Each open-source model is deployed as a Modal cls. The adapter here
is a thin RPC client: it base64-encodes the audio, calls the remote Modal
class method via modal.Cls.from_name(), and returns the response.

Subclasses define MODAL_APP_NAME and MODAL_FUNCTION_NAME (in the form
"ClassName.method_name"). All model-specific prompt formatting and
inference logic lives server-side in the Modal app.

Deployed apps are looked up in the Modal environment named by the
``MODAL_ENVIRONMENT`` variable (``run.py --modal-environment`` sets it). When
it is unset, Modal's own default environment for the active profile is used.
"""

import base64
import os

from loguru import logger

from .base import BaseModel
from ..message_utils import ensure_messages, last_user_text

class ModalRemoteModel(BaseModel):

    NAME = None
    MODAL_APP_NAME = None
    MODAL_FUNCTION_NAME = None
    MODAL_INSTANCE_MODE = "cached"
    MODAL_MAX_RETRIES = 0
    MODAL_RETRY_BACKOFF_S = 3.0
    MODAL_CALL_TIMEOUT_S = None

    def __init__(self, **kwargs):
        import modal
        super().__init__(**kwargs)
        if self.MODAL_APP_NAME is None or self.MODAL_FUNCTION_NAME is None:
            raise ValueError(
                f'{self.__class__.__name__} must define MODAL_APP_NAME and MODAL_FUNCTION_NAME')

        class_name, method_name = self.MODAL_FUNCTION_NAME.split('.', 1)
        environment_name = os.environ.get("MODAL_ENVIRONMENT") or None
        self._remote_cls = modal.Cls.from_name(
            self.MODAL_APP_NAME,
            class_name,
            environment_name=environment_name,
        )
        self._method_name = method_name
        self._instance_mode = str(
            getattr(self, "MODAL_INSTANCE_MODE", "cached") or "cached"
        ).strip().lower()
        if self._instance_mode not in {"cached", "per_call"}:
            raise ValueError(
                f"Unsupported MODAL_INSTANCE_MODE for {self.__class__.__name__}: "
                f"{self._instance_mode}"
            )
        self._refresh_remote_method()

    def remote_class_kwargs(self) -> dict:
        return {}

    def _create_remote_method(self):
        instance = self._remote_cls(**self.remote_class_kwargs())
        return instance, getattr(instance, self._method_name)

    def _refresh_remote_method(self):
        self._instance, self._method = self._create_remote_method()

    def _remote_method_for_attempt(self):
        if self._instance_mode == "per_call":
            _, method = self._create_remote_method()
            return method
        return self._method

    @staticmethod
    def _is_retryable_modal_error(exc: Exception) -> bool:
        message = str(exc).lower()
        retryable_markers = (
            "function call has expired",
            "deadline exceeded",
            "request timed out",
            "timed out",
            "connection reset",
        )
        return any(marker in message for marker in retryable_markers)

    @staticmethod
    def _encode_audio(audio_path: str) -> tuple[str, str]:
        with open(audio_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        ext = audio_path.rsplit('.', 1)[-1].lower()
        fmt = {'mp3': 'mp3', 'flac': 'flac', 'ogg': 'ogg'}.get(ext, 'wav')
        return audio_b64, fmt

    @classmethod
    def _encode_audios(cls, audio_paths: str | list[str]) -> list[dict[str, str]]:
        return [
            {'data': audio_b64, 'format': audio_fmt}
            for audio_b64, audio_fmt in (
                cls._encode_audio(audio_path)
                for audio_path in cls.normalize_audio_paths(audio_paths)
            )
        ]

    def generate_inner(self, msg: dict):
        import time

        audio_paths = self.normalize_audio_paths(msg['audio'])
        audios = self._encode_audios(audio_paths)
        payload_meta = dict(msg.get('meta', {}))
        payload_meta.update(self.sampling_request_meta())
        messages = ensure_messages(msg, audio_count=len(audio_paths))
        prompt_text = last_user_text(messages)

        payload = {
            'provider': 'modal',
            'modal_app_name': self.MODAL_APP_NAME,
            'modal_function_name': self.MODAL_FUNCTION_NAME,
            'audio_paths': audio_paths,
            'audio_b64': audios[0]['data'] if audios else '',
            'audio_format': audios[0]['format'] if audios else '',
            'audios': audios,
            'messages': messages,
            'meta': payload_meta,
        }

        max_attempts = max(1, int(self.MODAL_MAX_RETRIES) + 1)
        call_timeout_s = getattr(self, "MODAL_CALL_TIMEOUT_S", None)
        last_error = None
        for attempt in range(1, max_attempts + 1):
            function_call = None
            try:
                t0 = time.perf_counter()
                method = self._remote_method_for_attempt()
                if call_timeout_s is None:
                    data = method.remote(item=payload)
                else:
                    function_call = method.spawn(item=payload)
                    data = function_call.get(timeout=float(call_timeout_s))
                wall_s = time.perf_counter() - t0
                submission_payload = data.get('submission_payload')
                submission_trace = data.get('submission_trace')
                if submission_trace is None:
                    submission_trace = {
                        'transport': 'modal',
                        'exact_submission_available': True,
                        'builder': f'{self.MODAL_APP_NAME}/{self.MODAL_FUNCTION_NAME}',
                    }
                return {
                    'prompt': data.get('prompt', prompt_text),
                    'response_text': data.get('response'),
                    'request_payload': payload,
                    'response_payload': data,
                    'submission_payload': submission_payload,
                    'submission_trace': submission_trace,
                    'response_status': 'ok',
                    'response_error': None,
                    'wall_time_s': float(wall_s),
                }
            except Exception as e:
                if function_call is not None:
                    try:
                        function_call.cancel()
                    except Exception:
                        pass
                last_error = e
                if isinstance(e, ValueError):
                    raise
                if attempt >= max_attempts or not self._is_retryable_modal_error(e):
                    break
                logger.warning(
                    "Retrying expired Modal call for {} ({}/{}): {}",
                    self.NAME or self.__class__.__name__,
                    attempt,
                    max_attempts - 1,
                    e,
                )
                if self._instance_mode == "cached":
                    self._refresh_remote_method()
                time.sleep(self.MODAL_RETRY_BACKOFF_S)

        raise RuntimeError(
            f'Modal function {self.MODAL_APP_NAME}/{self.MODAL_FUNCTION_NAME} '
            f'failed: {last_error}')
