import os
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Never

from loguru import logger

from ..message_utils import last_user_text

try:
    import librosa
except ImportError:
    librosa = None

try:
    import torch
except ImportError:
    torch = None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


class BaseModel:

    NAME = None

    def __init__(self, **kwargs):
        require_submission_payload = kwargs.pop('require_submission_payload', None)
        if require_submission_payload is None:
            require_submission_payload = os.environ.get(
                'AUDIOBENCH_REQUIRE_SUBMISSION_PAYLOAD',
                '1',
            )
        self.require_submission_payload = str(require_submission_payload).strip().lower() not in {
            '',
            '0',
            'false',
            'no',
            'off',
        }

        sampling_mode = kwargs.pop('sampling_mode', None)
        if sampling_mode is None:
            sampling_mode = os.environ.get('AUDIOBENCH_SAMPLING_MODE', 'adapter-default')
        self.sampling_mode = str(sampling_mode or 'adapter-default')
        kwargs.pop('input_adapters', None)

        temperature = kwargs.pop('temperature', None)
        if temperature is None:
            temperature = os.environ.get('AUDIOBENCH_TEMPERATURE')
        self.temperature = _optional_float(temperature)

        top_p = kwargs.pop('top_p', None)
        if top_p is None:
            top_p = os.environ.get('AUDIOBENCH_TOP_P')
        self.top_p = _optional_float(top_p)

        seed = kwargs.pop('seed', None)
        if seed is None:
            seed = os.environ.get('AUDIOBENCH_SEED')
        self.seed = _optional_int(seed)

    @abstractmethod
    def generate_inner(self, msg: dict) -> tuple[str, str] | dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def check_audio_legal(audio_path: str | list[str], max_duration: float = 60) -> bool:
        if librosa is None:
            return True
        if isinstance(audio_path, str):
            duration = librosa.get_duration(path=audio_path)
            if duration > max_duration or duration < 0.1:
                return False
        else:
            for path in audio_path:
                duration = librosa.get_duration(path=path)
                if duration > max_duration or duration < 0.1:
                    return False
        return True

    @staticmethod
    def normalize_audio_paths(audio_path: str | list[str]) -> list[str]:
        if isinstance(audio_path, str):
            return [audio_path]
        return list(audio_path)

    @staticmethod
    def clip_label(index: int) -> str:
        if 0 <= index < 26:
            return chr(ord('A') + index)
        return str(index + 1)

    @staticmethod
    def fallback_prompt_text(msg: dict) -> str:
        text = str(msg.get('text') or '').strip()
        if text:
            return text
        return last_user_text(msg.get('messages'))

    @staticmethod
    def normalize_generation_result(
        result: tuple[str, str] | dict[str, Any],
        fallback_prompt: str,
    ) -> dict[str, Any]:
        if isinstance(result, tuple):
            prompt, response = result
            return {
                'prompt': prompt,
                'response_text': response,
                'request_payload': None,
                'response_payload': None,
                'response_status': 'ok',
                'response_error': None,
            }

        if isinstance(result, dict):
            out = {
                'prompt': str(result.get('prompt', fallback_prompt) or fallback_prompt),
                'response_text': result.get('response_text', result.get('response')),
                'request_payload': result.get('request_payload'),
                'response_payload': result.get('response_payload'),
                'submission_payload': result.get('submission_payload'),
                'submission_trace': result.get('submission_trace'),
                'response_status': str(result.get('response_status', 'ok')),
                'response_error': (
                    str(result['response_error'])
                    if result.get('response_error') is not None
                    else None
                ),
            }
            wt = result.get('wall_time_s')
            if wt is not None:
                try:
                    out['wall_time_s'] = float(wt)
                except (TypeError, ValueError):
                    pass
            return out

        raise TypeError(f'Unsupported model result type: {type(result)!r}')

    def benchmark_sampling_enabled(self) -> bool:
        return self.sampling_mode == 'benchmark-default'

    def sampling_request_meta(self) -> dict[str, Any]:
        meta = {'_sampling_mode': self.sampling_mode}
        if self.temperature is not None:
            meta['_benchmark_temperature'] = self.temperature
        if self.top_p is not None:
            meta['_benchmark_top_p'] = self.top_p
        if self.seed is not None:
            meta['_benchmark_seed'] = self.seed
        return meta

    def _ensure_submission_payload(self, normalized: dict[str, Any]) -> None:
        if not self.require_submission_payload:
            return
        if str(normalized.get('response_status') or '') == 'skipped':
            return

        submission_payload = normalized.get('submission_payload')
        if submission_payload is None:
            raise ValueError(
                f'{self.NAME or self.__class__.__name__} must return submission_payload '
                f'for non-skipped model results'
            )

        if normalized.get('submission_trace') is None:
            normalized['submission_trace'] = {
                'exact_submission_available': True,
                'builder': f'{self.__class__.__name__}.generate_inner',
            }

    def __call__(self, msg: dict) -> dict[str, Any]:
        ctx = torch.inference_mode() if torch is not None else nullcontext()
        with ctx:
            fallback_prompt = self.fallback_prompt_text(msg)
            if not self.check_audio_legal(msg['audio']):
                dataset_name = str(msg.get('meta', {}).get('dataset_name') or 'unknown')
                logger.warning(
                    f'dataset: {dataset_name}, audio: {msg["audio"]}, audio duration out of bounds, skipping this sample')
                return {
                    'prompt': fallback_prompt,
                    'response_text': None,
                    'request_payload': None,
                    'response_payload': None,
                    'response_status': 'skipped',
                    'response_error': 'audio_duration_out_of_bounds',
                }
            result = self.generate_inner(msg)
            normalized = self.normalize_generation_result(result, fallback_prompt)
            self._ensure_submission_payload(normalized)
            return normalized
