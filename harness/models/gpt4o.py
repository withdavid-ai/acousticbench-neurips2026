import base64
import os

from loguru import logger

from .base import BaseModel
from ..message_utils import (
    ensure_messages,
    flatten_messages_to_text,
    last_user_text,
    render_messages,
)


class GPT4oAudio(BaseModel):
    """GPT-4o audio-input adapter via the OpenAI Chat Completions API.

    Requires: pip install openai
    Environment: OPENAI_API_KEY
    """

    NAME = 'GPT-4o'
    # Dated snapshot from paper Table 5; the bare `gpt-4o-audio-preview` alias
    # can be repointed by OpenAI.
    DEFAULT_MODEL_NAME = 'gpt-4o-audio-preview-2025-06-03'

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, **kwargs):
        from openai import OpenAI
        super().__init__(**kwargs)
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise EnvironmentError('OPENAI_API_KEY not set')
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @staticmethod
    def _audio_format(path: str) -> str:
        ext = path.rsplit('.', 1)[-1].lower()
        return {'mp3': 'mp3', 'flac': 'flac', 'ogg': 'ogg'}.get(ext, 'wav')

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 300) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)

    @staticmethod
    def _silent_audio_b64() -> str:
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b'\x00\x00' * 1600)
        return base64.b64encode(buf.getvalue()).decode()

    def _render_messages(self, msg: dict, audio_paths: list[str]) -> list[dict]:
        messages = ensure_messages(msg, audio_count=len(audio_paths))
        encoded_audio_parts = {}
        for audio_path in audio_paths:
            with open(audio_path, 'rb') as handle:
                encoded_audio_parts[audio_path] = {
                    'data': base64.b64encode(handle.read()).decode(),
                    'format': self._audio_format(audio_path),
                }
        rendered = render_messages(
            messages,
            audio_paths,
            build_text_part=lambda text, _role: (
                {'type': 'text', 'text': text} if str(text).strip() else None
            ),
            build_audio_part=lambda audio_path, _audio_index, _role: {
                'type': 'input_audio',
                'input_audio': encoded_audio_parts[audio_path],
            },
        )
        normalized: list[dict] = []
        for message in rendered:
            role = message['role']
            content = message['content']
            if role != 'user':
                normalized.append({
                    'role': role,
                    'content': flatten_messages_to_text([message], include_roles=False),
                })
            else:
                has_audio = any(
                    isinstance(p, dict) and p.get('type') == 'input_audio'
                    for p in (content if isinstance(content, list) else [])
                )
                if not has_audio:
                    if not isinstance(content, list):
                        content = [{'type': 'text', 'text': content}] if content else []
                    content = [
                        {'type': 'input_audio', 'input_audio': {'data': self._silent_audio_b64(), 'format': 'wav'}},
                        *content,
                    ]
                normalized.append({'role': role, 'content': content})
        return normalized

    def generate_inner(self, msg: dict):
        audio_paths = self.normalize_audio_paths(msg['audio'])
        prompt = last_user_text(ensure_messages(msg, audio_count=len(audio_paths)))
        messages = self._render_messages(msg, audio_paths)
        request_payload = {
            'provider': 'openai',
            'model_name': self.model_name,
            'audio_paths': audio_paths,
            'prompt': prompt,
            'messages': messages,
            'sampling': {
                'mode': self.sampling_mode,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'seed': self.seed,
            },
        }

        try:
            request_kwargs = {
                'model': self.model_name,
                'messages': messages,
            }
            if self.benchmark_sampling_enabled():
                if self.temperature is not None:
                    request_kwargs['temperature'] = self.temperature
            else:
                request_kwargs['temperature'] = 0.0
            if self.top_p is not None:
                request_kwargs['top_p'] = self.top_p
            if self.seed is not None:
                request_kwargs['seed'] = self.seed
            request_payload['request_kwargs'] = {
                key: value
                for key, value in request_kwargs.items()
                if key != 'messages'
            }
            resp = self.client.chat.completions.create(**request_kwargs)
            response_payload = (
                resp.model_dump()
                if hasattr(resp, 'model_dump')
                else {
                    'id': getattr(resp, 'id', None),
                    'model': getattr(resp, 'model', None),
                }
            )
            response_text = None
            if resp.choices:
                response_text = resp.choices[0].message.content
            return {
                'prompt': prompt,
                'response_text': response_text,
                'request_payload': request_payload,
                'response_payload': response_payload,
                'submission_payload': {
                    'provider': 'openai',
                    'transport': 'chat-completions',
                    'model_name': self.model_name,
                    'messages': messages,
                    'request_kwargs': request_payload.get('request_kwargs'),
                },
                'submission_trace': {
                    'exact_submission_available': True,
                    'builder': f'{self.__class__.__name__}.generate_inner',
                },
                'response_status': 'ok',
                'response_error': None,
            }
        except Exception as e:
            logger.error(f'GPT-4o API error: {e}')
            return {
                'prompt': prompt,
                'response_text': None,
                'request_payload': request_payload,
                'response_payload': None,
                'submission_payload': {
                    'provider': 'openai',
                    'transport': 'chat-completions',
                    'model_name': self.model_name,
                    'messages': messages,
                    'request_kwargs': request_payload.get('request_kwargs'),
                },
                'submission_trace': {
                    'exact_submission_available': False,
                    'builder': f'{self.__class__.__name__}.generate_inner',
                    'error': str(e),
                },
                'response_status': 'error',
                'response_error': str(e),
            }
