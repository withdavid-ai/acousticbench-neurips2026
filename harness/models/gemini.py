import os

from loguru import logger

from .base import BaseModel
from ..message_utils import (
    ensure_messages,
    flatten_messages_to_text,
    last_user_text,
    render_messages,
)

# Gemini model IDs exactly as listed in paper Table 5. Google publishes no
# dated snapshots for these models, so the IDs below are the most specific
# identifiers available. The two `-preview` IDs can be retired or repointed by
# Google at short notice; if a call fails with a model-not-found error, the
# model has been withdrawn and the reference trajectories can no longer be
# regenerated against it.
GEMINI_MODEL_IDS = {
    'Gemini-2.5-Flash-Audio': 'gemini-2.5-flash',
    'Gemini-2.5-Pro': 'gemini-2.5-pro',
    'Gemini-3-Flash': 'gemini-3-flash-preview',
    'Gemini-3.1-Pro': 'gemini-3.1-pro-preview',
}


def _generation_config_supports_seed(genai) -> bool:
    """Whether the installed SDK's GenerationConfig proto has a `seed` field.

    google-generativeai 0.8.x (google-ai-generativelanguage 0.6.x) rejects an
    unknown `seed` key, which would turn every request into an error row.
    """
    try:
        return 'seed' in genai.protos.GenerationConfig.meta.fields
    except Exception:
        return False


class GeminiAudioBase(BaseModel):
    """Base adapter for Google Gemini models with audio input.

    Uses base64 inline data (inlineData) instead of file uploads for
    simplicity and consistency
     with OpenAI / OpenRouter adapters.

    Requires: pip install google-generativeai
    Environment: GOOGLE_API_KEY or GEMINI_API_KEY
    """

    NAME = None  # subclasses set this

    def __init__(self, model_name: str = 'gemini-2.5-flash', system_instruction: str | None = None, **kwargs):
        import google.generativeai as genai
        super().__init__(**kwargs)
        api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise EnvironmentError('GOOGLE_API_KEY or GEMINI_API_KEY not set')
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        self.model_name = model_name
        self.system_instruction = system_instruction
        self._send_seed = _generation_config_supports_seed(genai)
        if self.seed is not None and not self._send_seed:
            logger.info(
                f'{self.NAME or model_name}: installed google-generativeai SDK has no '
                f'GenerationConfig.seed field; seed {self.seed} is recorded but not sent.'
            )

    @staticmethod
    def _mime_type(path: str) -> str:
        ext = path.rsplit('.', 1)[-1].lower()
        return {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'flac': 'audio/flac',
            'ogg': 'audio/ogg',
            'aac': 'audio/aac',
            'm4a': 'audio/mp4',
        }.get(ext, 'audio/wav')

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 600) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)

    def get_system_instruction(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        if not messages:
            return self.system_instruction, []

        first_message = messages[0]
        remaining_messages = list(messages)
        first_system_instruction = None
        if first_message.get('role') == 'system':
            first_system_instruction = flatten_messages_to_text([first_message], include_roles=False) or None
            remaining_messages = messages[1:]

        for idx, message in enumerate(remaining_messages, start=1 if first_message.get('role') == 'system' else 0):
            if message.get('role') == 'system':
                raise ValueError(
                    f"Gemini adapters only support a system message in the first position; "
                    f"found an additional system message at index {idx}."
                )

        if self.system_instruction and first_system_instruction:
            system_instruction = "\n\n".join(
                part for part in (self.system_instruction, first_system_instruction) if str(part).strip()
            ).strip()
        else:
            system_instruction = first_system_instruction or self.system_instruction
        return system_instruction or None, remaining_messages

    def _get_model(self, system_instruction: str | None):
        if system_instruction == self.system_instruction:
            return self.model
        return self._genai.GenerativeModel(
            self.model_name,
            system_instruction=system_instruction,
        )

    def generate_inner(self, msg: dict):
        audio_paths = self.normalize_audio_paths(msg['audio'])
        messages = ensure_messages(msg, audio_count=len(audio_paths))
        prompt = last_user_text(messages)
        system_instruction, request_messages = self.get_system_instruction(messages)

        try:
            model = self._get_model(system_instruction)
            audio_bytes_by_path = {}
            for audio_path in audio_paths:
                with open(audio_path, 'rb') as handle:
                    audio_bytes_by_path[audio_path] = handle.read()
            rendered_messages = render_messages(
                request_messages,
                audio_paths,
                build_text_part=lambda text, _role: text if str(text).strip() else None,
                build_audio_part=lambda audio_path, _audio_index, _role: {
                    'mime_type': self._mime_type(audio_path),
                    'data': audio_bytes_by_path[audio_path],
                },
            )
            contents = [
                {
                    'role': 'model' if message['role'] == 'assistant' else 'user',
                    'parts': message['content'],
                }
                for message in rendered_messages
            ]
            request_payload = {
                'provider': 'google',
                'model_name': self.model_name,
                'audio_paths': audio_paths,
                'prompt': prompt,
                'system_instruction': system_instruction,
                'messages': messages,
                'sampling': {
                    'mode': self.sampling_mode,
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'seed': self.seed,
                },
            }
            generation_config = {}
            if self.benchmark_sampling_enabled():
                if self.temperature is not None:
                    generation_config['temperature'] = self.temperature
            else:
                generation_config['temperature'] = 0.0
            if self.top_p is not None:
                generation_config['top_p'] = self.top_p
            if self.seed is not None and self._send_seed:
                generation_config['seed'] = self.seed
            request_payload['generation_config'] = generation_config or None
            resp = model.generate_content(
                contents,
                generation_config=generation_config or None,
            )
            response_payload = {'text': getattr(resp, 'text', None)}
            to_dict = getattr(resp, 'to_dict', None)
            if callable(to_dict):
                try:
                    response_payload = to_dict()
                except Exception:
                    pass
            return {
                'prompt': prompt,
                'response_text': getattr(resp, 'text', None),
                'request_payload': request_payload,
                'response_payload': response_payload,
                'submission_payload': {
                    'provider': 'google',
                    'transport': 'gemini-generate-content',
                    'model_name': self.model_name,
                    'system_instruction': system_instruction,
                    'contents': [
                        {
                            'role': str(content.get('role') or 'user'),
                            'parts': [
                                (
                                    {
                                        'type': 'text',
                                        'text': part,
                                    }
                                    if isinstance(part, str)
                                    else {
                                        'type': 'inline_audio',
                                        'mime_type': str(part.get('mime_type') or ''),
                                        'byte_count': len(part.get('data') or b''),
                                    }
                                )
                                for part in content.get('parts') or []
                            ],
                        }
                        for content in contents
                    ],
                    'generation_config': generation_config or None,
                },
                'submission_trace': {
                    'exact_submission_available': True,
                    'builder': f'{self.__class__.__name__}.generate_inner',
                },
                'response_status': 'ok',
                'response_error': None,
            }
        except Exception as e:
            logger.error(f'Gemini API error ({self.model_name}): {e}')
            return {
                'prompt': prompt,
                'response_text': None,
                'request_payload': {
                    'provider': 'google',
                    'model_name': self.model_name,
                    'audio_paths': audio_paths,
                    'prompt': prompt,
                    'system_instruction': system_instruction,
                    'messages': messages,
                },
                'response_payload': None,
                'submission_payload': {
                    'provider': 'google',
                    'transport': 'gemini-generate-content',
                    'model_name': self.model_name,
                    'system_instruction': system_instruction,
                    'contents': None,
                    'generation_config': None,
                },
                'submission_trace': {
                    'exact_submission_available': False,
                    'builder': f'{self.__class__.__name__}.generate_inner',
                    'error': str(e),
                },
                'response_status': 'error',
                'response_error': str(e),
            }


class Gemini25FlashNativeAudio(GeminiAudioBase):
    NAME = 'Gemini-2.5-Flash-Audio'

    def __init__(self, **kwargs):
        super().__init__(model_name=GEMINI_MODEL_IDS[self.NAME], **kwargs)


class Gemini3Flash(GeminiAudioBase):
    NAME = 'Gemini-3-Flash'

    def __init__(self, **kwargs):
        super().__init__(model_name=GEMINI_MODEL_IDS[self.NAME], **kwargs)


class Gemini31Pro(GeminiAudioBase):
    NAME = 'Gemini-3.1-Pro'

    def __init__(self, **kwargs):
        super().__init__(model_name=GEMINI_MODEL_IDS[self.NAME], **kwargs)


class Gemini25Pro(GeminiAudioBase):
    NAME = 'Gemini-2.5-Pro'

    def __init__(self, **kwargs):
        super().__init__(model_name=GEMINI_MODEL_IDS[self.NAME], **kwargs)
