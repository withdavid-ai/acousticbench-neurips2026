import os

from .base import BaseModel
from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel

DEFAULT_AUDIO_FLAMINGO3_VARIANT = "chat"
AUDIO_FLAMINGO3_VARIANTS = {
    "chat": "AudioFlamingo3Server.generate",
    "base": "AudioFlamingo3BaseServer.generate",
}

class AudioFlamingo3(ModalRemoteModel):
    NAME = 'Audio-Flamingo-3'
    MODAL_APP_NAME = 'audiobench-audio-flamingo3'
    MODAL_FUNCTION_NAME = AUDIO_FLAMINGO3_VARIANTS[DEFAULT_AUDIO_FLAMINGO3_VARIANT]

    def __init__(self, **kwargs):
        variant = (
            os.environ.get("AUDIO_FLAMINGO3_VARIANT")
            or DEFAULT_AUDIO_FLAMINGO3_VARIANT
        ).strip().lower()
        if variant not in AUDIO_FLAMINGO3_VARIANTS:
            raise ValueError(
                f"Unknown AUDIO_FLAMINGO3_VARIANT={variant!r}; "
                f"expected one of {sorted(AUDIO_FLAMINGO3_VARIANTS)}"
            )
        self.MODAL_FUNCTION_NAME = AUDIO_FLAMINGO3_VARIANTS[variant]
        super().__init__(**kwargs)

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 600) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)


class AudioFlamingo3Docker(DockerRemoteModel):
    NAME = 'Audio-Flamingo-3'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 600) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)
