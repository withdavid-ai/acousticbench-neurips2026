from .base import BaseModel
from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel


class VoxtralSmall24B(ModalRemoteModel):
    NAME = 'Voxtral-Small-24B'
    MODAL_APP_NAME = 'audiobench-voxtral'
    MODAL_FUNCTION_NAME = 'VoxtralSmall24BServer.generate'

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 1800) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)


class VoxtralMini3B(ModalRemoteModel):
    NAME = 'Voxtral-Mini-3B'
    MODAL_APP_NAME = 'audiobench-voxtral'
    MODAL_FUNCTION_NAME = 'VoxtralMini3BServer.generate'

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 1800) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)


class VoxtralSmall24BDocker(DockerRemoteModel):
    NAME = 'Voxtral-Small-24B'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 1800) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)


class VoxtralMini3BDocker(DockerRemoteModel):
    NAME = 'Voxtral-Mini-3B'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 1800) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)
