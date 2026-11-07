from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel


class KimiAudio(ModalRemoteModel):
    NAME = 'Kimi-Audio'
    MODAL_APP_NAME = 'audiobench-kimi-audio'
    MODAL_FUNCTION_NAME = 'KimiAudioServer.generate'


class KimiAudioDocker(DockerRemoteModel):
    NAME = 'Kimi-Audio'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2
