from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel


class MiMoAudio(ModalRemoteModel):
    NAME = 'MiMo-Audio-7B'
    MODAL_APP_NAME = 'audiobench-mimo-audio'
    MODAL_FUNCTION_NAME = 'MiMoAudioServer.generate'


class MiMoAudioDocker(DockerRemoteModel):
    NAME = 'MiMo-Audio-7B'
    GENERATE_PATH = '/generate'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2
