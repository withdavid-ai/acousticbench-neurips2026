from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel


class StepAudio2Mini(ModalRemoteModel):
    NAME = 'Step-Audio-2-mini'
    MODAL_APP_NAME = 'audiobench-step-audio2'
    MODAL_FUNCTION_NAME = 'StepAudio2MiniServer.generate'


class StepAudio2MiniDocker(DockerRemoteModel):
    NAME = 'Step-Audio-2-mini'
    GENERATE_PATH = '/generate'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2
