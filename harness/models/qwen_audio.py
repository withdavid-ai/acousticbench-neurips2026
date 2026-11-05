from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel


class Qwen2AudioChat(ModalRemoteModel):
    NAME = 'Qwen2-Audio-7B-Instruct'
    MODAL_APP_NAME = 'audiobench-qwen2-audio'
    MODAL_FUNCTION_NAME = 'Qwen2AudioChatServer.generate_chat'


class Qwen2AudioChatDocker(DockerRemoteModel):
    NAME = 'Qwen2-Audio-7B-Instruct'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2
