from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel


class Phi4Multimodal(ModalRemoteModel):
    NAME = 'Phi-4-Multimodal'
    MODAL_APP_NAME = 'audiobench-phi4mm'
    MODAL_FUNCTION_NAME = 'Phi4MMServer.generate'


class Phi4MultimodalDocker(DockerRemoteModel):
    NAME = 'Phi-4-Multimodal'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2
