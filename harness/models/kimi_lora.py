from .base import BaseModel
from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel


class KimiAudioLoRA(ModalRemoteModel):
    NAME = None
    MODAL_APP_NAME = 'audiobench-kimi-lora'
    MODAL_FUNCTION_NAME = 'KimiLoRAServer.generate'
    # Keep one parameterized Modal class instance per adapter in the client.
    # The server loads/export-caches the adapter in @modal.enter(), then
    # generate() reuses the resident self.model for subsequent calls.
    MODAL_INSTANCE_MODE = 'cached'
    MODAL_CALL_TIMEOUT_S = 1200
    ADAPTER_NAME = None

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 200) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)

    def __init__(self, adapter_name: str | None = None, **kwargs):
        self.adapter_name = str(adapter_name or self.ADAPTER_NAME or '').strip()
        if not self.adapter_name:
            raise ValueError(f'{self.__class__.__name__} must define ADAPTER_NAME')
        super().__init__(**kwargs)

    def remote_class_kwargs(self) -> dict:
        return {'adapter_name': self.adapter_name}


class KimiAudioLoRARAD(KimiAudioLoRA):
    NAME = 'Kimi-Audio-LoRA-RAD'
    ADAPTER_NAME = 'rad'


class KimiAudioLoRARSQ(KimiAudioLoRA):
    NAME = 'Kimi-Audio-LoRA-RSQ'
    ADAPTER_NAME = 'rsq'


class KimiAudioLoRASERClassify(KimiAudioLoRA):
    NAME = 'Kimi-Audio-LoRA-SER-Classify'
    ADAPTER_NAME = 'ser_classify'


class KimiAudioLoRASERIntensity(KimiAudioLoRA):
    NAME = 'Kimi-Audio-LoRA-SER-Intensity'
    ADAPTER_NAME = 'ser_intensity'


class KimiAudioLoRASimpleAudioQA(KimiAudioLoRA):
    NAME = 'Kimi-Audio-LoRA-Simple-AudioQA'
    ADAPTER_NAME = 'simple_audioqa'


class KimiAudioLoRADocker(DockerRemoteModel):
    """Docker counterpart of :class:`KimiAudioLoRA`.

    Each adapter is served by its own container (one Docker process per
    LoRA) bound to a distinct port and ``ACOUSTICBENCH_*_ENDPOINT`` env var.
    The selected adapter is configured server-side via the ``ADAPTER_NAME``
    environment variable at ``docker run`` time, so this client adapter is a
    plain HTTP shim that doesn't need to send the adapter name in the
    request payload.
    """

    NAME = None
    ADAPTER_NAME = None
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2

    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = 200) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)


class KimiAudioLoRADockerRAD(KimiAudioLoRADocker):
    NAME = 'Kimi-Audio-LoRA-RAD'
    ADAPTER_NAME = 'rad'


class KimiAudioLoRADockerRSQ(KimiAudioLoRADocker):
    NAME = 'Kimi-Audio-LoRA-RSQ'
    ADAPTER_NAME = 'rsq'


class KimiAudioLoRADockerSERClassify(KimiAudioLoRADocker):
    NAME = 'Kimi-Audio-LoRA-SER-Classify'
    ADAPTER_NAME = 'ser_classify'


class KimiAudioLoRADockerSERIntensity(KimiAudioLoRADocker):
    NAME = 'Kimi-Audio-LoRA-SER-Intensity'
    ADAPTER_NAME = 'ser_intensity'


class KimiAudioLoRADockerSimpleAudioQA(KimiAudioLoRADocker):
    NAME = 'Kimi-Audio-LoRA-Simple-AudioQA'
    ADAPTER_NAME = 'simple_audioqa'


KIMI_LORA_MODEL_NAMES: tuple[str, ...] = (
    KimiAudioLoRARAD.NAME,
    KimiAudioLoRARSQ.NAME,
    KimiAudioLoRASERClassify.NAME,
    KimiAudioLoRASERIntensity.NAME,
    KimiAudioLoRASimpleAudioQA.NAME,
)
