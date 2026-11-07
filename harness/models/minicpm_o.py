from .base import BaseModel
from .docker_base import DockerRemoteModel
from .modal_base import ModalRemoteModel
from ..minicpm_input_adapter import MiniCPMConcatAdapter, minicpm_runtime_adapter_context

_MINICPM_MAX_AUDIO_SECONDS = 300.0


class _MiniCPMLongAudioMixin:
    @staticmethod
    def check_audio_legal(audio_path, max_duration: float = _MINICPM_MAX_AUDIO_SECONDS) -> bool:
        return BaseModel.check_audio_legal(audio_path, max_duration=max_duration)


class _MiniCPMConcatMixin(_MiniCPMLongAudioMixin):
    _INPUT_ADAPTERS = (MiniCPMConcatAdapter(),)

    def generate_inner(self, msg: dict):
        with minicpm_runtime_adapter_context(msg, self._INPUT_ADAPTERS) as adapted:
            return super().generate_inner(adapted)


class MiniCPMo45(_MiniCPMConcatMixin, ModalRemoteModel):
    NAME = 'MiniCPM-o-4.5'
    MODAL_APP_NAME = 'audiobench-minicpm-o'
    MODAL_FUNCTION_NAME = 'MiniCPMo45Server.generate'


class MiniCPMo45Docker(_MiniCPMConcatMixin, DockerRemoteModel):
    NAME = 'MiniCPM-o-4.5'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2
