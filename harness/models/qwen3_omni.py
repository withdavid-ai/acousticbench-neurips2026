import os

_THINKER_MAX_TOKENS_ENV = "AUDIOBENCH_QWEN3_OMNI_THINKER_MAX_TOKENS"


def _inject_thinker_meta(msg: dict) -> dict:
    prepared = dict(msg)
    prepared['meta'] = dict(msg.get('meta', {}))
    thinker_max_tokens = os.environ.get(_THINKER_MAX_TOKENS_ENV, '').strip()
    if thinker_max_tokens:
        prepared['meta']['_thinker_max_new_tokens'] = thinker_max_tokens
    return prepared


try:
    from .modal_base import ModalRemoteModel

    class Qwen3OmniModal(ModalRemoteModel):
        NAME = 'Qwen3-Omni'
        MODAL_APP_NAME = 'audiobench-qwen3-omni'
        MODAL_FUNCTION_NAME = 'Qwen3OmniServer.generate'
        MODAL_MAX_RETRIES = 2

        def generate_inner(self, msg: dict):
            return super().generate_inner(_inject_thinker_meta(msg))

except Exception:
    pass


from .docker_base import DockerRemoteModel


class Qwen3OmniDocker(DockerRemoteModel):
    NAME = 'Qwen3-Omni'
    REQUEST_TIMEOUT_S = 600
    MAX_RETRIES = 2

    def generate_inner(self, msg: dict):
        return super().generate_inner(_inject_thinker_meta(msg))
