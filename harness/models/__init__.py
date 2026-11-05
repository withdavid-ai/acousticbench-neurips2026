"""Model registry — discovers and builds model adapters.

Models are registered by their adapter files in this package. Each adapter
defines a class with a NAME attribute that becomes the model identifier.

For open-weight models, adapters may define both a Modal and a Docker class
sharing the same NAME. The active backend (set via set_backend()) determines
which class is used.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from .base import BaseModel

_EXCLUDED_CLASSES = {"BaseModel", "ModalRemoteModel", "DockerRemoteModel", "GeminiAudioBase"}

_backend: str = "docker"
_all_classes: dict[str, list[type]] | None = None


def set_backend(backend: str) -> None:
    global _backend, _registry
    if backend not in ("docker", "modal", "api"):
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'docker', 'modal', or 'api'.")
    _backend = backend
    _registry = None


def get_backend() -> str:
    return _backend


def _classify_backend(cls: type) -> str:
    if hasattr(cls, "MODAL_APP_NAME") and cls.MODAL_APP_NAME:
        return "modal"
    if hasattr(cls, "GENERATE_PATH"):
        return "docker"
    return "api"


def _discover_all() -> dict[str, list[type]]:
    global _all_classes
    if _all_classes is not None:
        return _all_classes

    by_name: dict[str, list[type]] = {}
    package = importlib.import_module(__package__)

    for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
        if name.startswith("_") or name == "message_utils":
            continue
        try:
            module = importlib.import_module(f"{__package__}.{name}")
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(cls, BaseModel)
                    and cls.__name__ not in _EXCLUDED_CLASSES
                    and getattr(cls, "NAME", None)
                ):
                    by_name.setdefault(cls.NAME, []).append(cls)
        except Exception:
            pass

    _all_classes = by_name
    return by_name


_registry: dict[str, type] | None = None


def _discover_models() -> dict[str, type]:
    global _registry
    if _registry is not None:
        return _registry

    models = {}
    for name, classes in _discover_all().items():
        if len(classes) == 1:
            models[name] = classes[0]
            continue
        for cls in classes:
            if _classify_backend(cls) == _backend:
                models[name] = cls
                break
        else:
            models[name] = classes[0]

    _registry = models
    return models


def build_model(name: str, **kwargs) -> BaseModel:
    """Build a model adapter by name using the active backend."""
    models = _discover_models()
    if name not in models:
        available = sorted(models.keys())
        raise ValueError(
            f"Unknown model: {name!r}. Available: {available}"
        )
    return models[name](**kwargs)


def build_model_with_backend(name: str, backend: str, **kwargs) -> BaseModel:
    """Build a model adapter by name with an explicit backend override."""
    for cls in _discover_all().get(name, []):
        if _classify_backend(cls) == backend:
            return cls(**kwargs)
    raise ValueError(
        f"No {backend!r} adapter found for model {name!r}."
    )


def list_models() -> dict[str, dict[str, Any]]:
    """List all available models with their metadata."""
    result = {}
    for name, classes in sorted(_discover_all().items()):
        backends = [_classify_backend(cls) for cls in classes]
        active_cls = None
        for cls in classes:
            if _classify_backend(cls) == _backend:
                active_cls = cls
                break
        if active_cls is None:
            active_cls = classes[0]

        info: dict[str, Any] = {
            "type": _classify_backend(active_cls),
            "backends": sorted(set(backends)),
        }
        if hasattr(active_cls, "MODAL_APP_NAME") and active_cls.MODAL_APP_NAME:
            info["modal_app"] = active_cls.MODAL_APP_NAME
        result[name] = info
    return result


def get_model_metadata(name: str) -> dict[str, Any]:
    """Return metadata about a registered model without instantiating it."""
    classes = _discover_all().get(name)
    if not classes:
        available = sorted(_discover_all().keys())
        raise ValueError(f"Unknown model: {name!r}. Available: {available}")

    backends = sorted({_classify_backend(cls) for cls in classes})
    return {"backends": backends}
