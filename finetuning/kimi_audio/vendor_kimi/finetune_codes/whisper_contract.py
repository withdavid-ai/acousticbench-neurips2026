from __future__ import annotations

from typing import Optional

import torch


def normalize_whisper_input_feature(
    whisper_input_feature: Optional[torch.FloatTensor],
) -> Optional[list[list[torch.Tensor]]]:
    if whisper_input_feature is None:
        return None
    if not isinstance(whisper_input_feature, list):
        if hasattr(whisper_input_feature, "squeeze"):
            whisper_input_feature = whisper_input_feature.squeeze(0)
        if getattr(whisper_input_feature, "ndim", 0) == 1:
            whisper_input_feature = [[whisper_input_feature]]
        else:
            whisper_input_feature = [list(whisper_input_feature)]
    elif whisper_input_feature and not isinstance(whisper_input_feature[0], list):
        whisper_input_feature = [list(whisper_input_feature)]
    return whisper_input_feature


def is_packed_whisper_feature(
    feature: torch.Tensor,
    *,
    adaptor_input_dim: int,
) -> bool:
    ndim = int(getattr(feature, "ndim", 0))
    if ndim not in (2, 3):
        return False
    return int(feature.shape[-1]) == int(adaptor_input_dim)


def is_precomputed_whisper_hidden_state(
    feature: torch.Tensor,
    *,
    adaptor_input_dim: int,
) -> bool:
    ndim = int(getattr(feature, "ndim", 0))
    if ndim not in (2, 3):
        return False
    return int(feature.shape[-1]) * 4 == int(adaptor_input_dim)


def pack_precomputed_whisper_hidden_state(
    feature: torch.Tensor,
    *,
    adaptor_input_dim: int,
) -> torch.Tensor:
    if not is_precomputed_whisper_hidden_state(
        feature,
        adaptor_input_dim=adaptor_input_dim,
    ):
        raise ValueError(
            "Expected precomputed Whisper hidden states with last_dim * 4 == "
            f"{int(adaptor_input_dim)}, got shape={tuple(feature.shape)}"
        )
    if int(getattr(feature, "ndim", 0)) == 2:
        feature = feature.unsqueeze(0)
    if int(feature.shape[1]) % 4 != 0:
        raise ValueError(
            "Expected Whisper hidden-state sequence length divisible by 4, got "
            f"shape={tuple(feature.shape)}"
        )
    return feature.reshape(
        int(feature.shape[0]),
        int(feature.shape[1] // 4),
        int(feature.shape[2] * 4),
    )
