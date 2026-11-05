import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from .modeling_kimia import MoonshotKimiaForCausalLM
from .whisper_contract import (
    is_packed_whisper_feature,
    is_precomputed_whisper_hidden_state,
    normalize_whisper_input_feature,
    pack_precomputed_whisper_hidden_state,
)


class KimiAudioModel(MoonshotKimiaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.whisper_model = WhisperEncoder("openai/whisper-large-v3", mel_batch_size=20, unfreeze_online_whisper_model=True)

    @classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_name_or_path)

        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True, **model_load_kwargs,
        )

        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=True
        )
        kimia_model = cls(audio_model.config)

        # merge audio model and whisper model's state dict
        pretrained_state_dict = audio_model.state_dict()

        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        kimia_model.load_state_dict(pretrained_state_dict)

        return kimia_model

    @staticmethod
    def export_model(input_dir, output_dir):
        print("Loading model from {}".format(input_dir))
        kimiaudio = KimiAudioModel.from_pretrained(input_dir)

        print("Saving Kimi-Audio LM to {}".format(output_dir))
        audio_model = MoonshotKimiaForCausalLM(kimiaudio.config)
        audio_model_state_dict = {k: v for k, v in kimiaudio.state_dict().items() if not k.startswith("whisper_model")}
        audio_model.load_state_dict(audio_model_state_dict)

        audio_model.save_pretrained(output_dir)

        source_dir = Path(__file__).resolve().parent
        shutil.copyfile(
            str(source_dir / "configuration_moonshot_kimia.py"),
            os.path.join(output_dir, "configuration_moonshot_kimia.py"),
        )
        shutil.copyfile(
            str(source_dir / "modeling_kimia.py"),
            os.path.join(output_dir, "modeling_moonshot_kimia.py"),
        )

        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel

        whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")

        kimiaudio_whisper_encoder_state_dict = {k.replace("speech_encoder.", "encoder."): v for k, v in kimiaudio.whisper_model.state_dict().items() if k.startswith("speech_encoder")}

        missing_keys, unexpected_keys = whisper_model.load_state_dict(kimiaudio_whisper_encoder_state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        for k in missing_keys:
            assert k.startswith("decoder"), f"Missing keys: {k}"

        whisper_model.save_pretrained(os.path.join(output_dir, "whisper-large-v3"))

        print("Exported Kimi-Audio LM and Whisper model to {}".format(output_dir))


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        normalized_whisper = normalize_whisper_input_feature(whisper_input_feature)
        whisper_feats = None
        if normalized_whisper is not None:
            if next(self.whisper_model.parameters()).dtype != torch.bfloat16:
                self.whisper_model = self.whisper_model.to(dtype=torch.bfloat16)
            adaptor_input_dim = int(self.config.kimia_adaptor_input_dim)
            adaptor_dtype = next(self.model.vq_adaptor.parameters()).dtype
            whisper_feats = []
            for sample_features in normalized_whisper:
                sample_whisper_feats = []
                for feature in sample_features:
                    if isinstance(feature, torch.Tensor):
                        feature_tensor = feature.to(torch.float32)
                    else:
                        feature_tensor = torch.from_numpy(feature)
                    if getattr(feature_tensor, "ndim", 0) == 1:
                        feature_tensor = feature_tensor.unsqueeze(0)
                    feature_tensor = feature_tensor[:, :].to(
                        torch.cuda.current_device(), dtype=torch.float32
                    )
                    # Checkpoint eval can hand us precomputed Whisper states from
                    # `tokenize_waveform()` in `[batch, time, 1280]` form. Those
                    # must be packed for the adaptor, not re-encoded through Whisper.
                    if is_packed_whisper_feature(
                        feature_tensor,
                        adaptor_input_dim=adaptor_input_dim,
                    ):
                        packed_feature = (
                            feature_tensor.unsqueeze(0)
                            if getattr(feature_tensor, "ndim", 0) == 2
                            else feature_tensor
                        )
                    elif is_precomputed_whisper_hidden_state(
                        feature_tensor,
                        adaptor_input_dim=adaptor_input_dim,
                    ):
                        packed_feature = pack_precomputed_whisper_hidden_state(
                            feature_tensor,
                            adaptor_input_dim=adaptor_input_dim,
                        )
                    else:
                        with torch.autocast(device_type="cuda", enabled=False):
                            whisper_hidden = self.whisper_model(feature_tensor)
                        if is_packed_whisper_feature(
                            whisper_hidden,
                            adaptor_input_dim=adaptor_input_dim,
                        ):
                            packed_feature = whisper_hidden
                        else:
                            packed_feature = pack_precomputed_whisper_hidden_state(
                                whisper_hidden,
                                adaptor_input_dim=adaptor_input_dim,
                            )
                    packed_feature = packed_feature.to(dtype=adaptor_dtype)
                    sample_whisper_feats.append(packed_feature)
                whisper_feats.append(sample_whisper_feats)
        return super().forward(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=return_dict,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--action", type=str, choices=["init_from_pretrained", "export_model"], default="init_from_pretrained")
    parser.add_argument("--output_dir", type=str, default="output/pretrained_hf")
    parser.add_argument("--input_dir", type=str, default="output/finetuned_hf")
    args = parser.parse_args()

    if args.action == "init_from_pretrained":

        model = KimiAudioModel.init_from_pretrained(args.model_name, model_load_kwargs={})

        os.makedirs(args.output_dir, exist_ok=True)
        # save model
        model.save_pretrained(args.output_dir)
    elif args.action == "export_model":
        KimiAudioModel.export_model(args.input_dir, args.output_dir)
    else:
        raise ValueError(f"Invalid action: {args.action}")
