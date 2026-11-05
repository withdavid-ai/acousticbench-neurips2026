import argparse

import torch
from peft import PeftModel

from finetuning.kimi_audio.vendor_kimi.finetune_codes.model import KimiAudioModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_model = KimiAudioModel.from_pretrained(
        args.base_model_path,
        device_map=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    merged_model = adapter_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
