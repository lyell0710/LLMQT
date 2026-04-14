import torch
import transformers
import torch.nn as nn

from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
)

from .config import QuantConfig
from huggingface_hub import snapshot_download, save_torch_state_dict

TRANSFORMERS_AUTO_MAPPING_DICT = {
    "llama": "AutoModelForCausalLM",
    "opt": "AutoModelForCausalLM",
    "qwen2": "AutoModelForCausalLM",
    "qwen3": "AutoModelForCausalLM",
    "qwen3_moe": "AutoModelForCausalLM",
    "deepseek_v3": "AutoModelForCausalLM",
    "qwen2_distilled_r1": "AutoModelForCausalLM",
    "llama4": "AutoModelForCausalLM",
}


class BaseModelForCausalLM(nn.Module):
    def __init__(
        self,
        model,
        model_type,
        is_quantized,
        config,
        quant_config,
    ):
        """The base model for all models."""
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config: QuantConfig = quant_config

    @classmethod
    def from_pretrained(
        self,
        model_path,
        model_type,
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ):
        """A method for initialization of pretrained models, usually in FP16."""
        # Get weights path and quant config by AutoConfig and QuantConfig
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            "",
            safetensors,
            trust_remote_code=trust_remote_code,
        )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)
        print("target cls name is ", target_cls_name)

        if model_init_kwargs.get("low_cpu_mem_usage") is None:
            model_init_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if (
            model_init_kwargs.get("use_cache") is None
            and model_type != "llama4"
            and not (
                (target_cls_name == "AutoModelForVision2Seq")
                or (target_cls_name == "AutoModelForTextToWaveform")
            )
        ):
            model_init_kwargs["use_cache"] = use_cache

        model = target_cls.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            device_map=device_map,
            **model_init_kwargs,
        )

        model.eval()

        return self(
            model,
            model_type,
            is_quantized=False,
            config=config,
            quant_config=quant_config,
        )

    @torch.no_grad()
    def quantize(
        self,
        tokenizer=None,
        quant_config={},
        calib_data="pileval",
        duo_scaling=True,  # whther to scale using both w/x or just x,
        fake_quant=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
        **kwargs,
    ):
        self.quant_config: QuantConfig = QuantConfig.from_dict(quant_config)

        if hasatter(self, "modiles_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert

        quantizer_cls = get_concrete_quantizer_cls(self.quant_config.quant_method)
        self.quantizer = quantizer_cls(
            self,
            self.model,
            self.model_type,
            tokenizer,
            self.quant_config,
            self.quant_config.quant_method,
        )
        self.quantizer.quantize()
        self.is_quantized = True

    def save_quantized(self, save_dir):
        save_torch_state_dict(
            state_dict=self.model.state_dict(),
            save_directory=save_dir,
        )

    def _load_config(
        self,
        model_path,
        safetensors=True,
        trust_remote_code=True,
    ):

        # step1: download model if path is not a dir
        model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)

        # step2: load config and set seqlen

        quant_config = QuantConfig.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        return model_path, config, quant_config
