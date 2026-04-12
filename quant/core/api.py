import torch
from transformers import AutoConfig
from quant.nn_models import *

Quant_CASUAL_LM_MODEL_MAP = {"qwen2": Qwen2ModelForCausalLM}


def check_and_getmodel_type(model_dir, trust_remote_code=True, **model_init_kwargs):
    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code, **model_init_kwargs
    )
    print("Model type is:", config.model_type)
    if config.model_type not in Quant_CASUAL_LM_MODEL_MAP.keys():
        raise TypeError(
            f"Model type {config.model_type} is not supported for quantization"
        )
    model_type = config.model_type

    return model_type


class AutoQuantizeForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "You must instantiate AutoQuantizeForCausalLM  with from_pretrained function"
        )

    @classmethod
    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map="auto",
        low_cpu_mem_usage=False,
        use_cache=False,
        **model_init_kwargs,
    ):
        model_type = check_and_getmodel_type(
            model_path, trust_remote_code, **model_init_kwargs
        )
        return Quant_CASUAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path,
            model_type,
            torch_dtype=torch_dtype,  # 模型的权重类型
            trust_remote_code=trust_remote_code,  # 相信远端的 code或模型
            safetensors=safetensors,  # 模型权重格式
            device_map=device_map,  # 指定模型加载设备
            low_cpu_mem_usage=low_cpu_mem_usage,  # 控制模型加载时是否尽量减少内存占用，如果为true，模型
            # 会分布加载权重到 gpu，避免一次性占用大量内存
            use_cache=use_cache,  # 控制是否使kv cache来缓存 kv
            **model_init_kwargs,
        )
