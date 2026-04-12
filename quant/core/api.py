import torch
from transformers import AutoConfig
from quant.nn_modules import *

Quant_CASUAL_LM_MAP={
    "qwen2":Qwen2ModelForCausalLM
}
def check_and_getmodel_type(model_path,trust_remote_code=True,**model_init_kwargs):
    config=AutoConfig.from_pretrained(model_dir,trust_remote_code,**model_init_kwargs)
    print("Model type is:",config.model_type)
    if(config.model_type not in Quant_CASUAL_LM_MAP.keys()):
        raise TypeError(f"Model type {config.model_type} is not supported for quantization")
    model_type=config.model_type

    return model_type
class AutoQuantizeForCausalLM:
    def __init__(self):

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
       model_type=check_and_getmodel_type(model_path,trust_remote_code,**model_init_kwargs)