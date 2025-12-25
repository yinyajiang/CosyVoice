import sys
import os

os.environ["VLLM_NO_USAGE_STATS"] = "1"
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')


def _vllmAutoModel(**kwargs):
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
    from cosyvoice.cli.cosyvoice import AutoModel

    kwargs['load_vllm'] = True
    return AutoModel(**kwargs)


def _withoutVllmAutoModel(**kwargs):
    from cosyvoice.cli.cosyvoice import AutoModel

    kwargs['load_vllm'] = False
    return AutoModel(**kwargs)


def AutoCosyVoice(**kwargs):
    if kwargs.get('load_vllm', False):
        return _vllmAutoModel(**kwargs)
    else:
        return _withoutVllmAutoModel(**kwargs)


