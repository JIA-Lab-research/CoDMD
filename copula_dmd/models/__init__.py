from .wan.wan_wrapper import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
from copula_dmd.bidirectional_trajectory_pipeline import BidirectionalInferenceWrapper
from transformers.models.t5.modeling_t5 import T5Block


def get_diffusion_wrapper(model_name, model_path=None):
    """Get diffusion wrapper class with model path.

    Returns a wrapper class that can be instantiated without arguments.
    """
    if model_name != "wan":
        raise ValueError(f"Unsupported model: {model_name}")

    class WrapperWithConfig(WanDiffusionWrapper):
        def __init__(self):
            super().__init__(model_path=model_path)

    return WrapperWithConfig


def get_text_encoder_wrapper(model_name):
    return WanTextEncoder


def get_vae_wrapper(model_name):
    return WanVAEWrapper


def get_inference_pipeline_wrapper(model_name, **kwargs):
    return BidirectionalInferenceWrapper(**kwargs)


def get_block_class(block_name):
    if block_name == "T5Block":
        return T5Block
    raise ValueError(f"Unknown block class: {block_name}")
