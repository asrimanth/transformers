import copy
from typing import Dict

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

EDSR_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class EDSRConfig(PretrainedConfig):
    r"""
     This is the configuration class to store the configuration of a [`EDSRModel`]. It is used to instantiate an
    enhanced Deep Residual networks for super resolution according to the specified arguments, defining the model
    architecture. Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.
    Args:
        upscale (`int`, defaults to 2):
            The scale factor of the model, can be 2, 3, or 4
            upscale = 2 => Model outputs an image which is 2x bigger than the input image.
            upscale = 3 => Model outputs an image which is 3x bigger than the input image.
            upscale = 4 => Model outputs an image which is 4x bigger than the input image.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        num_res_block (`int`, *optional*, defaults to 16):
            Number of residual blocks.
        num_feature_maps (`int`, *optional*, defaults to 64):
            Number of feature maps.
        res_scale (`int`, *optional*, defaults to 1):
            Residual scaling.

    Example:
    ```python
    >>> from transformers import EDSRConfig, EDSRModel

    >>> configuration = EDSRConfig()
    >>> # Initializing a model (with random weights) from the microsoft/swin2sr_tiny_patch4_windows8_256 style configuration
    >>> model = EDSRModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    attribute_map = {
        "hidden_size": "embed_dim",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        upscale: int = 2,
        num_channels: int = 3,
        hidden_act: str = "relu",
        num_res_block: int = 16,
        num_feature_maps: int = 64,
        res_scale: int = 1,
        rgb_range: int = 255,
        rgb_mean: tuple = (0.4488, 0.4371, 0.4040),
        rgb_std: tuple = (1.0, 1.0, 1.0),
        **kwargs,
    ):
        # Config should not be importable.

        self.upscale = upscale
        self.num_channels = num_channels
        self.hidden_act = hidden_act
        self.num_res_block = num_res_block
        self.num_feature_maps = num_feature_maps
        self.res_scale = res_scale
        self.rgb_range = rgb_range
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        super().__init__(**kwargs)

        # This is the configuration of EDSR Baseline x2 with 16 res blocks and 64 feature maps.
        # TODO: Implement to_dict method, example at DPT Config.
        # TODO: push_to_hub example at mask2former
        # TODO: Mean and std in config

    def to_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        return output
