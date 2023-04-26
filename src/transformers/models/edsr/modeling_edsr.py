import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_edsr import EDSRConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EDSRConfig"
_FEAT_EXTRACTOR_FOR_DOC = "EDSRImageProcessor"

# Base docstring
_CHECKPOINT_FOR_DOC = "asrimanth/edsr-base-x2"


EDSR_PRETRAINED_MODEL_ARCHIVE_LIST = ["asrimanth/edsr-base-x2", "asrimanth/edsr-base-x3", "asrimanth/edsr-base-x4"]
temp = {
    "asrimanth/edsr-baseline-x2": "https://huggingface.co/huggingface/edsr-base-x2/resolve/main/config.json",
    "asrimanth/edsr-baseline-x3": "https://huggingface.co/huggingface/edsr-base-x3/resolve/main/config.json",
    "asrimanth/edsr-baseline-x4": "https://huggingface.co/huggingface/edsr-base-x4/resolve/main/config.json",
}


logger = logging.get_logger(__name__)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class EDSRMeanShift(nn.Conv2d):
    def __init__(self, config, sign=-1):
        super(EDSRMeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(config.rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * config.rgb_range * torch.Tensor(config.rgb_mean) / std
        for param in self.parameters():
            param.requires_grad = False


class EDSRBasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, bias=False, batch_norm=True, activation=nn.ReLU(True)
    ):
        block_module = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if batch_norm:
            block_module.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            block_module.append(activation)

        super(EDSRBasicBlock, self).__init__(*block_module)


class EDSRResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, batch_norm=False, activation=nn.ReLU(True), res_scale=1):
        super(EDSRResBlock, self).__init__()
        res_block_module = []
        for i in range(2):
            res_block_module.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if batch_norm:
                res_block_module.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                res_block_module.append(activation)

        self.edsr_body = nn.Sequential(*res_block_module)
        self.res_scale = res_scale

    def forward(self, feature_maps):
        residual = self.edsr_body(feature_maps).mul(self.res_scale)
        residual += feature_maps

        return residual


class EDSRUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, batch_norm=False, activation=nn.ReLU(True), bias=True):
        upsample_module = []
        scale = int(scale)
        # if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
        if scale == 2 or scale == 4:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                upsample_module.append(conv(n_feats, 4 * n_feats, 3, bias))
                upsample_module.append(nn.PixelShuffle(2))
                if batch_norm:
                    upsample_module.append(nn.BatchNorm2d(n_feats))
                if activation is not None:
                    upsample_module.append(activation)

        elif scale == 3:
            upsample_module.append(conv(n_feats, 9 * n_feats, 3, bias))
            upsample_module.append(nn.PixelShuffle(3))
            if batch_norm:
                upsample_module.append(nn.BatchNorm2d(n_feats))
            if activation is not None:
                upsample_module.append(activation)
        else:
            print(scale, type(scale))
            raise NotImplementedError

        super(EDSRUpsampler, self).__init__(*upsample_module)


class EDSRPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EDSRConfig
    base_model_prefix = "edsr"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.trunc_normal_(module.weight.data, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value


EDSR_START_DOCSTRING = r"""
    Parameters:
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
        config ([`EDSRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EDSR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """
    The bare EDSR Model without an EDSRUpsampler head on top for image super resolution and restoration.
    """,
    EDSR_START_DOCSTRING,
)
class EDSRModel(EDSRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        num_res_block = config.num_res_block
        num_feature_maps = config.num_feature_maps
        kernel_size = 3
        activation = self.get_activation_function(config.hidden_act)
        self.sub_mean = EDSRMeanShift(config)
        self.add_mean = EDSRMeanShift(config, sign=1)

        # define head module
        edsr_head = [default_conv(config.num_channels, num_feature_maps, kernel_size)]

        # define body module
        edsr_body = [
            EDSRResBlock(
                default_conv, num_feature_maps, kernel_size, activation=activation, res_scale=config.res_scale
            )
            for _ in range(num_res_block)
        ]
        edsr_body.append(default_conv(num_feature_maps, num_feature_maps, kernel_size))

        self.edsr_head = nn.Sequential(*edsr_head)
        self.edsr_body = nn.Sequential(*edsr_body)

    def get_activation_function(self, activation_str: str):
        activation_str_to_activation_dict = {
            "relu": nn.ReLU(True),
            "gelu": nn.GELU(True),
            "selu": nn.SELU(True),
        }
        return activation_str_to_activation_dict[activation_str.lower()]

    @add_start_docstrings_to_model_forward(EDSR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, pixel_values: Optional[torch.FloatTensor] = None, output_hidden_states: bool = False
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import EDSRImageProcessor, EDSRForImageSuperResolution
         >>> from datasets import load_dataset

         >>> processor = EDSRImageProcessor.from_pretrained("asrimanth/edsr-base-x2")
         >>> model = EDSRForImageSuperResolution.from_pretrained("asrimanth/edsr-base-x2")
         ```
         """

        hidden_states = () if output_hidden_states else None

        pixel_values = self.sub_mean(pixel_values)
        if output_hidden_states:
            hidden_states = hidden_states + (pixel_values,)
        pixel_values = self.edsr_head(pixel_values)
        if output_hidden_states:
            hidden_states = hidden_states + (pixel_values,)

        residual = self.edsr_body(pixel_values)
        pixel_values += residual
        if output_hidden_states:
            hidden_states = hidden_states + (pixel_values,)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=pixel_values,
            hidden_states=hidden_states,
        )


@add_start_docstrings(
    """
    EDSR Model with an upsampler head on top (a final layer on top of the feature maps), for image super resolution and restoration.
    """,
    EDSR_START_DOCSTRING,
)
class EDSRForImageSuperResolution(EDSRPreTrainedModel):
    def __init__(self, config):
        super(EDSRPreTrainedModel, self).__init__(config)
        self.edsr_model = EDSRModel(config)

        # define tail module
        kernel_size = 3
        edsr_tail = [
            EDSRUpsampler(default_conv, config.upscale, config.num_feature_maps, activation=None),
            default_conv(config.num_feature_maps, config.num_channels, kernel_size),
        ]

        self.upsampler = nn.Sequential(*edsr_tail)

    @add_start_docstrings_to_model_forward(EDSR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageSuperResolutionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, ImageSuperResolutionOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import EDSRImageProcessor, EDSRForImageSuperResolution
         >>> from datasets import load_dataset

         >>> processor = EDSRImageProcessor.from_pretrained("asrimanth/edsr-base-x2")
         >>> model = EDSRForImageSuperResolution.from_pretrained("asrimanth/edsr-base-x2")
         ```"""

        pixel_values = self.edsr_model(pixel_values)[0]
        pixel_values = self.upsampler(pixel_values)
        pixel_values = self.edsr_model.add_mean(pixel_values)

        return ImageSuperResolutionOutput(
            reconstruction=pixel_values,
        )
