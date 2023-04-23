# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert EDSR checkpoints from the original repository. URL: https://github.com/sanghyun-son/EDSR-PyTorch
"""

import argparse

import numpy as np
import requests
import torch
import torchvision
from PIL import Image

from transformers import EDSRConfig, EDSRForImageSuperResolution, EDSRImageProcessor


def get_edsr_config(checkpoint_url):
    config = EDSRConfig()
    print(checkpoint_url)
    if "edsr_baseline_x2" in checkpoint_url:
        config.n_resblocks = 16
        config.n_feats = 64
        config.upscale = 2
    elif "edsr_baseline_x3" in checkpoint_url:
        config.n_resblocks = 16
        config.n_feats = 64
        config.upscale = 3
    elif "edsr_baseline_x4" in checkpoint_url:
        config.n_resblocks = 16
        config.n_feats = 64
        config.upscale = 4
    elif "edsr_x2" in checkpoint_url:
        config.n_resblocks = 32
        config.n_feats = 256
        config.upscale = 2
    elif "edsr_x3" in checkpoint_url:
        config.n_resblocks = 32
        config.n_feats = 256
        config.upscale = 3
    elif "edsr_x4" in checkpoint_url:
        config.n_resblocks = 32
        config.n_feats = 256
        config.upscale = 4

    return config


def rename_key(name):
    if "model" in name:
        name = name.replace("model", "edsr_model")
    if "head" in name:
        name = name.replace("head", "edsr_head")
    if "body" in name:
        name = name.replace("body", "edsr_body")
    if "tail" in name:
        name = name.replace("tail", "upsampler")

    return name


def load_sample_image(image_url):
    # url = "https://lh4.googleusercontent.com/-Anmw5df4gj0/AAAAAAAAAAI/AAAAAAAAAAc/6HxU8XFLnQE/photo.jpg64"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image.save("input_image.png")
    image = torchvision.transforms.functional.pil_to_tensor(image).float().unsqueeze(0)
    return image


def save_image(out_tensor, save_path="model_out_pil.png"):
    numpy_image = torch.clip(out_tensor.squeeze(0), 0, 255)
    numpy_image = numpy_image.cpu().detach().numpy().transpose(1, 2, 0)
    out_image = torchvision.transforms.functional.to_pil_image(np.uint8(numpy_image))
    out_image.save(save_path)


@torch.no_grad()
def convert_edsr_checkpoint(checkpoint_url: str, pytorch_dump_folder_path: str, push_to_hub: bool):
    config = get_edsr_config(checkpoint_url)

    SAMPLE_IMAGE_URL = "https://lh4.googleusercontent.com/-Anmw5df4gj0/AAAAAAAAAAI/AAAAAAAAAAc/6HxU8XFLnQE/photo.jpg64"

    url_to_slice_dict = {
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt": torch.tensor(
            [[146.6154, 128.6703, 73.1800], [112.6714, 90.5857, 68.5255], [56.6039, 62.8006, 77.7706]]
        ),
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt": torch.tensor(
            [[139.0591, 136.4635, 126.6529], [133.7725, 126.7542, 110.9417], [108.5025, 99.2904, 80.3906]]
        ),
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt": torch.tensor(
            [[143.0898, 140.2990, 135.3582], [140.3488, 135.8599, 128.9865], [128.6764, 121.7250, 111.3058]]
        ),
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt": 0,
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt": 0,
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt": 0,
    }

    model = EDSRForImageSuperResolution(config)
    model.eval()

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")

    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # add prefix to all keys except the head
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith("upsampler"):
            key = "edsr_model." + key
        state_dict[key] = val

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

    print("Missing:", missing_keys)
    print("Unexpected:", unexpected_keys)

    if len(missing_keys) > 0:
        raise ValueError("Missing keys when converting: {}".format(missing_keys))
    for key in unexpected_keys:
        if not ("relative_position_index" in key or "relative_coords_table" in key or "self_mask" in key):
            raise ValueError(f"Unexpected key {key} in state_dict")

    # verify values
    EDSRImageProcessor()
    pixel_values = load_sample_image(SAMPLE_IMAGE_URL)

    # pixel_values = processor(image).pixel_values

    # pixel_values = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
    print(pixel_values.shape)
    outputs = model(pixel_values)
    print(outputs.reconstruction.shape)
    save_image(outputs.reconstruction)
    # assert values
    expected_slice = url_to_slice_dict[checkpoint_url]
    expected_shape = torch.Size(
        [1, 3, pixel_values.shape[-2] * config.upscale, pixel_values.shape[-1] * config.upscale]
    )
    print("Shape of reconstruction:", outputs.reconstruction.shape)
    print("Actual values of the reconstruction:\n", outputs.reconstruction[0, 0, :3, :3])

    assert (
        outputs.reconstruction.shape == expected_shape
    ), f"Shape of reconstruction should be {expected_shape}, but is {outputs.reconstruction.shape}"
    assert torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=1e-3)
    print("Looks ok!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt",
        type=str,
        help="URL of the original EDSR checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the hub.")

    args = parser.parse_args()
    convert_edsr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
