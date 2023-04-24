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

    url_to_name_dict = {
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt": "edsr-base-x2",
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt": "edsr-base-x3",
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt": "edsr-base-x4",
    }
    url_to_slice_dict = {
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt": torch.tensor(
            [[ 1.0845,  0.1934,  0.1843],[0.6235,  0.2663,  0.0347],[0.5703, -0.0288,  0.1822]]
        ),
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt": torch.tensor(
            [[1.8288, 0.5871, 0.2986],[0.7461, 0.6465, 0.1928],[0.4508, 0.3917, 0.3607]]
        ),
        "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt": torch.tensor([
            [1.6611, 0.6265, 0.3864],[0.9503, 0.3109, 0.1029],[0.5610, 0.3287, 0.4297]]
        ),
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
    processor = EDSRImageProcessor()
    pixel_values = load_sample_image(SAMPLE_IMAGE_URL)
    pixel_values = processor(pixel_values, return_tensors="pt").pixel_values

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

    model_name = url_to_name_dict[checkpoint_url]
    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"asrimanth/{model_name}")
        processor.push_to_hub(f"asrimanth/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
        type=str,
        help="URL of the original EDSR checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the hub.")

    args = parser.parse_args()
    convert_edsr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
