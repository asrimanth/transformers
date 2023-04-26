# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch EDSR model. """
import inspect
import unittest
import requests

from transformers import EDSRConfig, EDSRModel
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor

if is_torch_available():
    import torch
    from torch import nn

    from transformers import EDSRForImageSuperResolution, EDSRModel, EDSRImageProcessor
    from transformers.models.edsr.modeling_edsr import EDSR_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image


class EDSRModelTester:
    def __init__(
        self,
        upscale: int = 2,
        num_channels: int = 3,
        batch_size: int = 8,
        image_size: int = 128,
        hidden_act: str = "relu",
        num_res_block: int = 16,
        num_feature_maps: int = 64,
        res_scale: int = 1,
        rgb_range: int = 255,
        rgb_mean: list = [0.4488, 0.4371, 0.4040],
        rgb_std: list = [1.0, 1.0, 1.0],
        **kwargs,
    ):
        self.upscale = upscale
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.image_size = image_size
        self.hidden_act = hidden_act
        self.num_res_block = num_res_block
        self.num_feature_maps = num_feature_maps
        self.res_scale = res_scale
        self.rgb_range = rgb_range
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return EDSRConfig(
            upscale=self.upscale,
            num_channels=self.num_channels,
            hidden_act=self.hidden_act,
            num_res_block=self.num_res_block,
            num_feature_maps=self.num_feature_maps,
            res_scale=self.res_scale,
            rgb_range=self.rgb_range,
            rgb_mean=self.rgb_mean,
            rgb_std=self.rgb_std,
        )

    def create_and_check_model(self, config, pixel_values):
        model = EDSRModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.num_feature_maps, self.image_size, self.image_size)
        )

    def create_and_check_for_image_super_resolution(self, config, pixel_values):
        model = EDSRForImageSuperResolution(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        self.parent.assertEqual(
            result.reconstruction.shape, 
            (self.batch_size, self.num_channels, self.image_size * self.upscale, self.image_size * self.upscale)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class EDSRModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (EDSRModel, EDSRForImageSuperResolution) if is_torch_available() else ()

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = EDSRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EDSRConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_for_image_super_resolution(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_super_resolution(*config_and_inputs)

    @unittest.skip(reason="EDSR does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EDSR does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="EDSR does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="EDSRNet does not support input and output embeddings")
    def test_model_common_attributes(self):
        # config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # for model_class in self.all_model_classes:
        #     model = model_class(config)
        #     self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
        #     x = model.get_output_embeddings()
        #     self.assertTrue(x is None or isinstance(x, nn.Linear))
        pass

    @unittest.skip(reason="EDSRNet does not support attention")
    def test_attention_outputs(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        for model_name in EDSR_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = EDSRModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    # overwriting because of `logit_scale` parameter
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "logit_scale" in name:
                    continue
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


@require_vision
@require_torch
class EDSRModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_super_resolution_head(self):
        # TODO update to appropriate organization
        image_url = "https://lh4.googleusercontent.com/-Anmw5df4gj0/AAAAAAAAAAI/AAAAAAAAAAc/6HxU8XFLnQE/photo.jpg64"
        url_to_slice_dict = {
            "asrimanth/edsr-base-x2": torch.tensor(
                [[1.0845, 0.1934, 0.1843],[0.6235, 0.2663, 0.0347],[0.5703, -0.0288,  0.1822]]
            ),
            "asrimanth/edsr-base-x3": torch.tensor(
                [[1.8288, 0.5871, 0.2986],[0.7461, 0.6465, 0.1928],[0.4508, 0.3917, 0.3607]]
            ),
            "asrimanth/edsr-base-x4": torch.tensor([
                [1.6611, 0.6265, 0.3864],[0.9503, 0.3109, 0.1029],[0.5610, 0.3287, 0.4297]]
            ),
        }

        for repo_url, expected_slice in url_to_slice_dict.items():
            model = EDSRForImageSuperResolution.from_pretrained(repo_url).to(torch_device)
            processor = EDSRImageProcessor.from_pretrained(repo_url)

            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            # image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
            inputs = processor(images=image, return_tensors="pt").to(torch_device)

            # forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            # verify the slices
            upscale = int(repo_url[-1])
            expected_shape = inputs.pixel_values.shape
            expected_shape = torch.Size([expected_shape[0], expected_shape[1], expected_shape[2] * upscale, expected_shape[3] * upscale])

            self.assertEqual(outputs.reconstruction.shape, expected_shape)
            expected_slice = expected_slice.to(torch_device)
            print(inputs.pixel_values[0, 0, :3, :3], outputs.reconstruction[0, 0, :3, :3], expected_slice)
            self.assertTrue(torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=1e-4))
            break
