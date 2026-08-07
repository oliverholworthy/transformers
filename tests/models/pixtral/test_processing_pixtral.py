# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import re
import unittest

import numpy as np
import torch
from parameterized import parameterized

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import PixtralProcessor


@require_vision
class PixtralProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PixtralProcessor
    tiny_model_id = "hf-internal-testing/tiny-processor-pixtral"
    model_id = "mistral-community/pixtral-12b"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.url_0 = url_to_local_path(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
        )
        cls.image_0 = np.random.randint(255, size=(3, 876, 1300), dtype=np.uint8)
        cls.url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        cls.image_1 = np.random.randint(255, size=(3, 480, 640), dtype=np.uint8)
        cls.image_2 = np.random.randint(255, size=(3, 1024, 1024), dtype=np.uint8)
        cls.image_token = processor.image_token

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        processor = super()._setup_from_pretrained(model_id, **kwargs)
        processor.tokenizer.pad_token_id = 0  # loaded tokenizer has no PAD defined
        return processor

    @parameterized.expand([(1, "pt"), (2, "pt")])
    @unittest.skip("Not tested before, to investigate")
    def test_apply_chat_template_image(self, batch_size, return_tensors):
        pass

    def test_image_token_filling(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 500, 316))
        expected_image_tokens = 640
        image_token_index = 10

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        inputs = processor(
            text=[processor.apply_chat_template(messages)],
            images=[image],
            return_tensors="pt",
        )
        image_tokens = (inputs["input_ids"] == image_token_index).sum().item()
        self.assertEqual(expected_image_tokens, image_tokens)

    def test_from_pretrained_subfolder_tokenizer(self):
        processor = PixtralProcessor.from_pretrained("hf-internal-testing/tiny-flux2", subfolder="tokenizer")
        self.assertIsInstance(processor, PixtralProcessor)
        self.assertIsNotNone(processor.tokenizer)

    def test_tokenizers_backend_preserves_image_id_parity_when_splitting_special_tokens(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        processor.tokenizer.split_special_tokens = True
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        prompt = "USER: [IMG] describe it"
        inputs = processor(text=prompt, images=self.image_0, return_tensors=None)
        flat_expansion = "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END] describe it"
        expected = processor.tokenizer(
            flat_expansion,
            add_special_tokens=True,
            split_special_tokens=False,
            return_tensors=None,
        )

        self.assertEqual(inputs["input_ids"], [expected["input_ids"]])

    def test_tokenizers_backend_does_not_activate_unconsumed_literal_image_markers(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        processor.tokenizer.split_special_tokens = True
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}
        prompt = "Image: [IMG]. Literal text: [IMG] and [IMG_END]."

        processor.tokenizer.split_special_tokens = False
        baseline = processor(text=prompt, images=self.image_0, return_tensors=None)
        self.assertEqual(baseline["input_ids"][0].count(processor.image_token_id), 5)
        self.assertEqual(baseline["input_ids"][0].count(processor.image_end_token_id), 2)

        processor.tokenizer.split_special_tokens = True
        inputs = processor(text=prompt, images=self.image_0, return_tensors=None)
        input_ids = inputs["input_ids"][0]
        self.assertEqual(input_ids.count(processor.image_token_id), 4)
        self.assertEqual(input_ids.count(processor.image_break_token_id), 1)
        self.assertEqual(input_ids.count(processor.image_end_token_id), 1)

    def test_flat_text_uses_the_first_image_marker_as_the_placeholder(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        processor.tokenizer.split_special_tokens = True
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}
        prompt = "Literal first [IMG], intended placeholder [IMG]"

        inputs = processor(
            text=prompt,
            images=self.image_0,
            return_tensors=None,
            return_text_replacement_offsets=True,
        )

        # Flat text carries no ownership metadata, so its first marker is the processor-designated placeholder.
        self.assertEqual(inputs["text_replacement_offsets"][0][0]["span"], (14, 19))

    def test_chat_template_preserves_image_token_ownership(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        processor.tokenizer.split_special_tokens = True
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Literal first [IMG]. "},
                    {"type": "image", "image": self.image_0},
                    {"type": "text", "text": " Literal last [IMG] and [IMG_END]."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors=None,
            processor_kwargs={"return_text_replacement_offsets": True},
        )

        input_ids = inputs["input_ids"][0].tolist()
        self.assertEqual(input_ids.count(processor.image_token_id), 4)
        self.assertEqual(input_ids.count(processor.image_break_token_id), 1)
        self.assertEqual(input_ids.count(processor.image_end_token_id), 1)
        self.assertEqual(len(inputs["text_replacement_offsets"][0]), 1)
        rendered_prompt = processor.apply_chat_template(messages, tokenize=False)
        owned_start = rendered_prompt.find(processor.image_token, rendered_prompt.find(processor.image_token) + 1)
        self.assertEqual(
            inputs["text_replacement_offsets"][0][0]["span"],
            (owned_start, owned_start + len(processor.image_token)),
        )

    def test_chat_template_preserves_image_token_ownership_in_batches(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        processor.tokenizer.split_special_tokens = True
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Literal first [IMG]. "},
                        {"type": "image", "image": self.image_0},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Literal before [IMG]. "},
                        {"type": "image", "image": self.image_1},
                        {"type": "text", "text": " Literal between [IMG]. "},
                        {"type": "image", "image": self.image_2},
                    ],
                }
            ],
        ]

        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors=None,
            processor_kwargs={"return_text_replacement_offsets": True, "padding": True},
        )

        self.assertEqual([len(offsets) for offsets in inputs["text_replacement_offsets"]], [1, 2])
        for batch_idx, (rendered_prompt, offsets) in enumerate(
            zip(processor.apply_chat_template(conversations, tokenize=False), inputs["text_replacement_offsets"])
        ):
            marker_starts = [match.start() for match in re.finditer(re.escape(processor.image_token), rendered_prompt)]
            expected_starts = marker_starts[1::2] if batch_idx else marker_starts[1:]
            self.assertEqual(
                [offset["span"] for offset in offsets],
                [(start, start + len(processor.image_token)) for start in expected_starts],
            )

    def test_chat_template_fails_closed_when_literal_ownership_cannot_be_tracked(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        processor.tokenizer.split_special_tokens = True
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Literal [IMG]."},
                    {"type": "image", "image": self.image_0},
                ],
            }
        ]
        transforming_template = "{% for block in messages[0]['content'] %}{% if block['type'] == 'text' %}{{ block['text'] | lower }}{% else %}[IMG]{% endif %}{% endfor %}"

        with self.assertRaisesRegex(ValueError, "ownership cannot be tracked safely"):
            processor.apply_chat_template(
                messages,
                chat_template=transforming_template,
                tokenize=True,
                return_dict=True,
            )

    def test_tokenizers_backend_rejects_partial_image_truncation(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        processor.tokenizer.split_special_tokens = True
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        with self.assertRaisesRegex(ValueError, "Truncation would split a processor-owned special-token span"):
            processor(
                text="[IMG] trailing text",
                images=self.image_0,
                truncation=True,
                max_length=3,
                return_tensors=None,
            )

        inputs = processor(
            text="[IMG] trailing text",
            images=self.image_0,
            truncation=True,
            max_length=6,
            return_tensors=None,
        )
        self.assertEqual(
            inputs["input_ids"],
            [[10, 10, 12, 10, 10, 13]],
        )

    def test_processor_with_single_image(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        prompt_string = "USER: [IMG]\nWhat's the content of the image? ASSISTANT:"

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=self.image_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=self.url_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test passing inputs as a single list
        inputs_image = processor(text=prompt_string, images=[self.image_0], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test as nested single list
        inputs_image = processor(text=prompt_string, images=[[self.image_0]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

    def test_processor_with_multiple_images_single_list(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        prompt_string = "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:"

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=[self.image_0, self.image_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
                    )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=[self.url_0, self.url_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in as a nested list
        inputs_url = processor(text=prompt_string, images=[[self.image_0, self.image_1]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_url["input_ids"][0].tolist(),
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

    def test_processor_with_multiple_images_multiple_lists(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        prompt_string = [
            "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:",
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ]
        processor.tokenizer.pad_token = "</s>"
        image_inputs = [[self.image_0, self.image_1], [self.image_2]]

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 2)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 2)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing as a single flat list
        inputs_image = processor(
            text=prompt_string, images=[self.image_0, self.image_1, self.image_2], return_tensors="pt", padding=True
        )
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

    def test_processor_returns_full_length_batches(self):
        # to avoid https://github.com/huggingface/transformers/issues/34204
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = [
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ] * 5
        processor.tokenizer.pad_token = "</s>"
        image_inputs = [[self.image_0]] * 5

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 5)
        self.assertTrue(len(inputs_image["pixel_values"]) == 5)
