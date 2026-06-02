import base64
import copy
import math
import os
import pickle
import random
from io import BytesIO
from typing import Optional

import cv2
import lmdb
import numpy as np
from omegaconf import DictConfig, ListConfig
from PIL import Image
from qwen_vl_utils import smart_resize
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from .utils import MAX_VISUAL_TOKENS, MIN_VISUAL_TOKENS

# ANNO_ID_MATCHING_TEMPLATE = (
#     "Given two images of one same scene, annotated with some ids on some regions, identify the regions' matching relations. "
#     "Please first describe the visual features of the marked regions in each images, then find the matching relations between the regions across images of the same scene. "
#     "Analyze carefully and output the thinking process between <think> </think> tags, and then give the final answer in JSON format between <answer> </answer> tags. "
#     "e.g. <think> your thinking process here </think> "
#     '<answer> {"1": "3", "2": "5"} </answer> '
# )


ANNO_MATCH_TEMPLATE = r"""You are given two images of the same physical scene, each annotated with several region IDs (e.g., "1", "2", "3").

Your task is to identify the one-to-one correspondence between regions in the two images. Every region in the first image has exactly one match in the second.

Please provide your response in **two parts**: `thinking process` and `final answer`.

*  **`thinking process`**: A piece of sentences wrapped by <thinking> </thinking> tags where you explain your thinking process step-by-step:
    1. **Describe Visual Regions**: For each annotated ID in both images, describe the key visual characteristics, including:
    - color, shape, texture
    - relative position in the image (e.g., top-left, center-right)
    - nearby reference objects (e.g., “near red door”)

    2. **Compare Viewpoints**: Analyze the geometric or perspective relationship between the two images, such as:
    - rotation (e.g., “B is 90° clockwise rotated from A”)
    - camera shift, zoom, scale difference
    - occlusion or deformation between views

    3. **Find Matching Relations**: Based on region appearance, position consistency, and scene layout, infer which regions in image A match those in image B.


*  **`final answer`**: A JSON object wrapped by <answer> </answer> tags where you give the final answer based on thinking process. The JSON object is a mapping of region IDs from image A (as string keys) to the corresponding region IDs in image B (as string values).


**Example of the required output format:**

<thinking>

"Describe your reasoning process step-by-step here. Including but not limited to the visual descriptions, the viewpoint analysis, and the final alignment decisions. Do not skip steps."

</thinking>
<answer>

```json
{ "1": "2",  "3": "1", "4": "3", ..., }
```

</answer>

"""

# ANNO_MATCH_MULTI_TEMPLATE = r"""You are given two images of the same physical scene, each annotated with several region IDs (e.g., "1", "2", "3").

# Your task is to identify the underlying correspondence between regions in the two images. Note that **maybe not** every region in the first image has match in the second.

# Please provide your response in **two parts**: `thinking process` and `final answer`.

# *  **`thinking process`**: Your analysis where you explain your analyzing and thinking process, which includes but not limited to four aspects:
#     1. **Describe Visual Regions**: For each annotated ID in both images, describe the key visual characteristics, including as least:
#     - color, shape, texture
#     - relative position in the image (e.g., top-left, center-right)
#     - nearby reference objects (e.g., “near red door”)
#     - spatial relation to other annotated regions

#     2. **Compare Viewpoints**: Analyze the geometric or perspective relationship between the two images, such as:
#     - overview of the two images' contents, and the similarity as well as differences.
#     - rotation (e.g., “B is 90° clockwise rotated from A”)
#     - reasoning about possible camera shift, zoom, scale difference
#     - occlusion or deformation between views

#     3. **Infer Matching Relations**: Based on region appearance, position consistency, and scene layout, infer which regions in image A match those in image B.

#     4. **Reflection**: Consider the implications of the identified correspondences:
#     - check the consistency of the inferred local region matches and the view change inferred from global scene differences. Are all the findings consistent to the scene view point changes?
#     - confirm the regions one-by-one you thought having no matches on the other image. Is the mismatch obvious, or just ambiguous that you're uncertain? Maybe re-consider ambiguous regions.


# *  **`final answer`**: Your careful answer based on analysis and reflection process in your thinking part, which is wrapped by <answer> </answer> tags where you give the final answer based on thinking process. The JSON object is a mapping of region IDs from image A (as string keys) to the corresponding region IDs in image B (as string values). If a region in image A has no corresponding region in image B, please map it to "none".


# **Example of the required output format which you should follow:**

# "Describe your reasoning process step-by-step here. You can make additional analysis not mentioned above, but do not skip steps."
# <answer>

# ```json
# { "1": "2",  "3": "1", "4": "none", ..., }
# ```

# </answer>

# """

ANNO_MATCH_MULTI_TEMPLATE = r"""You are given two images of the same physical scene, each having several regions annotated with circles and IDs (e.g., "1", "2", "3").

Your task is to identify the underlying correspondence between regions in the two images. Note that **maybe not** every region in the first image has a match in the second.

Please provide your response in **two parts**: `thinking process` and `final answer`, each wrapped by some special tags.

*  **`thinking process`**: Your analysis where you show your analyzing and thinking process **wrapped in <thinking> </thinking> tags**, which includes but not limited to:
    
    1. **Describe Visual Regions**: For each annotated area in both images, describe the key visual characteristics and **spatial context within the scene**, including:
    - **Intrinsic properties**: color, shape, texture, size, material
    - **Spatial relationships in the 3D scene** (NOT pixel coordinates):
        * What objects/structures are directly adjacent to this region? (e.g., "attached to a wooden door", "sitting on a metal shelf")
        * What is this region positioned relative to in the physical space? (e.g., "below the window", "behind the chair", "left side of the bookshelf")
        * Semantic context: What functional area or object group does it belong to? (e.g., "part of the dining area", "on the workspace desk")
    - **Avoid** describing regions by their pixel locations (top-left, center-right, etc.) unless necessary for disambiguation
    - Focus on **scene-level landmarks** as reference points (e.g., "near the entrance", "opposite to the main table", "in the corner with the lamp")

    2. **Compare Viewpoints**: Analyze the geometric or perspective relationship between the two images, such as:
    - Overview of the two images' contents, focusing on **how the physical scene layout appears** in each view
    - **Camera transformation**: rotation angle (e.g., "camera rotated ~90° clockwise around the room center"), translation (e.g., "camera moved closer to the left wall"), zoom/scale differences
    - **Occlusion changes**: which objects/regions become visible or hidden due to viewpoint change?
    - **Perspective distortion**: how do spatial relationships appear to change due to different viewing angles? (e.g., "objects on the right side now appear more frontal")

    3. **Infer Matching Relations**: Based on region appearance and **scene-relative spatial relationships**, establish correspondences:
    - Iterate through each annotated region in image 1, and for each one, compare it sequentially with every annotated region in image 2 to infer the matching likelihood and reasonableness of region correspondences between the two images
    - Prioritize matching based on **what surrounds each region** and **3D spatial context** rather than 2D image positions
    - Use **stable scene anchors** (walls, large furniture, architectural features) to reason about region identity across views
    - Consider how the viewpoint change transforms the **spatial relationships** you identified in step 1
    - Example reasoning: "Region A-1 is next to a red door and below a window. Region B-3 is also adjacent to the same red door (now seen from a different angle) and below the same window structure, so A-1 matches B-3"


*  **`final answer`**: Your final answer based on your thinking part **wrapped by <answer> </answer> tags**. The JSON object is a mapping of region IDs from first image (as string keys) to the corresponding region IDs in the second image (as string values). **For regions in the first image that have no match in the second image, use "none" as the value.**


**Example of the required output format which you should follow:**

<thinking>

"Describe your reasoning process step-by-step here but **do not repeatedly analyze**, focusing on scene-level spatial relationships and context. Identify what each region is near/attached to/part of in the physical scene, then use these spatial anchors to establish correspondences across viewpoints."

</thinking>
<answer>

```json
{ "1": "2",  "2": "none", "3": "1", "4": "3" }
```
</answer>

"""


EXAMPLE_JSON = r"""
```json
[
    {"point_2d": [100, 100], "label": "1"},
    {"point_2d": [200, 200], "label": "2"},
    {"point_2d": [300, 300], "label": "3"}
]
```
"""

# ANNO_GROUND_TEMPLATE = (
#     "Given two images of one same scene, with one annotated with some ids on some regions, "
#     "please identify the corresponding regions on the other image, and output the 2d coordinates in json format. "
#     "**To find the corresponding regions**, please first analyze the difference between the two images, "
#     "since the most distinguished difference is camera movement. "
#     "Then describe the visual features of the marked regions in the first image "
#     "and find the matching candidate regions on the second image. "
#     "Finally, for each region in the first image, output the corresponding 2d coordinates with id on the second image. "
#     "Analyze carefully and output the thinking process between <think> </think> tags, "
#     "and then give the final answer in JSON format between <answer> </answer> tags, label is the id of the region in the first image. "
#     "e.g. <think> your thinking process here </think> "
#     f"<answer> {EXAMPLE_JSON} </answer> "
# )

# ANNO_GROUND_TEMPLATE = (
#     "You are given two images of the same scene. The first image contains annotations with numerical IDs "
#     "marking specific regions. Your task is to identify the corresponding regions in the second image "
#     "and output their 2D coordinates in JSON format.\n\n"
#     "Please provide your response in **two parts**: `thinking process` and `final answer`.\n"

#     "*  **`thinking process`**: A piece of sentences wrapped by <think> </think> tags where you explain your thinking process step-by-step:\n"
#     "**Step-by-step approach:**\n"
#     "1. **Analyze the relationship between images**: Identify any differences such as camera movement, "
#     "rotation, zoom level, perspective changes, or lighting variations.\n\n"

#     "2. **Describe annotated regions**: For each marked region in the first image, describe its:\n"
#     "   - Visual appearance (color, texture, shape)\n"
#     "   - Distinctive features or patterns\n"
#     "   - Spatial relationships with nearby objects or landmarks\n\n"

#     "3. **Locate corresponding regions**: Using the descriptions and understanding of image differences, "
#     "find the matching regions in the second image.\n\n"

#     "4. **Determine coordinates**: For each region, identify the 2D pixel coordinates (x, y) that best "
#     "represent the region's location in the second image (e.g., center point or bounding box).\n\n"

#     "*  **`final answer`**: A JSON object wrapped by <answer> </answer> tags where you give the final answer based on thinking process. The JSON object is a list of coordinates with labels corresponding to the region IDs from the first image.\n\n"

#     "**Output format:**\n"
#     "- Enclose your detailed reasoning process in <think></think> tags\n"
#     "- Provide the final answer in valid JSON format within <answer></answer> tags\n\n"

#     "**Example of the required output format:**\n"

#     "<think>\n"
#     "Image analysis: The second image appears to be taken from a slightly higher angle with the camera "
#     "shifted 50 pixels to the right...\n"
#     "Region ID 1: This is a red car in the first image, located at the center-left. In the second image, "
#     "due to the camera shift, it appears at coordinates (320, 180)...\n"
#     "</think>\n\n"
#     f"<answer>\n{EXAMPLE_JSON}\n</answer>"
# )

ANNO_GROUND_TEMPLATE = r"""You are given two images of the same physical scene. The first image contains annotations with numerical IDs marking specific regions. Your task is to identify the corresponding locations in the second image and output their 2D pixel coordinates.

**Coordinate Convention**: Origin (0, 0) is at top-left corner. Output the center point of each region.

Please provide your response in **two parts**: `thinking process` and `final answer`.

*  **`thinking process`**: A piece of sentences wrapped by <think> </think> tags where you explain your thinking process step-by-step:
    1. **Describe Annotated Regions**: For each marked region in the first image, describe:
    - visual characteristics (color, shape, texture, distinctive features)
    - spatial position (e.g., top-left, center-right)
    - nearby reference objects or landmarks
    
    2. **Analyze Image Relationship**: Identify the geometric relationship between the two images:
    - camera movement (left/right shift, up/down shift)
    - rotation or tilt
    - zoom or scale changes
    - perspective differences
    
    3. **Locate Corresponding Regions**: Based on visual features and the image relationship, find each region in the second image and estimate its center coordinates.

*  **`final answer`**: A JSON object wrapped by <answer> </answer> tags where you give the final answer based on thinking process. The JSON object is a list of coordinates with labels corresponding to the region IDs from the first image.

**Example of the required output format:**

<think> 

"Describe your reasoning process step-by-step here. Include the visual descriptions, the viewpoint analysis, and the final decisions. Do not skip steps.",

</think>

<answer>
```json
[
    {"point_2d": [100, 100], "label": "1"},
    {"point_2d": [200, 200], "label": "2"},
    {"point_2d": [300, 300], "label": "3"}
]
```
</answer>

"""


# ANNO_GROUND_TEMPLATE = r"""You are given two images of the same physical scene. The first image contains annotations with numerical IDs marking specific regions. Your task is to identify the corresponding regions in the second image and output their 2D pixel coordinates in JSON format.

# **Coordinate System:**
# - Origin (0, 0) is at the top-left corner
# - X increases to the right, Y increases downward
# - Output coordinates should represent the CENTER of each marked region

# **Analysis Framework:**

# 1. **Identify geometric transformations** between images:
#    - Camera translation (left/right, up/down, forward/backward)
#    - Rotation or tilt angles
#    - Zoom/scale changes
#    - Perspective shift effects

# 2. **For each annotated region**, analyze systematically:
#    - **Appearance**: Color, texture, shape, size, brightness
#    - **Context**: What objects/structures surround it?
#    - **Spatial anchors**: Distance and direction from stable landmarks
#    - **Distinctive features**: Unique patterns, edges, corners, or markers

# 3. **Establish correspondence** by:
#    - Using epipolar geometry constraints if applicable
#    - Tracking consistent feature patterns across both views
#    - Verifying that spatial relationships between regions are preserved
#    - Cross-checking multiple regions for geometric consistency

# 4. **Handle edge cases**:
#    - If uncertain about location, use your best estimate and explain uncertainty
#    - If multiple candidates exist, choose the most geometrically consistent match

# **Required Output Structure:**

# **Part 1 - Thinking Process** (enclosed in <think></think> tags):
# Provide detailed reasoning including:
# - Overall geometric relationship between the two images
# - For EACH region ID sequentially:
#   • Visual characteristics observed in image 1
#   • How you located it in image 2 (or why you couldn't find it)
#   • Coordinate estimation process and confidence level

# **Part 2 - Final Answer** (enclosed in <answer></answer> tags):
# - Valid JSON array with objects containing:
#   • "point_2d": [x, y] as integers, or null if region not found
#   • "label": The region ID from image 1 (as string)
# - Maintain the same order as regions appear in image 1

# **Example Output Format:**

# <think>
# Geometric Analysis: Image 2 appears to be captured from approximately 30 degrees to the right with slight upward tilt. Estimated horizontal shift of ~50 pixels and 10-degree rotation observed from comparing background structures.

# Region 1 Analysis (marked in top-left area of image 1):
# - Visual features: Red vehicle with distinctive rectangular shape and chrome bumper
# - Spatial context: Located adjacent to tall lamp post, approximately 40% from left edge
# - Corresponding location in image 2: Due to rightward camera shift, the vehicle has moved toward center-left
# - Estimated coordinates: (100, 150) at vehicle's center
# - Confidence: High - clear distinctive color and shape, consistent geometric transformation

# Region 2 Analysis (marked in center of image 1):
# - Visual features: Blue rectangular signage with white text pattern
# - Spatial context: Positioned on building facade, aligned with window row
# - Corresponding location in image 2: Shifted right and slightly down due to camera movement
# - Estimated coordinates: (250, 280)
# - Confidence: High - text pattern clearly visible and spatial relationships maintained

# Region 3 Analysis (marked in right portion of image 1):
# - Visual features: Third-floor window with distinctive arch shape
# - Spatial context: Upper-right quadrant, part of building structure
# - Status in image 2: Appears to be outside the frame boundary due to camera rotation
# - Coordinates: null (not visible in second image)
# - Confidence: Not applicable - region confirmed out of frame
# </think>

# <answer>
# ```json
# [
#     {"point_2d": [100, 100], "label": "1"},
#     {"point_2d": [200, 200], "label": "2"},
#     {"point_2d": [300, 300], "label": "3"}
# ]
# ```
# </answer>

# **Important Guidelines:**
# - Always provide entries for ALL region IDs from image 1.
# - Coordinate estimates should be precise (aim for within 10-20 pixels of true center)
# - Prioritize geometric consistency over individual feature matching alone
# - Use stable background features as reference points when possible
# - If genuinely uncertain, explain your reasoning clearly in the thinking process
# """


class AnnoDB(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # @zhonghao, qwen2_5_vl and qwen3_vl have different visual patch size
        self.visual_patch_size = processor.image_processor.patch_size
        self.visual_merge_size = processor.image_processor.merge_size
        self.visual_temporal_patch_size = processor.image_processor.temporal_patch_size

        self.min_pixels = MIN_VISUAL_TOKENS * ((self.visual_patch_size * self.visual_merge_size) ** 2)
        self.max_pixels = MAX_VISUAL_TOKENS * ((self.visual_patch_size * self.visual_merge_size) ** 2)

        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        assert len(data_files) == 1, "Currently only support one lmdb dataset"
        self.data_files = copy.deepcopy(data_files[0])
        self.original_data_files = copy.deepcopy(data_files)  # use for resume

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        max_sample = config.get("max_sample", None)

        with lmdb.open(
            str(self.data_files),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        ) as handle:
            # Get dataset size
            with handle.begin() as txn:
                meta_value = txn.get(b"__meta__")
                if meta_value is None:
                    self.meta_data = {"num_samples": 500}
                    self.num_samples = 500
                    # raise ValueError("No metadata found in LMDB")
                else:
                    self.meta_data = pickle.loads(meta_value)
                    self.num_samples = (
                        self.meta_data["num_samples"]
                        if max_sample is None
                        else min(max_sample, self.meta_data["num_samples"])
                    )

        self.post_init()

    def post_init(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_samples

    def _get_db(self):
        if self.datadb:
            return self.datadb
        else:
            self.datadb = lmdb.open(
                str(self.data_files),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            return self.datadb

    def make_openai_messages(self, messages: list[dict]):
        def encode_image(image: str | Image.Image) -> str:
            buffer = BytesIO()
            if isinstance(image, str):
                Image.open(image).save(buffer, format="JPEG")
            elif isinstance(image, Image.Image):
                image.save(buffer, format="JPEG")
            else:
                raise ValueError("Image should be a file path or PIL Image object.")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

        openai_messages = []
        for msg in messages:
            if msg["role"] == "user":
                content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        content.append(item)
                    elif item["type"] == "image":
                        assert not isinstance(item["image"], list), (
                            "processed messages should not contain list of images"
                        )

                        base64_image = encode_image(item["image"])
                        content.append(
                            {
                                "type": "image_url",
                                "min_pixels": item["min_pixels"],
                                "max_pixels": item["max_pixels"],
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            }
                        )
                openai_messages.append({"role": "user", "content": content})

            elif msg["role"] in ["assistant", "system"]:
                openai_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                raise ValueError(f"Unknown role {msg['role']}")
        return openai_messages

    def make_messages(self, messages: list[dict]):
        prompt_messages = []
        total_turn = len(messages)
        for i, msg in enumerate(messages):
            min_pixels = self.history_min_pixels if i < total_turn - 2 else self.min_pixels
            max_pixels = self.history_max_pixels if i < total_turn - 2 else self.max_pixels

            if msg["role"] == "user":
                if msg["content"].get("images", None) is not None:
                    text = msg["content"].get("text", None)
                    content = []

                    for img in msg["content"]["images"]:
                        image = {"type": "image"}
                        image["image"] = img
                        image["min_pixels"] = min_pixels
                        image["max_pixels"] = max_pixels

                        content.append(image)
                    content.append({"type": "text", "text": text})

                    prompt_messages.append({"role": "user", "content": content})

                else:
                    raise ValueError("User message must contain an image.")

            elif msg["role"] == "assistant" or msg["role"] == "system":
                prompt_messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]["text"]}],
                    }
                )
            else:
                raise ValueError(f"Unknown role {msg['role']}")

        return prompt_messages

    def _get_db_item(self, safe_index):
        datadb = self._get_db()
        with datadb.begin() as txn:
            db_sample = txn.get(f"{safe_index:08d}".encode())

        try:
            sample = pickle.loads(bytes(db_sample))

        except Exception as e:
            raise ValueError(f"Sample {safe_index} not found") from e

        # Decode images
        img1 = Image.fromarray(
            cv2.cvtColor(cv2.imdecode(np.frombuffer(sample["image1"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        )
        img2 = Image.fromarray(
            cv2.cvtColor(cv2.imdecode(np.frombuffer(sample["image2"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        )

        sample_data = {
            "db_idx": safe_index,
            "image1": img1,
            "image2": img2,
            "matches": sample["matches"],
        }
        if "overlap" in sample:
            try:
                sample_data["overlap"] = float(sample["overlap"])
            except (TypeError, ValueError):
                sample_data["overlap"] = sample["overlap"]
        if "meta" in sample:
            sample_data["meta"] = copy.deepcopy(sample["meta"])

        return sample_data

    def rescale_coordinates(self, sample: dict, to_relative: bool = True) -> tuple[int, int]:
        """Process the answer by adjusting the bounding box coordinates.

        Args:
            sample (dict): The input sample containing bounding box information.
            to_relative (bool): Whether to rescale the coords to relative form.
        Return:
            tuple[int, int]: The actual height and width of the image input to the model.
        """
        img1, img2 = sample["image1"], sample["image2"]
        assert img1.size == img2.size, f"Image sizes are not equal: {img1.size} != {img2.size}"
        w, h = img1.size
        h_r, w_r = smart_resize(
            height=h,
            width=w,
            factor=self.visual_patch_size * self.visual_merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        sample["image1"] = img1.resize((w_r, h_r), Image.LANCZOS)
        sample["image2"] = img2.resize((w_r, h_r), Image.LANCZOS)

        h_scale = h_r / h
        w_scale = w_r / w

        matches = sample.get("matches", [])
        assert len(matches) > 0, "No matches found in sample."

        for pair in matches:
            x1, x2, y1, y2 = pair["x1"], pair["x2"], pair["y1"], pair["y2"]
            ori_is_relative = pair.get("is_relative", False)
            if not ori_is_relative:
                if to_relative:
                    # @zhonghao, handle resizing problem, using relative coordinates
                    pair["x1"] = int(float(x1 / w) * 1000)
                    pair["x2"] = int(float(x2 / w) * 1000)
                    pair["y1"] = int(float(y1 / h) * 1000)
                    pair["y2"] = int(float(y2 / h) * 1000)
                    pair["is_relative"] = True
                else:
                    pair["x1"] = int(x1 * w_scale)
                    pair["x2"] = int(x2 * w_scale)
                    pair["y1"] = int(y1 * h_scale)
                    pair["y2"] = int(y2 * h_scale)
                    pair["is_relative"] = False

            if ori_is_relative and not to_relative:
                pair["x1"] = int(float(x1) * w_r / 1000)
                pair["x2"] = int(float(x2) * w_r / 1000)
                pair["y1"] = int(float(y1) * h_r / 1000)
                pair["y2"] = int(float(y2) * h_r / 1000)
                pair["is_relative"] = False

        return h_r, w_r


def annotate_image(
    src_img: cv2.Mat | Image.Image,
    points: list[dict],
    config: dict = None,
) -> Image.Image:
    """Create an annotated image from the matches.

    Args:
        img (cv2.Mat): The image to annotate.
        matches (list[dict]): The matches to annotate, formatted as { 'label': str, 'point_2d': (x, y) }.

    Returns:
        img (cv2.Mat): The annotated image.
    """
    if isinstance(src_img, Image.Image):
        cvt_img = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
        img = cvt_img.copy()
        h, w, _ = img.shape
    else:
        h, w, _ = src_img.shape
        img = src_img.copy()

    if config is None:
        config = {}

    color = config.get("color", (0, 255, 0))
    circle_thickness = config.get("circle_thickness", 2)
    font_thickness = config.get("font_thickness", 1)

    text_color = config.get("text_color", (255, 255, 255))  # White text
    font = config.get("font", cv2.FONT_HERSHEY_SIMPLEX)
    font_scale = config.get("font_scale", 0.5)

    padding = int(config.get("label_padding", 2))
    radius = int(config.get("radius", 8))
    line_type = config.get("line_type", cv2.LINE_AA)

    for point in points:
        (x, y) = point["point_2d"]
        is_relative = point.get("is_relative", False)

        x = int(x * w / 1000) if is_relative else int(x)
        y = int(y * h / 1000) if is_relative else int(y)

        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        label = str(point["label"])

        cv2.circle(img, (x, y), radius, color, circle_thickness, lineType=line_type)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w = max(1, text_w)
        text_h = max(1, text_h)

        text_x = int(np.clip(x - text_w // 2, padding, max(padding, w - text_w - padding)))

        preferred_baseline = y - radius - padding
        if preferred_baseline - text_h - padding >= 0:
            text_y = preferred_baseline
        else:
            text_y = y + radius + padding + text_h

        text_y = int(np.clip(text_y, text_h + padding, h - padding))

        rect_top_f = text_y - text_h - padding
        rect_bottom_f = text_y + baseline + padding
        rect_left_f = text_x - padding
        rect_right_f = text_x + text_w + padding

        rect_top = max(0, int(np.floor(rect_top_f)))
        rect_bottom = min(h - 1, int(np.ceil(rect_bottom_f)))
        if rect_bottom <= rect_top:
            rect_bottom = min(h - 1, rect_top + text_h + baseline + 2 * padding)

        rect_left = max(0, int(np.floor(rect_left_f)))
        rect_right = min(w - 1, int(np.ceil(rect_right_f)))
        if rect_right <= rect_left:
            rect_right = min(w - 1, rect_left + text_w + 2 * padding)

        cv2.rectangle(img, (rect_left, rect_top), (rect_right, rect_bottom), (0, 0, 0), -1)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, line_type)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def filter_sparse_points(matches, min_distance=20, max_points=20, eps_scale=0.8, use_greedy=True):
    """
    Filter matches to reduce crowding by clustering nearby points and selecting sparse representatives.

    Args:
        matches: List of match dictionaries with x1, y1, x2, y2 coordinates
        min_distance: Minimum distance between selected points (in pixels)
        max_points: Maximum number of points to select

    Returns:
        List of filtered matches
    """
    if len(matches) <= max_points:
        return matches

    # Extract coordinates for both images
    points_img1 = np.array([[match["x1"], match["y1"]] for match in matches])
    points_img2 = np.array([[match["x2"], match["y2"]] for match in matches])

    # Combine coordinates for clustering (weight both images equally)
    combined_points = np.hstack([points_img1, points_img2])

    # Use DBSCAN clustering to group nearby points
    eps = min_distance * eps_scale  # Slightly smaller eps for clustering
    clustering = DBSCAN(eps=eps, min_samples=1).fit(combined_points)
    labels = clustering.labels_

    # Select one representative point from each cluster
    unique_labels = np.unique(labels)
    selected_matches = []

    for label in unique_labels:
        # Get all points in this cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_matches = [matches[i] for i in cluster_indices]

        if len(cluster_matches) == 1:
            selected_matches.append(cluster_matches[0])
        else:
            # Select the point closest to cluster centroid
            cluster_points = combined_points[cluster_indices]
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            best_idx = cluster_indices[np.argmin(distances)]
            selected_matches.append(matches[best_idx])

    # If we still have too many points, use greedy selection for maximum spacing
    if len(selected_matches) > max_points:
        if use_greedy:
            selected_matches = greedy_max_spacing_selection(selected_matches, max_points, min_distance)
        else:
            selected_matches = random.sample(selected_matches, max_points)

    return selected_matches


def greedy_max_spacing_selection(matches, max_points, min_distance):
    """
    Greedily select points with maximum spacing using combined distance from both images.

    Args:
        matches: List of match dictionaries
        max_points: Maximum number of points to select
        min_distance: Minimum distance between points

    Returns:
        List of selected matches
    """
    if len(matches) <= max_points:
        return matches

    # Extract coordinates
    points_img1 = np.array([[match["x1"], match["y1"]] for match in matches])
    points_img2 = np.array([[match["x2"], match["y2"]] for match in matches])

    selected_indices = []
    remaining_indices = list(range(len(matches)))

    # Start with a random point
    first_idx = random.choice(remaining_indices)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    while len(selected_indices) < max_points and remaining_indices:
        best_idx = None
        best_min_distance = 0

        # Find the point that maximizes minimum distance to already selected points
        for candidate_idx in remaining_indices:
            candidate_img1 = points_img1[candidate_idx]
            candidate_img2 = points_img2[candidate_idx]

            min_dist = float("inf")
            for selected_idx in selected_indices:
                selected_img1 = points_img1[selected_idx]
                selected_img2 = points_img2[selected_idx]

                # Calculate combined distance (average of distances in both images)
                dist1 = np.linalg.norm(candidate_img1 - selected_img1)
                dist2 = np.linalg.norm(candidate_img2 - selected_img2)
                combined_dist = (dist1 + dist2) / 2

                min_dist = min(min_dist, combined_dist)

            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_idx = candidate_idx

        if best_idx is not None and best_min_distance >= min_distance:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            # If no point meets minimum distance requirement, break
            break

    return [matches[i] for i in selected_indices]


def draw_img(sample, filter_crowded_points=True, min_point_distance=20, max_points_per_image=6):
    src_matches = copy.deepcopy(sample["matches"])
    width = int(sample.get("width") or sample["image1"].size[0])
    height = int(sample.get("height") or sample["image2"].size[1])

    def _to_abs(match: dict) -> dict:
        if match.get("is_relative", False):
            converted = dict(match)
            converted["x1"] = int(round(float(match["x1"]) * width / 1000.0))
            converted["y1"] = int(round(float(match["y1"]) * height / 1000.0))
            converted["x2"] = int(round(float(match["x2"]) * width / 1000.0))
            converted["y2"] = int(round(float(match["y2"]) * height / 1000.0))
            converted["is_relative"] = False
            return converted
        converted = dict(match)
        converted["x1"] = int(round(float(match["x1"])))
        converted["y1"] = int(round(float(match["y1"])))
        converted["x2"] = int(round(float(match["x2"])))
        converted["y2"] = int(round(float(match["y2"])))
        converted["is_relative"] = False
        return converted

    abs_matches = [_to_abs(match) for match in src_matches]

    def _matches_are_spaced(selected: list[dict], candidate: dict) -> bool:
        if not selected:
            return True
        for existing in selected:
            if (
                math.hypot(
                    float(candidate["x1"]) - float(existing["x1"]), float(candidate["y1"]) - float(existing["y1"])
                )
                < min_point_distance
            ):
                return False
            if (
                math.hypot(
                    float(candidate["x2"]) - float(existing["x2"]), float(candidate["y2"]) - float(existing["y2"])
                )
                < min_point_distance
            ):
                return False
        return True

    def _enforce_spacing(candidates: list[dict]) -> list[dict]:
        selected: list[dict] = []
        signatures: set[tuple] = set()
        for match in candidates:
            if len(selected) >= max_points_per_image:
                break
            sig = (match["x1"], match["y1"], match["x2"], match["y2"])
            if sig in signatures:
                continue
            if _matches_are_spaced(selected, match):
                selected.append(match)
                signatures.add(sig)
        return selected

    # Apply sparse filtering if enabled
    if filter_crowded_points:
        matches = filter_sparse_points(abs_matches, min_point_distance, max_points_per_image)
        matches = _enforce_spacing(matches)
        print(f"  Filtered to {len(matches)} points from {len(abs_matches)} original matches")
    else:
        matches = _enforce_spacing(abs_matches)

    if len(matches) < max_points_per_image:
        existing = {(match["x1"], match["y1"], match["x2"], match["y2"]) for match in matches}
        residual = [m for m in abs_matches if (m["x1"], m["y1"], m["x2"], m["y2"]) not in existing]
        random.shuffle(residual)
        for match in residual:
            if len(matches) >= max_points_per_image:
                break
            if _matches_are_spaced(matches, match):
                matches.append(match)

    img1_annotated = cv2.cvtColor(np.array(sample["image1"]), cv2.COLOR_RGB2BGR)
    img2_annotated = cv2.cvtColor(np.array(sample["image2"]), cv2.COLOR_RGB2BGR)

    # Create shuffled IDs for each image
    num_matches = len(matches)
    ids_img = list(range(1, num_matches + 1))  # [1, 2, 3, ..., n]
    random.shuffle(ids_img)  # Shuffle IDs for second image

    # Create mapping for later reference
    id_mapping = {}
    for i in range(num_matches):
        id_mapping[str(i + 1)] = str(ids_img[i])

    # Random colors for better visibility
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    color = random.choice(colors)
    # Random circle thickness (1-4 pixels)
    circle_thickness = random.randint(1, 2)
    thickness = random.randint(1, 2)  # Random font thickness (1-3 pixels)
    # Draw ID numbers with random font thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    config = {
        "color": color,
        "circle_thickness": circle_thickness,
        "font_thickness": thickness,
        "font": font,
        "font_scale": font_scale,
        "text_color": (255, 255, 255),
    }

    points_img1 = [{"point_2d": (m["x1"], m["y1"]), "label": str(i + 1)} for i, m in enumerate(matches)]
    points_img2 = [{"point_2d": (m["x2"], m["y2"]), "label": id_mapping[str(i + 1)]} for i, m in enumerate(matches)]

    return {
        "annotated_image1": annotate_image(img1_annotated, points_img1, config=config),
        "annotated_image2": annotate_image(img2_annotated, points_img2, config=config),
        "id_mapping": id_mapping,
    }
