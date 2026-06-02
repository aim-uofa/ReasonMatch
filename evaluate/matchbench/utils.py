from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


JSON_BLOCK_RE = re.compile(r"```json(.*?)```", re.IGNORECASE | re.DOTALL)
BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


ANNO_MATCH_MULTI_NO_THINK_TEMPLATE = r"""You are given two images of the same physical scene, each having several regions annotated with circles and IDs (e.g., "1", "2", "3").

Your task is to identify the underlying correspondence between regions in the two images. Note that **maybe not** every region in the first image has a match in the second.

Please directly provide your answer **wrapped in <answer></answer> tags** as JSON format. The JSON object is a mapping of region IDs from image A (as string keys) to the corresponding region IDs in image B (as string values). **For regions in A that have no match in B, use "none" as the value.**

**Example of the required output format which you should follow:**
<answer>

```json
{ "1": "2",  "2": "none", "3": "1", "4": "3" }
```
</answer>
"""


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


*  **`final answer`**: Your final answer based on your thinking part **wrapped by <answer> </answer> tags**. The JSON object is a mapping of region IDs from image A (as string keys) to the corresponding region IDs in image B (as string values). **For regions in A that have no match in B, use "none" as the value.**


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


def load_annotation_index(index_path: Path) -> list[dict]:
    with index_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Annotation index must be a list, got {type(data)}")
    return data


def read_metadata(testset_root: Path, relative_path: str) -> dict:
    meta_path = testset_root / relative_path
    with meta_path.open("r", encoding="utf-8") as fh:
        meta_data = json.load(fh)

    meta_data["image1"] = str(testset_root / meta_data["image1"].split("testset/")[-1])
    meta_data["image2"] = str(testset_root / meta_data["image2"].split("testset/")[-1])
    return meta_data


def build_prompt(metadata: dict, think: bool = True) -> str:
    return ANNO_MATCH_MULTI_TEMPLATE if think else ANNO_MATCH_MULTI_NO_THINK_TEMPLATE


def extract_json_from_text(text: str) -> tuple[str | None, str]:
    if not text:
        return None, ""
    fenced = JSON_BLOCK_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        return candidate, text
    braces = BRACE_RE.search(text)
    if braces:
        return braces.group(0), text
    return None, text


def safe_parse_json(candidate: str | None) -> dict | list | None:
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def compare_mappings(
    gt: dict[str, Any],
    pred: dict[str, Any] | None
) -> tuple[float, float, float, int]:
    if not isinstance(gt, dict):
        return 0.0, 0.0, 0.0, 0

    if not isinstance(pred, dict) or not pred:
        return 0.0, 0.0, 0.0, len(gt)

    def is_none_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.lower() == "none":
            return True
        return False

    correct_count = 0
    valid_pred_count = 0

    for key, pred_value in pred.items():
        if is_none_value(pred_value):
            continue

        valid_pred_count += 1
        if key in gt and str(pred_value) == str(gt[key]):
            correct_count += 1

    precision = correct_count / max(valid_pred_count, 1)
    recall = correct_count / max(len(gt), 1)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1, len(gt)
