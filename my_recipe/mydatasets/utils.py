import logging
import random

import cv2
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from sklearn.cluster import DBSCAN

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

# TODO: @zhognhao: only for image inference, maybe need modification on video training
MAX_VISUAL_TOKENS = 1500
MIN_VISUAL_TOKENS = 100


def build_model_inputs(self, messages):
    """Builds model inputs from messages"""
    row_dict = {}

    if self.processor is not None:
        # from verl.utils.dataset.vision_utils import process_image, process_video

        raw_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        multi_modal_data = {}

        # @zhonghao: currently, multimodal message format only supports qwen2_vl style
        images, videos = process_vision_info(messages, image_patch_size=self.visual_patch_size)

        if images:
            multi_modal_data["image"] = images
        if videos:
            multi_modal_data["video"] = [video.numpy() for video in videos]

        # images = None
        # row_dict_images = row_dict.pop(self.image_key, None)
        # if row_dict_images:
        #     images = [process_image(image) for image in row_dict_images]

        #     # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
        #     # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
        #     multi_modal_data["image"] = images

        # videos = None
        # row_dict_videos = row_dict.pop(self.video_key, None)
        # if row_dict_videos:
        #     videos = [process_video(video) for video in row_dict_videos]

        #     # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
        #     # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
        #     multi_modal_data["video"] = [video.numpy() for video in videos]

        # NOTE: set do_resize=False to avoid implicit resize behavior when processing images.
        model_inputs = self.processor(
            text=[raw_prompt], images=images, videos=videos, return_tensors="pt", do_resize=False
        )

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        row_dict["multi_modal_data"] = multi_modal_data

        # We will do batch.union() in the trainer,
        # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
        if self.return_multi_modal_inputs:
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

    else:
        if self.apply_chat_template_kwargs.get("chat_template") is None:
            assert hasattr(self.tokenizer, "chat_template"), (
                "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                "models like GLM can copy chat_template.jinja from instruct models"
            )
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=self.max_prompt_length,
        pad_token_id=self.tokenizer.pad_token_id,
        left_pad=True,
        truncation=self.truncation,
    )

    if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
        # qwen-vl mrope
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids[0],
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
            attention_mask=attention_mask[0],
        )  # (3, seq_length)
        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
    elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
        from verl.models.transformers.glm4v import get_rope_index

        vision_position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids[0],
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            attention_mask=attention_mask[0],
        )  # (3, seq_length)
        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
    else:
        position_ids = compute_position_id_with_mask(attention_mask)

    row_dict["input_ids"] = input_ids[0]
    row_dict["attention_mask"] = attention_mask[0]
    row_dict["position_ids"] = position_ids[0]

    raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > self.max_prompt_length:
        if self.truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
        elif self.truncation == "right":
            raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
        elif self.truncation == "middle":
            left_half = self.max_prompt_length // 2
            right_half = self.max_prompt_length - left_half
            raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
        elif self.truncation == "error":
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

    row_dict["raw_prompt_ids"] = raw_prompt_ids
    # encode prompts without chat template
    if self.return_raw_chat:
        row_dict["raw_prompt"] = messages

    # get prompts with chat template
    if self.return_full_prompt:
        row_dict["full_prompts"] = raw_prompt  # array of strings

    # add index for each prompt
    if "extra_info" not in row_dict or row_dict["extra_info"] is None:
        row_dict["extra_info"] = dict()
    index = row_dict.get("extra_info", {}).get("index", 0)
    tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
    interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
    need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
    if need_tools_kwargs and not tools_kwargs:
        logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
    row_dict["index"] = index
    row_dict["tools_kwargs"] = tools_kwargs
    row_dict["interaction_kwargs"] = interaction_kwargs
    return row_dict


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


def draw_correspondences(
    sample: dict,
    filter_crowded_points=True,
    min_point_distance=20,
    max_points_per_image=6,
):
    # sample_data = {
    #     "video": sample["video"],
    #     "image1": img1,
    #     "image2": img2,
    #     "anno_image1": anno_img1,
    #     "anno_image2": anno_img2,
    #     "id_mapping": sample["id_mapping"],
    #     "reverse_mapping": sample["reverse_mapping"],
    #     "matches": sample["matches"],
    #     "sequential_names": sample["sequential_names"],
    # }

    img1 = sample["image1"]
    img2 = sample["image2"]
    matches = sample["matches"]

    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    # Apply sparse filtering if enabled
    if filter_crowded_points:
        matches = filter_sparse_points(matches, min_point_distance, max_points_per_image)
        print(f"  Filtered to {len(matches)} points from {len(matches)} original matches")

    # img1 = cv2.resize(img1, (960, 540), interpolation=cv2.INTER_CUBIC)
    # img2 = cv2.resize(img2, (960, 540), interpolation=cv2.INTER_CUBIC)

    # Create copies for annotation
    img1_annotated = img1.copy()
    img2_annotated = img2.copy()

    # Create shuffled IDs for each image
    num_matches = len(matches)
    ids_img = list(range(1, num_matches + 1))  # [1, 2, 3, ..., n]
    random.shuffle(ids_img)  # Shuffle IDs for second image

    # Create mapping for later reference
    id_mapping = {}
    for i in range(num_matches):
        id_mapping[str(i + 1)] = str(ids_img[i])

    plot_id(img1_annotated, img2_annotated, id_mapping, matches)

    return {
        "img1_annotated": img1_annotated,
        "img2_annotated": img2_annotated,
        "id_mapping": id_mapping,
    }


def plot_id(img1_annotated, img2_annotated, id_mapping, matches):
    # Random colors for better visibility
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    color = random.choice(colors)
    # Random circle thickness (1-4 pixels)
    circle_thickness = random.randint(1, 2)
    thickness = random.randint(1, 2)  # Random font thickness (1-3 pixels)

    # Draw matching points with shuffled IDs
    for match_idx, match in enumerate(matches):
        id_img1 = match_idx + 1
        id_img2 = int(id_mapping[str(id_img1)])

        x1, y1 = int(match["x1"]), int(match["y1"])
        x2, y2 = int(match["x2"]), int(match["y2"])

        text_color = (255, 255, 255)  # White text

        # Draw circles at matching points with random thickness
        cv2.circle(img1_annotated, (x1, y1), 8, color, circle_thickness)
        cv2.circle(img2_annotated, (x2, y2), 8, color, circle_thickness)

        # Draw ID numbers with random font thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        # Get text size for better positioning (use longer ID for sizing)
        max_id_str = str(max(id_img1, id_img2))
        (text_w, text_h), _ = cv2.getTextSize(max_id_str, font, font_scale, thickness)

        # Draw black background rectangle for text
        cv2.rectangle(
            img1_annotated,
            (x1 - text_w // 2 - 2, y1 - text_h - 10),
            (x1 + text_w // 2 + 2, y1 - 5),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            img2_annotated,
            (x2 - text_w // 2 - 2, y2 - text_h - 10),
            (x2 + text_w // 2 + 2, y2 - 5),
            (0, 0, 0),
            -1,
        )

        # Draw different IDs on each image
        if y1 - 8 < text_h:
            cv2.putText(
                img1_annotated,
                str(id_img1),
                (x1 - text_w // 2, y1 + 8 + text_h),
                font,
                font_scale,
                text_color,
                thickness,
            )
        else:
            cv2.putText(
                img1_annotated,
                str(id_img1),
                (x1 - text_w // 2, y1 - 8),
                font,
                font_scale,
                text_color,
                thickness,
            )

        if y2 - 8 < text_h:
            cv2.putText(
                img2_annotated,
                str(id_img2),
                (x2 - text_w // 2, y2 + 8 + text_h),
                font,
                font_scale,
                text_color,
                thickness,
            )
        else:
            cv2.putText(
                img2_annotated,
                str(id_img2),
                (x2 - text_w // 2, y2 - 8),
                font,
                font_scale,
                text_color,
                thickness,
            )
