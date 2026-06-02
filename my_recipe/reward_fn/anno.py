import json
import re

import jieba
import numpy as np


def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    return zip(*[words[i:] for i in range(ngram_size)])


def zipngram_chinese_word(text: str, ngram_size: int):
    words = jieba.lcut(text.strip())
    return ["".join(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]


def detect_language(text, threshold=0.3):
    """
    判断文本主要语言（中文/英文）

    参数：
    text: 输入文本
    threshold: 中文判定阈值（中文字符占比>threshold时判定为中文）

    返回：
    "chinese" 或 "english"
    """
    chinese_count = 0
    english_count = 0

    for char in text:
        # 检测中文字符（Unicode范围：4E00-9FFF 包含常用汉字）
        if "\u4e00" <= char <= "\u9fff":
            chinese_count += 1
        # 检测英文字符（大小写字母）
        elif "a" <= char.lower() <= "z":
            english_count += 1

    total_valid = chinese_count + english_count

    # 处理无有效字符的情况
    if total_valid == 0:
        return "english"  # 默认返回英文

    # 计算中文占比
    chinese_ratio = chinese_count / total_valid

    # 根据阈值判断
    return "chinese" if chinese_ratio > threshold else "english"


def calculate_repetition_penalty(generation, ngram_size: int = 9, max_penalty: float = -3, token_level=False) -> float:
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive!")

    if max_penalty == 0:
        return 0

    ngrams = set()
    total = 0
    if not token_level:
        if detect_language(generation) == "chinese":
            for ng in zipngram_chinese_word(generation, ngram_size):
                ngrams.add(ng)
                total += 1
        else:
            for ng in zipngram(generation, ngram_size):
                ngrams.add(ng)
                total += 1
        if total == 0:
            return 0
    else:
        for i in range(len(generation)):
            ng = tuple(generation[i : i + ngram_size])
            ngrams.add(ng)
            total += 1
    scaling = 1 - len(ngrams) / total
    return scaling * max_penalty


def offset_sigmoid_reward(l2_distance, scale=0.04, offset=80.0):
    """
    Offset sigmoid: reward = 1 / (1 + exp(scale * (l2_distance - offset)))

    Args:
        l2_distance: L2 distance between predicted and ground truth
        scale: Controls steepness of the curve (default: 0.12)
        offset: Distance at which reward = 0.5 (default: 25.0)

    Returns:
        Reward in [0, 1]
    """
    return 1.0 / (1.0 + np.exp(scale * (l2_distance - offset)))


def json_extract(prediction_text: str) -> dict:
    try:
        # 方法1: 尝试提取<answer>标签中的JSON
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", prediction_text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1)
            # 移除可能的markdown代码块标记
            answer_content = re.sub(r"```json\s*|\s*```", "", answer_content)
            # 提取JSON部分
            start_idx = answer_content.find("{")
            end_idx = answer_content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = answer_content[start_idx:end_idx]
            else:
                json_str = answer_content
        else:
            # 方法2: 直接提取整个文本中的第一个JSON对象
            start_idx = prediction_text.find("{")
            end_idx = prediction_text.rfind("}") + 1
            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError("No JSON structure found in prediction text")
            json_str = prediction_text[start_idx:end_idx]

        ##################
        # 修复 JSON 字符串 #
        ##################

        # 清理控制字符和格式问题
        # 移除控制字符（保留换行和制表符用于后续处理）
        json_str = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", json_str)
        # 移除多余的空白字符和换行
        json_str = re.sub(r"\s+", " ", json_str)
        json_str = json_str.strip()

        # 修复常见的JSON格式问题
        # 1. 修复缺失逗号的问题（字段之间）
        json_str = re.sub(r'"\s+"([a-zA-Z_])', r'", "\1', json_str)
        # 2. 修复尾部多余逗号
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        ##################
        # 修复 JSON 字符串 #
        ##################

        # 解析JSON
        pred_data = json.loads(json_str)

    except Exception:
        pred_data = None

    return pred_data


##########################
#    anno matching
#
##########################


def cal_format_reward(predict_str: str) -> float:
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0

    def content_format(predict_str: str) -> float:
        try:
            json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)
            if not json_match:
                return 0.0
            else:
                # 移除可能的markdown代码块标记
                answer_content = re.sub(r"```json\s*|\s*```", "", json_match.group(1))
                json.loads(answer_content)
                return 1.0

        except Exception:
            return 0.0

    def repeat_cal(predict_str: str) -> float:
        json_match = re.search(r"<thinking>\s*(.*?)\s*</thinking>", predict_str, re.DOTALL)
        if not json_match:
            return 0.0

        p = calculate_repetition_penalty(json_match.group(1))
        return p

    content_format_reward = content_format(predict_str)
    repetition_penalty = repeat_cal(predict_str)

    return {
        "thinking_format_reward": thinking_format_reward,
        "content_format_reward": content_format_reward,
        "repetition_penalty": repetition_penalty,
    }


def matching_acc_reward(predict_str: str, ground_truth: str) -> float:
    valid = 0
    unknown = 0
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)

    try:
        # 移除可能的markdown代码块标记
        answer_content = re.sub(r"```json\s*|\s*```", "", answer_match.group(1))
        pred = json.loads(answer_content)
        gt = json.loads(ground_truth)

        for key in gt.keys():
            if key in pred and str(pred[key]) == str(gt[key]):
                valid += 1

        for key in pred.keys():
            if key not in gt.keys():
                unknown += 1
        return {"accuracy_reward": ((valid - unknown * 0.5) / len(gt))}
    except Exception:
        return {"accuracy_reward": 0.0}


def anno_matching_score(solution_str: str, ground_truth: str) -> dict:
    format_reward = cal_format_reward(solution_str)
    accuracy_reward = matching_acc_reward(solution_str, ground_truth)

    format_reward.update(accuracy_reward)
    reward = format_reward
    score = 0.0

    for key, value in reward.items():
        score += value

    reward["score"] = score

    return reward


##########################
#    anno grounding
##########################


def grounding_acc_reward(predict_str: str, ground_truth: str, is_qwen3: bool, height: int, width: int) -> dict:
    def l2_dist2score(coord1, coord2):
        """calculate l2 distance and map a score accordingly

        Args:
            coord1 (tuple): The (x, y) coordinates of the first point.
            coord2 (tuple): The (x, y) coordinates of the second point.

        Returns:
            float: The mapped score based on the L2 distance.
        """
        try:
            dist = ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5
        except Exception:
            return 0.0
        return offset_sigmoid_reward(dist)

    unknown = 0
    duplicate = 0
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)

    try:
        # 移除可能的markdown代码块标记
        answer_content = re.sub(r"```json\s*|\s*```", "", answer_match.group(1))
        pred = json.loads(answer_content)
    except Exception:
        return {"accuracy_reward": 0.0}

    gt = json.loads(ground_truth)
    rescaled_gt = {}
    for point in gt:
        if is_qwen3 and point.get("is_relative", False):
            rescaled_gt[point["label"]] = point["point_2d"]
        elif not is_qwen3:
            rescaled_gt[point["label"]] = (
                int(point["point_2d"][0] * 1000 / width),
                int(point["point_2d"][1] * 1000 / height),
            )
        else:
            rescaled_gt[point["label"]] = (point["point_2d"][0] * 1000 / width, point["point_2d"][1] * 1000 / height)

    # gt be a list
    # [
    #   { 'point_2d': (x, y), 'label': '1', 'is_relative': True },
    #   { 'point_2d': (x, y), 'label': '2', 'is_relative': True },
    #    ...
    # ]
    # assuming pred be a list of dicts
    # [
    #   { 'point_2d': (x, y), 'label': '1' },
    #   { 'point_2d': (x, y), 'label': '2' },
    #    ...
    # ]

    try:
        pred_dict = {}
        for item in pred:
            id = item.get("label", None)
            coords = item.get("point_2d", None)

            if id is None or coords is None:
                continue
            if id in pred_dict.keys():
                duplicate += 1
            pred_dict[id] = coords

    except Exception:
        return {"accuracy_reward": 0.0}

    pred_keys = list(pred_dict.keys())

    for key in pred_keys:
        if key not in rescaled_gt.keys():
            unknown += 1
            pred_dict.pop(key)

    valid_scores = []
    for i, coord in rescaled_gt.items():
        if i not in pred_dict.keys():
            valid_scores.append(0)
            continue
        if not is_qwen3:
            # need to scale the predicted absolute coord to 1000*1000
            pred_coord = (int(pred_dict[i][0] * 1000 / width), int(pred_dict[i][1] * 1000 / height))
        else:
            pred_coord = pred_dict[i]

        valid_scores.append(l2_dist2score(coord, pred_coord))

    valid_score = sum(valid_scores) / len(rescaled_gt)
    penalty = (unknown * 0.5 + duplicate * 0.5) / len(rescaled_gt)
    # return {"accuracy_reward": ((valid_score - unknown * 0.5) / len(gt))}
    # FIXME: TODO
    return {"accuracy_reward": (valid_score - penalty)}


def anno_grounding_score(solution_str: str, ground_truth: str, is_qwen3: bool, height: int, width: int) -> dict:
    # re-use the format reward as matching task
    format_reward = cal_format_reward(solution_str)
    accuracy_reward = grounding_acc_reward(solution_str, ground_truth, is_qwen3, height, width)

    format_reward.update(accuracy_reward)
    reward = format_reward
    score = 0.0

    for key, value in reward.items():
        score += value

    reward["score"] = score

    return reward


def anno_score(solution_str: str, ground_truth: str, data_source: str, extra_info: object) -> dict:
    data_info = json.loads(data_source)
    problem_type = data_info.get("type", None)
    assert problem_type is not None, 'data_source must contain "type" field indicating problem type.'
    is_qwen3 = data_info.get("is_qwen3", False)
    height = data_info.get("height", None)
    width = data_info.get("width", None)

    assert height is not None and width is not None, 'data_source must contain "height" and "width" fields.'

    if problem_type == "ANNO_MATCH":
        return anno_matching_score(solution_str, ground_truth)
    elif problem_type == "ANNO_GROUND":
        return anno_grounding_score(solution_str, ground_truth, is_qwen3, height, width)
    else:
        raise NotImplementedError(f"Data source {data_source} not supported in anno_score.")
