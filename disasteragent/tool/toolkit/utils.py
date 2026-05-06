"""感知工具链通用辅助函数。"""
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pycocotools import mask as mask_utils


def extract_filename(path: str) -> str:
    """从完整路径中提取文件名（含扩展名）"""
    return Path(path).name


def load_json_cached(json_path: Path, cache: dict, cache_key: str) -> Any:
    """带缓存的 JSON 文件加载"""
    if cache_key not in cache:
        with open(json_path, "r", encoding="utf-8") as f:
            cache[cache_key] = json.load(f)
    return cache[cache_key]


def decode_rle_mask(rle_str: str, image_size: List[int]) -> Dict[str, Any]:
    """
    将简化的 RLE 字符串转换为 pycocotools 格式的 RLE 字典

    Args:
        rle_str: RLE 编码字符串
        image_size: [H, W]

    Returns:
        RLE 字典 {"size": [H, W], "counts": "..."}
    """
    return {
        "size": image_size,
        "counts": rle_str
    }


def rle_to_bbox(rle: Dict[str, Any]) -> tuple:
    """从 RLE 掩膜计算 bbox (x, y, w, h)"""
    binary_mask = mask_utils.decode(rle)
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    return (x, y, w, h)


# 语义标签别名映射：将各种变体统一到标准形式
_SEMANTIC_ALIASES = {
    # 复数 → 单数
    "buildings": "building",
    "roads": "road",
    # 组合表达 → 标准形式
    "river or water": "water body",
    "water body": "water body",
    "water_body": "water body",
    "farmland": "farmland",
    "agricultural land": "farmland",
    "agriculture": "farmland",
}


def normalize_semantic_label(label: str) -> str:
    """
    标准化语义标签，统一别名到标准形式
    
    Args:
        label: 原始语义标签（可能是复数、带空格等变体）
        
    Returns:
        标准化后的语义标签
    """
    normalized = label.lower().strip()
    return _SEMANTIC_ALIASES.get(normalized, normalized)
