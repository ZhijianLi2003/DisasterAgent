#安装 pip install 

#导入包
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math
from pycocotools import mask as mask_util

# ============================================================
# 感知层的定义和工具
# ============================================================

@dataclass
class BoundingBox:
    """边界框"""
    x: float       # 左上角 x
    y: float       # 左上角 y
    width: float   # 宽度
    height: float  # 高度
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """转换为 (x1, y1, x2, y2) 格式"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def center(self) -> Tuple[float, float]:
        """获取中心点"""
        return (self.x + self.width / 2, self.y + self.height / 2)



class SpatialRelation(Enum):
    """空间位置关系 - 8方向"""
    RIGHT = "to the right of"                   # 在右边
    UPPER_RIGHT = "to the upper-right of"       # 在右上方
    ABOVE = "above"                             # 在上方
    UPPER_LEFT = "to the upper-left of"         # 在左上方
    LEFT = "to the left of"                     # 在左边
    LOWER_LEFT = "to the lower-left of"         # 在左下方
    BELOW = "below"                             # 在下方
    LOWER_RIGHT = "to the lower-right of"       # 在右下方


def compute_distance(center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
    """计算两个中心点之间的欧氏距离"""
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def compute_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    计算两个 BoundingBox 的 IoU (Intersection over Union)
    
    Args:
        bbox1: 第一个边界框
        bbox2: 第二个边界框
        
    Returns:
        IoU 值 (0.0 ~ 1.0)
    """
    ax1, ay1 = bbox1.x, bbox1.y
    ax2, ay2 = bbox1.x + bbox1.width, bbox1.y + bbox1.height
    bx1, by1 = bbox2.x, bbox2.y
    bx2, by2 = bbox2.x + bbox2.width, bbox2.y + bbox2.height
    
    # 交集
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    if intersection <= 0:
        return 0.0
    
    # 并集
    area_a = bbox1.width * bbox1.height
    area_b = bbox2.width * bbox2.height
    union = area_a + area_b - intersection
    
    return intersection / union if union > 0 else 0.0


# ============================================================
# COCO 压缩 RLE 编解码与 Mask 计算 (基于 pycocotools)
# ============================================================
#
# COCO RLE 格式: {"size": [H, W], "counts": "压缩字符串"}
#
# 编码原理:
#   1. 二值掩码按列优先展平，统计交替的 0/1 游程长度 → counts 列表
#   2. counts[i] (i>2) 做 delta 编码: x = counts[i] - counts[i-2]
#   3. 每个 delta 值用变长编码写入字符串 (5-bit 数据 + 1-bit 续位, ASCII 偏移 48)
#

def _to_pycocotools_rle(rle: Dict[str, Any]) -> Dict[str, Any]:
    """将 counts 为 str 的 RLE 转换为 pycocotools 需要的 bytes 格式"""
    rle_copy = rle.copy()
    if isinstance(rle_copy["counts"], str):
        rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
    return rle_copy


def rle_encode(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    将二值掩码编码为 COCO 压缩 RLE 格式
    
    Args:
        binary_mask: 二值掩码 (H, W), 值为 0 或 1
        
    Returns:
        RLE 字典: {"size": [H, W], "counts": "压缩字符串"}
    """
    # pycocotools 要求 Fortran 连续的 uint8 数组
    mask_f = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_f)
    rle["counts"] = rle["counts"].decode("utf-8")  # bytes → str
    return rle


def rle_decode(rle: Dict[str, Any]) -> np.ndarray:
    """
    将 COCO 压缩 RLE 解码为二值掩码
    
    Args:
        rle: RLE 字典 {"size": [H, W], "counts": "压缩字符串"}
        
    Returns:
        二值掩码 np.ndarray (H, W), dtype=uint8
    """
    return mask_util.decode(_to_pycocotools_rle(rle))


def rle_area(rle: Dict[str, Any]) -> int:
    """
    从 RLE 计算掩码面积（无需解码）
    
    Args:
        rle: RLE 字典
        
    Returns:
        前景像素数量（面积）
    """
    return int(mask_util.area(_to_pycocotools_rle(rle)))


def compute_mask_iou(rle1: Dict[str, Any], rle2: Dict[str, Any]) -> float:
    """
    计算两个 RLE 掩码的 IoU
    
    Args:
        rle1: 第一个 RLE 掩码
        rle2: 第二个 RLE 掩码
        
    Returns:
        IoU 值 (0.0 ~ 1.0)
    """
    r1 = _to_pycocotools_rle(rle1)
    r2 = _to_pycocotools_rle(rle2)
    iou_matrix = mask_util.iou([r1], [r2], [0])
    return float(iou_matrix[0][0])


def compute_region_iou(
    bbox1: 'BoundingBox', bbox2: 'BoundingBox',
    mask1: Optional[Dict[str, Any]] = None,
    mask2: Optional[Dict[str, Any]] = None,
) -> float:
    """
    计算两个区域的 IoU：优先使用 mask (RLE)，无 mask 时退化为 bbox IoU
    
    Args:
        bbox1: 第一个边界框
        bbox2: 第二个边界框
        mask1: 第一个 RLE 掩码 (可选)
        mask2: 第二个 RLE 掩码 (可选)
        
    Returns:
        IoU 值 (0.0 ~ 1.0)
    """
    if mask1 is not None and mask2 is not None:
        return compute_mask_iou(mask1, mask2)
    return compute_iou(bbox1, bbox2)


def compute_region_area(
    bbox: 'BoundingBox',
    mask: Optional[Dict[str, Any]] = None,
) -> float:
    """
    计算区域面积：优先使用 mask (RLE)，无 mask 时退化为 bbox 面积
    
    Args:
        bbox: 边界框
        mask: RLE 掩码 (可选)
        
    Returns:
        面积（像素数）
    """
    if mask is not None:
        return float(rle_area(mask))
    return bbox.width * bbox.height


# 空间关系计算函数
def compute_spatial_relation(
    box1_center: Tuple[float, float],
    box2_center: Tuple[float, float]
) -> SpatialRelation:
    """
    计算 box2 相对于 box1 的空间方向关系（8方向）
    
    Args:
        box1_center: box1 的中心点 (x, y)
        box2_center: box2 的中心点 (x, y)
        
    Returns:
        SpatialRelation: 空间关系枚举
    """
    dx = box2_center[0] - box1_center[0]  # 正值: box2 在 box1 右边
    dy = box2_center[1] - box1_center[1]  # 正值: box2 在 box1 下方（图像坐标系y向下）
    
    # 计算角度（以 box1 为原点，box2 的方向）
    # 注意：图像坐标系 y 向下，所以 dy 取负来转换为标准数学坐标系
    angle = np.arctan2(-dy, dx) * 180 / np.pi
    
    # 8方向判断
    if -22.5 <= angle < 22.5:
        return SpatialRelation.RIGHT
    elif 22.5 <= angle < 67.5:
        return SpatialRelation.UPPER_RIGHT
    elif 67.5 <= angle < 112.5:
        return SpatialRelation.ABOVE
    elif 112.5 <= angle < 157.5:
        return SpatialRelation.UPPER_LEFT
    elif angle >= 157.5 or angle < -157.5:
        return SpatialRelation.LEFT
    elif -157.5 <= angle < -112.5:
        return SpatialRelation.LOWER_LEFT
    elif -112.5 <= angle < -67.5:
        return SpatialRelation.BELOW
    else:  # -67.5 <= angle < -22.5
        return SpatialRelation.LOWER_RIGHT


# 生成空间关系的文本描述
def get_spatial_relation_text(
    box1_name: str,
    box2_name: str,
    relation: SpatialRelation
) -> str:
    """
    生成空间关系的文本描述
    
    Args:
        box1_name: box1 的名称
        box2_name: box2 的名称
        relation: 空间关系
        
    Returns:
        str: 如 "building is to the upper-right of river"
    """
    return f"{box2_name} is {relation.value} {box1_name}"
