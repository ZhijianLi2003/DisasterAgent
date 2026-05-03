"""
视觉定位证据 (Visual-Grounding Evidence) — 在线

对应架构图中 Evidence Retrieval 的第二路：
  Graph node → bbox / polygon / drawing → buildings, roads → grounding evidence

输出：
  - grounding_pre.png:  灾前标注图（building + road）
  - grounding_post.png: 灾后标注图（building + road）

绘制规则：
  - building: bbox 矩形框，按损伤等级上色
  - road: mask 多边形轮廓，按损伤等级上色
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from agent.brain import Brain

from agent.state import WorkflowState

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


# ================== 颜色配置 ==================
# Building 损伤等级颜色
BUILDING_DAMAGE_COLORS: Dict[str, tuple] = {
    "Intact": (0, 200, 0),           # 绿色
    "No-damage": (0, 200, 0),        # 绿色
    "Minor-damage": (139, 90, 43),   # 棕色 
    "Damaged": (255, 140, 0),        # 橙色
    "Major-damage": (255, 140, 0),   # 橙色
    "Destroyed": (255, 0, 0),        # 红色
}

# Road 损伤等级颜色
ROAD_DAMAGE_COLORS: Dict[str, tuple] = {
    "Intact": (30, 144, 255),        # 蓝色
    "Flood-submerged": (255, 200, 0),   # 黄色
    "Debris-covered": (128, 128, 128), # 灰色
}

# 默认颜色（未知损伤等级）
DEFAULT_BUILDING_COLOR = (128, 128, 128)  # 灰色
DEFAULT_ROAD_COLOR = (128, 128, 128)      # 灰色


def _get_building_color(damage_label: Optional[str]) -> tuple:
    """根据 building 损伤等级获取颜色"""
    if not damage_label:
        return DEFAULT_BUILDING_COLOR
    for key, color in BUILDING_DAMAGE_COLORS.items():
        if key.lower() in damage_label.lower():
            return color
    return DEFAULT_BUILDING_COLOR


def _get_road_color(damage_label: Optional[str]) -> tuple:
    """根据 road 损伤等级获取颜色"""
    if not damage_label:
        return DEFAULT_ROAD_COLOR
    for key, color in ROAD_DAMAGE_COLORS.items():
        if key.lower() in damage_label.lower():
            return color
    return DEFAULT_ROAD_COLOR


def _resolve_image_path(raw_path: str, base_dir: Path) -> Optional[Path]:
    """解析图像路径，尝试多种可能的位置"""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.is_file():
        return p.resolve()
    for cand in (
        base_dir / raw_path,
        base_dir / Path(raw_path).name,
        Path.cwd() / raw_path,
    ):
        if cand.is_file():
            return cand.resolve()
    return None


def _normalize_semantic(label: str) -> str:
    """标准化语义标签"""
    try:
        from tool.toolkit.utils import normalize_semantic_label
        return normalize_semantic_label(label)
    except ImportError:
        return label.lower().strip()


def _is_building(semantic: str) -> bool:
    """判断是否为建筑语义"""
    return _normalize_semantic(semantic) == "building"


def _is_road(semantic: str) -> bool:
    """判断是否为道路语义"""
    return _normalize_semantic(semantic) == "road"


def _mask_to_polygon(mask_dict: Dict[str, Any]) -> Optional[List[tuple]]:
    """
    从 RLE mask 提取多边形轮廓点
    
    Args:
        mask_dict: RLE 格式的 mask {"size": [H, W], "counts": "..."}
        
    Returns:
        多边形顶点列表 [(x1, y1), (x2, y2), ...] 或 None
    """
    if mask_utils is None:
        return None
    if not mask_dict or not mask_dict.get("counts"):
        return None
    
    try:
        binary_mask = mask_utils.decode(mask_dict)
        
        try:
            import cv2
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            return [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        except ImportError:
            ys, xs = np.where(binary_mask > 0)
            if len(xs) == 0:
                return None
            from scipy.spatial import ConvexHull
            points = np.column_stack((xs, ys))
            if len(points) < 3:
                return None
            hull = ConvexHull(points)
            return [(int(points[i, 0]), int(points[i, 1])) for i in hull.vertices]
    except Exception:
        return None


def _draw_building_and_road(
    img: Image.Image,
    regions: Dict[str, Any],
    line_width: int = 2,
) -> Image.Image:
    """
    在图像上绘制 building 和 road
    
    绘制规则：
      - building: bbox 矩形框，按损伤等级上色
      - road: mask 多边形轮廓，按损伤等级上色
    
    Args:
        img: PIL Image 对象
        regions: STPG regions 字典 (region_id -> RegionNode)
        line_width: 边框线宽
        
    Returns:
        标注后的图像
    """
    draw = ImageDraw.Draw(img, "RGBA")
    
    for rid, region in regions.items():
        semantic = region.semantic
        normalized_sem = _normalize_semantic(semantic)
        damage_label = region.properties.get("damage_label")
        
        # 只处理 building 和 road
        if normalized_sem == "building":
            # Building: bbox 矩形框
            color = _get_building_color(damage_label)
            fill_color = (*color, 50)
            
            bbox = region.bbox
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=color,
                width=line_width,
            )
            draw.rectangle(
                [x, y, x + w, y + h],
                fill=fill_color,
            )
            
        elif normalized_sem == "road":
            # Road: mask 多边形轮廓
            color = _get_road_color(damage_label)
            fill_color = (*color, 50)
            
            polygon = _mask_to_polygon(region.mask)
            
            if polygon and len(polygon) >= 3:
                draw.polygon(polygon, outline=color, fill=fill_color)
                for i in range(len(polygon)):
                    p1 = polygon[i]
                    p2 = polygon[(i + 1) % len(polygon)]
                    draw.line([p1, p2], fill=color, width=line_width)
            else:
                # 回退到 bbox
                bbox = region.bbox
                x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
                draw.rectangle(
                    [x, y, x + w, y + h],
                    outline=color,
                    width=line_width,
                )
                draw.rectangle(
                    [x, y, x + w, y + h],
                    fill=fill_color,
                )
    
    return img


def grounding_evidence_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    视觉定位证据节点：在灾前和灾后图像上绘制 building 和 road 标注。
    
    输出:
      - grounding_pre.png:  灾前标注图
      - grounding_post.png: 灾后标注图
    """
    print("\n[GroundingEvidence] 开始视觉定位证据生成")
    
    if Image is None:
        print("[GroundingEvidence] 缺少 PIL/Pillow，跳过")
        return {"grounding_evidence": {}}

    stpg = state.get("stpg")
    if stpg is None:
        print("[GroundingEvidence] STPG 为空，跳过")
        return {"grounding_evidence": {}}

    output_dir = Path(state.get("output_dir", "output"))
    evidence_dir = output_dir / "grounding_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    grounding_evidence: Dict[str, str] = {}
    
    # 收集实际出现的损伤等级
    building_labels: set = set()
    road_labels: set = set()
    
    # ==================== 灾前图像 ====================
    pre_image_path = state.get("pre_image_path") or stpg.pre_image_path
    pre_img_path = _resolve_image_path(pre_image_path, output_dir)
    pre_regions = stpg.pre_graph.regions
    
    if pre_img_path and pre_img_path.is_file():
        pre_img = Image.open(pre_img_path).convert("RGB")
        
        pre_building_count = sum(1 for r in pre_regions.values() if _is_building(r.semantic))
        pre_road_count = sum(1 for r in pre_regions.values() if _is_road(r.semantic))
        
        print(f"[GroundingEvidence] 灾前图像: {pre_img_path}")
        print(f"[GroundingEvidence] 灾前 building: {pre_building_count}, road: {pre_road_count}")
        
        if pre_building_count > 0 or pre_road_count > 0:
            pre_annotated = pre_img.copy()
            pre_annotated = _draw_building_and_road(pre_annotated, pre_regions)
            pre_path = evidence_dir / "grounding_pre.png"
            pre_annotated.save(pre_path)
            print(f"[GroundingEvidence] 保存灾前标注图: {pre_path}")
            grounding_evidence["pre_image"] = str(pre_path)
            
            # 收集灾前损伤等级
            for r in pre_regions.values():
                label = r.properties.get("damage_label")
                if label:
                    if _is_building(r.semantic):
                        building_labels.add(label)
                    elif _is_road(r.semantic):
                        road_labels.add(label)
    else:
        print(f"[GroundingEvidence] 找不到灾前图像: {pre_image_path}")
    
    # ==================== 灾后图像 ====================
    post_image_path = state.get("post_image_path") or stpg.post_image_path
    post_img_path = _resolve_image_path(post_image_path, output_dir)
    post_regions = stpg.post_graph.regions
    
    if post_img_path and post_img_path.is_file():
        post_img = Image.open(post_img_path).convert("RGB")
        
        post_building_count = sum(1 for r in post_regions.values() if _is_building(r.semantic))
        post_road_count = sum(1 for r in post_regions.values() if _is_road(r.semantic))
        
        print(f"[GroundingEvidence] 灾后图像: {post_img_path}")
        print(f"[GroundingEvidence] 灾后 building: {post_building_count}, road: {post_road_count}")
        
        if post_building_count > 0 or post_road_count > 0:
            post_annotated = post_img.copy()
            post_annotated = _draw_building_and_road(post_annotated, post_regions)
            post_path = evidence_dir / "grounding_post.png"
            post_annotated.save(post_path)
            print(f"[GroundingEvidence] 保存灾后标注图: {post_path}")
            grounding_evidence["post_image"] = str(post_path)
            
            # 收集灾后损伤等级
            for r in post_regions.values():
                label = r.properties.get("damage_label")
                if label:
                    if _is_building(r.semantic):
                        building_labels.add(label)
                    elif _is_road(r.semantic):
                        road_labels.add(label)
    else:
        print(f"[GroundingEvidence] 找不到灾后图像: {post_image_path}")
    
    img_count = len([k for k in grounding_evidence if k.endswith("_image")])
    print(f"[GroundingEvidence] 生成 {img_count} 张证据图")
    
    # 根据实际存在的损伤等级生成自然语言颜色描述
    # Building: bbox, Road: polygon mask
    building_color_sentences = {
        "Intact": "Green bounding boxes represent intact buildings.",
        "No-damage": "Green bounding boxes represent undamaged buildings.",
        "Minor-damage": "Brown bounding boxes represent minor-damaged buildings.",
        "Damaged": "Orange bounding boxes represent damaged buildings.",
        "Major-damage": "Orange bounding boxes represent major-damaged buildings.",
        "Destroyed": "Red bounding boxes represent destroyed buildings.",
    }
    road_color_sentences = {
        "Intact": "Blue polygon masks represent intact roads.",
        "Flood-submerged": "Yellow polygon masks represent flood-submerged roads.",
        "Debris-covered": "Gray polygon masks represent debris-covered roads.",
    }
    
    desc_parts = []
    for label in building_labels:
        if label in building_color_sentences:
            desc_parts.append(building_color_sentences[label])
    for label in road_labels:
        if label in road_color_sentences:
            desc_parts.append(road_color_sentences[label])
    
    color_description = " ".join(desc_parts) if desc_parts else ""
    grounding_evidence["color_description"] = color_description
    
    # 保存颜色描述到文件
    if color_description:
        color_desc_path = evidence_dir / "color_description.txt"
        with open(color_desc_path, "w", encoding="utf-8") as f:
            f.write(color_description)
        print(f"[GroundingEvidence] 保存颜色描述: {color_desc_path}")
    
    return {
        "grounding_evidence": grounding_evidence,
    }
