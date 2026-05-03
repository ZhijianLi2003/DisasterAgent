"""
感知工具定义 (Perception Tool Definitions) - 离线版本

所有工具从预计算的 JSON 文件中读取结果，无需在线服务。

目前注册的工具：
  - CreateSTPGTool               : 创建 STPG（输入灾前/灾后图像路径）
  - SpatialPerceptionTool        : 空间分布感知器（从 sam3_concept_segment.json 读取分割结果）
  - TemporalChangePerceptionTool : 时间变化感知器（基于 STPG 构建演化边）
  - BuildingAssessTool           : 建筑物灾害评估（从离线 JSON 读取，支持 4 级 / 3 级标准）
  - RoadDamageAnalysisTool       : 道路损害分析（从 ChangeMamba_Road.json 读取）
  - DisasterTypeRecognitionTool  : 灾害类型识别（从 DINO_DTR.json 读取）
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from STPG.STPG import STPG

from tool.toolkit.tool_register import BaseTool, register_tool
from tool.toolkit.config import (
    SPATIAL_PERCEPTION_CONFIG, TEMPORAL_CHANGE_CONFIG, BUILDING_ASSESS_CONFIG,
    ROAD_DAMAGE_CONFIG, DISASTER_TYPE_CONFIG,
)
from tool.toolkit.utils import (
    extract_filename,
    load_json_cached,
    decode_rle_mask,
    rle_to_bbox,
    normalize_semantic_label,
)


# ============================================================
# 全局缓存
# ============================================================
_JSON_CACHE: Dict[str, Any] = {}


# ============================================================
# 1. 创建 STPG
# ============================================================
@register_tool
class CreateSTPGTool(BaseTool):
    """
    创建 STPG (Spatiotemporal Perception Graph)

    输入：灾前图像路径、灾后图像路径
    输出：STPG 实例（包含空的 pre_graph / post_graph / evolution_edges）
    """

    name = "create_stpg"
    description = "根据灾前/灾后图像路径创建一个空的 STPG (时空感知图)"

    def run(self, pre_image_path: str, post_image_path: str) -> STPG:
        print(f"  → 创建 STPG")
        print(f"    灾前图像: {pre_image_path}")
        print(f"    灾后图像: {post_image_path}")

        stpg = STPG(
            pre_image_path=pre_image_path,
            post_image_path=post_image_path,
        )
        print(f"    ✅ STPG 创建完成")
        return stpg


# ============================================================
# 2. 空间分布感知器 (离线版本)
# ============================================================
@register_tool
class SpatialPerceptionTool(BaseTool):
    """
    空间分布感知器 (离线版本)

    从预计算的 sam3_concept_segment.json 中读取分割结果，
    将分割结果组装到 STPG 的感知图中。

    构造参数：
        offline_json_path: SAM3 分割结果 JSON 文件路径
    """

    name = "spatial_perception"
    description = "从离线 JSON 读取 SAM3 语义分割结果，按指定 concepts 过滤后组装到 STPG 感知图中"

    def __init__(
        self,
        offline_json_path: Path = SPATIAL_PERCEPTION_CONFIG["offline_json_path"],
    ):
        self.offline_json_path = Path(offline_json_path)

    def _load_results(self) -> List[Dict[str, Any]]:
        """加载 SAM3 分割结果 JSON"""
        return load_json_cached(self.offline_json_path, _JSON_CACHE, "sam3_segment")

    def _match_concept(self, category: str, target_concepts: List[str]) -> Optional[str]:
        """
        检查 category 是否匹配 target_concepts 中的任一概念。
        支持模糊匹配（包含关系），返回匹配到的目标概念名，用于统一语义标签。
        
        Args:
            category:        JSON 中的类别名 (如 "buildings", "river or water")
            target_concepts: 用户指定的概念列表 (如 ["buildings", "river or water", "farmland"])
            
        Returns:
            匹配到的目标概念名，未匹配返回 None
        """
        category_lower = category.lower().strip()
        
        for concept in target_concepts:
            concept_lower = concept.lower().strip()
            # 完全匹配
            if category_lower == concept_lower:
                return concept
            # 包含匹配（双向）
            if concept_lower in category_lower or category_lower in concept_lower:
                return concept
        
        return None

    def run(
        self,
        image_path: str,
        phase: str,
        stpg: STPG,
        concepts: Optional[List[str]] = None,
    ) -> STPG:
        """
        从离线 JSON 读取指定图像的分割结果并填充到 STPG 中

        Args:
            image_path: 图片路径
            phase:      "pre" (灾前) 或 "post" (灾后)
            stpg:       已创建的 STPG 实例
            concepts:   要检测的语义概念列表，如 ["buildings", "roads", "farmland"]
                        - None: 提取所有可用的概念
                        - []: 空列表，不提取任何概念，直接返回

        Returns:
            更新后的 STPG 实例
        """
        phase_name = "灾前" if phase == "pre" else "灾后"
        image_filename = extract_filename(image_path)
        print(f"\n  → 空间分布感知器 ({phase_name}): {image_filename}")

        # 空列表：不提取任何概念，直接返回
        if concepts is not None and len(concepts) == 0:
            print(f"    ⏭️ concepts 为空列表，跳过")
            return stpg

        if concepts:
            print(f"    目标概念: {concepts}")

        results = self._load_results()
        record = self._find_record(results, image_filename, phase)

        if record is None:
            print(f"    ⚠️ 未找到匹配记录: {image_filename}")
            return stpg

        image_size = record["image_size"]
        all_concepts = record["concepts"]
        graph = stpg.pre_graph if phase == "pre" else stpg.post_graph

        for category, cat_data in all_concepts.items():
            # 如果指定了 concepts，进行过滤
            if concepts:
                matched_concept = self._match_concept(category, concepts)
                if matched_concept is None:
                    continue
                # 使用匹配到的目标概念名作为语义标签（保持一致性）
                semantic_label = normalize_semantic_label(matched_concept)
            else:
                semantic_label = normalize_semantic_label(category)

            num_masks = cat_data.get("num_masks", 0)
            if num_masks == 0:
                continue

            masks = cat_data["masks"]
            for mask_data in masks:
                bbox = tuple(mask_data["box"])
                score = mask_data["score"]
                rle = decode_rle_mask(mask_data["mask"], image_size)

                graph.add_region(
                    semantic=semantic_label,
                    bbox=bbox,
                    mask=rle,
                    confidence=score,
                    k=3,
                )

        return stpg

    def _find_record(self, results: List[Dict], image_filename: str, phase: str) -> Optional[Dict]:
        """根据文件名和阶段查找记录"""
        for record in results:
            if record["image"] == image_filename and record["image_type"] == phase:
                return record
        return None


# ============================================================
# 3. 时间变化感知器
# ============================================================
@register_tool
class TemporalChangePerceptionTool(BaseTool):
    """
    时间变化感知器

    基于已填充的 STPG（灾前/灾后感知图），构建演化边，
    识别区域的 MAINTAINED / DISAPPEARED / APPEARED 等变化。
    """

    name = "temporal_change_perception"
    description = "基于 STPG 的灾前/灾后感知图构建演化边，识别区域的时空变化关系"

    def __init__(
        self,
        min_iou: float = TEMPORAL_CHANGE_CONFIG["min_iou"],
        top_k: int = TEMPORAL_CHANGE_CONFIG["top_k"],
    ):
        self.min_iou = min_iou
        self.top_k = top_k

    def run(
        self,
        stpg: STPG,
        min_iou: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> STPG:
        iou = min_iou if min_iou is not None else self.min_iou
        k = top_k if top_k is not None else self.top_k

        print(f"\n  → 时间变化感知器")
        print(f"    参数: min_iou={iou}, top_k={k}")

        stpg.build_evolution_edges(min_iou=iou, top_k=k)
        stpg.show()

        print(f"    ✅ 演化边构建完成")
        return stpg


# ============================================================
# 4. 建筑物灾害评估工具 (离线版本)
# ============================================================
@register_tool
class BuildingAssessTool(BaseTool):
    """
    建筑物灾害评估工具 (离线版本)

    工作流程：
      1. 从 SAM3 结果中提取灾前/灾后图像的 building 实例
      2. 合并去重得到场景中的建筑物实例（灾前有灾后没有也算一个建筑物）
      3. 用 ChangeOS/ChangeMamba 的分类掩码进行像素投票确定损伤等级
      4. 将结果写入 STPG

    支持两种分级标准：
      - "4-level": 使用 ChangeOS 结果 (No-damage / Minor-damage / Major-damage / Destroyed)
      - "3-level": 使用 ChangeMamba 结果 (Intact / Damaged / Destroyed)
    """

    name = "building_assess"
    description = "从 SAM3 提取建筑物实例，结合 ChangeOS/ChangeMamba 分类掩码进行损伤评估，写入 STPG"

    def __init__(
        self,
        offline_sam3_path: Path = SPATIAL_PERCEPTION_CONFIG["offline_json_path"],
        offline_json_4level: Path = BUILDING_ASSESS_CONFIG["offline_json_4level"],
        offline_json_3level: Path = BUILDING_ASSESS_CONFIG["offline_json_3level"],
    ):
        self.offline_sam3_path = Path(offline_sam3_path)
        self.offline_json_4level = Path(offline_json_4level)
        self.offline_json_3level = Path(offline_json_3level)

    def _load_sam3_results(self) -> List[Dict[str, Any]]:
        """加载 SAM3 分割结果"""
        return load_json_cached(self.offline_sam3_path, _JSON_CACHE, "sam3_segment")

    def _load_damage_results(self, damage_scale: str) -> List[Dict[str, Any]]:
        """根据分级标准加载对应的损伤分类 JSON 文件"""
        if damage_scale == "4-level":
            return load_json_cached(self.offline_json_4level, _JSON_CACHE, "building_4level")
        else:
            return load_json_cached(self.offline_json_3level, _JSON_CACHE, "building_3level")

    def _find_sam3_record(self, results: List[Dict], image_filename: str, phase: str) -> Optional[Dict]:
        """根据文件名和阶段查找 SAM3 记录"""
        for record in results:
            if record["image"] == image_filename and record["image_type"] == phase:
                return record
        return None

    def _find_damage_record(self, results: List[Dict], pre_filename: str, post_filename: str) -> Optional[Dict]:
        """根据文件名查找损伤分类记录"""
        for record in results:
            if record["pre_image"] == pre_filename or record["post_image"] == post_filename:
                return record
        return None

    def _extract_buildings_from_sam3(self, record: Optional[Dict], image_size: List[int]) -> List[Dict]:
        """从 SAM3 记录中提取 building 类别的掩膜"""
        if record is None:
            return []
        
        concepts = record.get("concepts", {})
        buildings_data = None
        
        for key in concepts:
            if "building" in key.lower():
                buildings_data = concepts[key]
                break
        
        if buildings_data is None or buildings_data.get("num_masks", 0) == 0:
            return []
        
        buildings = []
        for mask_data in buildings_data["masks"]:
            rle = decode_rle_mask(mask_data["mask"], image_size)
            buildings.append({
                "rle": rle,
                "bbox": tuple(mask_data["box"]),
                "score": mask_data["score"],
            })
        
        # 对单时相内的重叠实例做 NMS 去重
        buildings = self._nms_buildings(buildings, iou_thresh=0.5)
        return buildings

    def _nms_buildings(self, buildings: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
        """
        对同一时相内的 building 实例做 NMS（非极大值抑制）去重
        
        按置信度降序排列，高置信度的保留，与其 IoU > 阈值的低置信度实例被抑制
        """
        if len(buildings) <= 1:
            return buildings
        
        # 按置信度降序排序
        sorted_blds = sorted(buildings, key=lambda x: x["score"], reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, bld in enumerate(sorted_blds):
            if i in suppressed:
                continue
            keep.append(bld)
            
            # 解码当前 mask
            mask_i = mask_utils.decode(bld["rle"])
            
            # 抑制与当前 mask IoU > 阈值的后续实例
            for j in range(i + 1, len(sorted_blds)):
                if j in suppressed:
                    continue
                mask_j = mask_utils.decode(sorted_blds[j]["rle"])
                iou = self._compute_mask_iou(mask_i, mask_j)
                if iou > iou_thresh:
                    suppressed.add(j)
        
        return keep

    def _merge_buildings(self, pre_buildings: List[Dict], post_buildings: List[Dict], iou_thresh: float = 0.15) -> List[Dict]:
        """
        合并灾前/灾后的建筑物实例（去重）
        
        策略：以灾前建筑物为基准，灾后建筑物如果与灾前有 IoU > 阈值则视为同一建筑物，
        否则视为新建筑物加入列表
        """
        merged = []
        post_matched = set()
        
        for pre_bld in pre_buildings:
            merged.append(pre_bld)
            pre_mask = mask_utils.decode(pre_bld["rle"])
            
            for i, post_bld in enumerate(post_buildings):
                if i in post_matched:
                    continue
                post_mask = mask_utils.decode(post_bld["rle"])
                iou = self._compute_mask_iou(pre_mask, post_mask)
                if iou > iou_thresh:
                    post_matched.add(i)
        
        for i, post_bld in enumerate(post_buildings):
            if i not in post_matched:
                merged.append(post_bld)
        
        return merged

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个二值掩膜的 IoU"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def _vote_damage_level(self, building_mask: np.ndarray, damage_masks: List[Dict], image_size: List[int]) -> int:
        """
        对建筑物掩膜进行像素投票，确定损伤等级
        
        Args:
            building_mask: 建筑物的二值掩膜
            damage_masks: 损伤分类结果中的所有建筑物掩膜列表
            image_size: 图像尺寸 [H, W]
            
        Returns:
            损伤等级 (0 表示未找到)
        """
        if len(damage_masks) == 0:
            return 0
        
        damage_label_map = np.zeros(image_size, dtype=np.uint8)
        
        for dam_bld in damage_masks:
            dam_rle = decode_rle_mask(dam_bld["mask"], image_size)
            dam_mask = mask_utils.decode(dam_rle)
            damage_level = dam_bld["damage_level"]
            damage_label_map[dam_mask > 0] = damage_level
        
        building_pixels = damage_label_map[building_mask > 0]
        
        if len(building_pixels) == 0:
            return 0
        
        non_zero_pixels = building_pixels[building_pixels > 0]
        if len(non_zero_pixels) == 0:
            return 0
        
        counts = np.bincount(non_zero_pixels)
        return int(np.argmax(counts))

    def run(
        self,
        pre_image_path: str,
        post_image_path: str,
        stpg: STPG,
        damage_scale: Literal["4-level", "3-level"] = "4-level",
    ) -> STPG:
        print(f"\n  → 建筑物灾害评估 (离线模式)")
        print(f"    分级标准: {damage_scale}")

        pre_filename = extract_filename(pre_image_path)
        post_filename = extract_filename(post_image_path)
        print(f"    查询: pre={pre_filename}, post={post_filename}")

        sam3_results = self._load_sam3_results()
        pre_sam3_record = self._find_sam3_record(sam3_results, pre_filename, "pre")
        post_sam3_record = self._find_sam3_record(sam3_results, post_filename, "post")

        if pre_sam3_record is None and post_sam3_record is None:
            print(f"    ⚠️ SAM3 结果中未找到匹配记录")
            return stpg

        image_size = (pre_sam3_record or post_sam3_record)["image_size"]

        pre_buildings = self._extract_buildings_from_sam3(pre_sam3_record, image_size)
        post_buildings = self._extract_buildings_from_sam3(post_sam3_record, image_size)
        print(f"    📊 SAM3 建筑物: 灾前 {len(pre_buildings)} 栋, 灾后 {len(post_buildings)} 栋")

        merged_buildings = self._merge_buildings(pre_buildings, post_buildings)
        print(f"    📊 合并去重后: {len(merged_buildings)} 栋")

        damage_results = self._load_damage_results(damage_scale)
        damage_record = self._find_damage_record(damage_results, pre_filename, post_filename)
        
        damage_masks = []
        if damage_record is not None:
            damage_masks = damage_record.get("buildings", [])
            print(f"    📊 损伤分类掩膜: {len(damage_masks)} 个")
        else:
            print(f"    ⚠️ 未找到损伤分类记录，将标记为 unknown")

        dam_labels = BUILDING_ASSESS_CONFIG[f"dam{'4' if damage_scale == '4-level' else '3'}_labels"]

        building_count_by_level: Dict[str, int] = {}

        for bld in merged_buildings:
            building_mask = mask_utils.decode(bld["rle"])
            
            damage_level = self._vote_damage_level(building_mask, damage_masks, image_size)
            damage_label = dam_labels.get(damage_level, "unknown")
            
            if damage_level == 0:
                damage_label = "unknown"
            
            building_count_by_level[damage_label] = building_count_by_level.get(damage_label, 0) + 1

            rle = bld["rle"]
            bbox = rle_to_bbox(rle)

            properties = {
                "damage_level": damage_level,
                "damage_label": damage_label,
                "source": "sam3+" + ("changeos" if damage_scale == "4-level" else "changemamba"),
            }

            stpg.pre_graph.add_region(
                semantic="building",
                bbox=bbox,
                mask=rle,
                confidence=bld.get("score", 1.0),
                properties=properties,
                k=3,
            )

            stpg.post_graph.add_region(
                semantic="building",
                bbox=bbox,
                mask=rle,
                confidence=bld.get("score", 1.0),
                properties=properties,
                k=3,
            )

        for label, count in building_count_by_level.items():
            total = len(merged_buildings)
            ratio = count / total if total > 0 else 0
            print(f"       {label}: {count} 栋 ({ratio:.1%})")

        print(f"    ✅ 已写入 STPG: pre_graph {len(stpg.pre_graph.regions)} 区域, post_graph {len(stpg.post_graph.regions)} 区域")
        return stpg


# ============================================================
# 5. 道路损害分析工具 (离线版本)
# ============================================================
@register_tool
class RoadDamageAnalysisTool(BaseTool):
    """
    道路损害分析工具 (离线版本)

    从预计算的 ChangeMamba_Road.json 中读取道路损害分析结果，
    并将结果写入 STPG。
    """

    name = "road_damage_analysis"
    description = "从离线 JSON 读取道路损害分析结果，写入 STPG"

    def __init__(
        self,
        offline_json_path: Path = ROAD_DAMAGE_CONFIG["offline_json_path"],
    ):
        self.offline_json_path = Path(offline_json_path)

    def _load_results(self) -> List[Dict[str, Any]]:
        return load_json_cached(self.offline_json_path, _JSON_CACHE, "road_damage")

    def run(
        self,
        pre_image_path: str,
        post_image_path: str,
        stpg: STPG,
    ) -> STPG:
        print(f"\n  → 道路损害分析 (离线模式)")

        pre_filename = extract_filename(pre_image_path)
        post_filename = extract_filename(post_image_path)
        print(f"    查询: pre={pre_filename}, post={post_filename}")

        results = self._load_results()
        record = self._find_record(results, pre_filename, post_filename)

        if record is None:
            print(f"    ⚠️ 未找到匹配记录")
            return stpg

        image_size = record["image_size"]
        roads = record["roads"]
        damage_labels = ROAD_DAMAGE_CONFIG["damage_labels"]

        print(f"    📊 检测到道路: {record['num_roads']} 条")

        road_count_by_level: Dict[str, int] = {}

        for road in roads:
            damage_level = road["damage_level"]
            damage_label = damage_labels.get(damage_level, "unknown")
            road_count_by_level[damage_label] = road_count_by_level.get(damage_label, 0) + 1

            rle = decode_rle_mask(road["mask"], image_size)
            bbox = rle_to_bbox(rle)

            properties = {
                "damage_level": damage_level,
                "damage_label": damage_label,
                "source": "changemamba_road",
            }

            stpg.pre_graph.add_region(
                semantic="road",
                bbox=bbox,
                mask=rle,
                confidence=1.0,
                properties=properties,
                k=3,
            )

            stpg.post_graph.add_region(
                semantic="road",
                bbox=bbox,
                mask=rle,
                confidence=1.0,
                properties=properties,
                k=3,
            )

        for label, count in road_count_by_level.items():
            ratio = count / record['num_roads'] if record['num_roads'] > 0 else 0
            print(f"       {label}: {count} 条 ({ratio:.1%})")

        print(f"    ✅ 已写入 STPG: pre_graph {len(stpg.pre_graph.regions)} 区域, post_graph {len(stpg.post_graph.regions)} 区域")
        return stpg

    def _find_record(self, results: List[Dict], pre_filename: str, post_filename: str) -> Optional[Dict]:
        for record in results:
            if record["pre_image"] == pre_filename or record["post_image"] == post_filename:
                return record
        return None


# ============================================================
# 6. 灾害类型识别工具 (离线版本)
# ============================================================
@register_tool
class DisasterTypeRecognitionTool(BaseTool):
    """
    灾害类型识别工具 (离线版本)

    从预计算的 DINO_DTR.json 中读取灾害类型识别结果。
    支持 10 种灾害类型：earthquake, flood, hurricane, volcano, explosion, 
    wildfire, tornado, tsunami, conflict, landslide
    """

    name = "disaster_type_recognition"
    description = "从离线 JSON 读取灾害类型识别结果"

    def __init__(
        self,
        offline_json_path: Path = DISASTER_TYPE_CONFIG["offline_json_path"],
    ):
        self.offline_json_path = Path(offline_json_path)

    def _load_results(self) -> List[Dict[str, Any]]:
        return load_json_cached(self.offline_json_path, _JSON_CACHE, "disaster_type")

    def run(
        self,
        pre_image_path: str,
        post_image_path: str,
        stpg: Optional[STPG] = None,
    ) -> STPG:
        """
        识别灾害类型，并写入 STPG

        Args:
            pre_image_path:  灾前图像路径
            post_image_path: 灾后图像路径
            stpg:            STPG 实例（可选，若提供则将结果写入）

        Returns:
            STPG: 更新了灾害类型的 STPG 实例
        """
        print(f"\n  → 灾害类型识别 (离线模式)")

        pre_filename = Path(pre_image_path).stem
        post_filename = Path(post_image_path).stem
        print(f"    查询: pre={pre_filename}, post={post_filename}")

        results = self._load_results()

        disaster_type = "unknown"
        confidence = 0.0

        for record in results:
            record_pre = Path(record["pre_image_path"]).stem
            record_post = Path(record["post_image_path"]).stem

            if record_pre == pre_filename or record_post == post_filename:
                disaster_type = record["disaster_type"]
                confidence = record.get("confidence", 1.0)
                print(f"    ✅ 识别结果: {disaster_type} (置信度: {confidence:.4f})")
                break
        else:
            print(f"    ⚠️ 未找到匹配记录，返回 unknown")

        if stpg is not None:
            stpg.disaster_type = disaster_type
            print(f"    ✅ 已写入 STPG: disaster_type={disaster_type}")
            return stpg
        
        # 若未传入 stpg，创建一个临时的 STPG 返回
        new_stpg = STPG(pre_image_path, post_image_path)
        new_stpg.disaster_type = disaster_type
        return new_stpg
