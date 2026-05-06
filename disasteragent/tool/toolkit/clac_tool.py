"""
STPG 统计工具 (Calculation Tools) — 基于时空感知图的聚合

从 STPG 灾后感知图（post_graph）中读取建筑物 / 道路节点与掩膜，生成供下游使用的结构化摘要。

目前注册的工具：
  - BuildingDisasterCountTool : 建筑物 Intact / Damaged / Destroyed / Unknown 栋数统计（Unknown 含显式 unknown 与缺失或非法标签）
  - RoadDamagePixelRatioTool   : 道路各类损伤像素占整幅灾后影像（全图 H×W）的比例
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from disasteragent.STPG.STPG import STPG

from disasteragent.tool.toolkit.tool_register import BaseTool, register_tool
from disasteragent.tool.toolkit.config import ROAD_DAMAGE_CONFIG
from disasteragent.tool.toolkit.utils import normalize_semantic_label


# ============================================================
# 7. 建筑物受灾统计
# ============================================================
@register_tool
class BuildingDisasterCountTool(BaseTool):
    """
    建筑物受灾栋数统计

    输入：STPG（仅使用灾后感知图 post_graph）
    输出：英文结构化文本（Intact / Damaged / Destroyed / Unknown）
    """

    name = "building_disaster_count"
    description = (
        "根据 STPG 灾后感知图统计建筑物栋数：Intact、Damaged、Destroyed、Unknown。"
        "Unknown 包括显式 unknown 标签以及缺失或无法识别的 damage_label。返回英文多行摘要。"
    )

    def _norm_damage_label_key(self, label: Optional[str]) -> str:
        """将 damage_label 规范化为小写连字符形式，供匹配使用。"""
        if not label:
            return ""
        return label.strip().lower().replace(" ", "-").replace("_", "-")

    def _damage_to_bucket(self, damage_label: Optional[str]) -> str:
        """
        将建筑物 damage_label 映射为计数桶：Intact / Damaged / Destroyed / Unknown。

        - Intact / Damaged / Destroyed → 对应桶（不区分大小写）
        - 显式 unknown、缺失、空串、无法识别的字符串 → 均计入 Unknown
        """
        if damage_label is None:
            return "Unknown"
        text = str(damage_label).strip()
        if not text:
            return "Unknown"

        key = self._norm_damage_label_key(text)
        if key == "unknown":
            return "Unknown"

        if key == "intact":
            return "Intact"
        if key == "damaged":
            return "Damaged"
        if key == "destroyed":
            return "Destroyed"

        return "Unknown"

    def _build_english_summary(self, stpg: STPG) -> str:
        """在灾后感知图（post_graph）上统计建筑物数量，返回多行英文摘要。"""
        graph = stpg.post_graph
        counts: Dict[str, int] = {
            "Intact": 0,
            "Damaged": 0,
            "Destroyed": 0,
            "Unknown": 0,
        }

        for region in graph.regions.values():
            if normalize_semantic_label(region.semantic) != "building":
                continue
            raw = None
            if region.properties:
                raw = region.properties.get("damage_label")
            if isinstance(raw, str):
                bucket = self._damage_to_bucket(raw)
            else:
                bucket = self._damage_to_bucket(None)
            counts[bucket] = counts.get(bucket, 0) + 1

        total = sum(counts.values())
        lines: List[str] = [
            "[Building disaster counts]",
            f"- Intact: {counts['Intact']} buildings",
            f"- Damaged: {counts['Damaged']} buildings",
            f"- Destroyed: {counts['Destroyed']} buildings",
            f"- Unknown: {counts['Unknown']} buildings",
            f"- Total: {total} buildings",
        ]
        if total == 0:
            lines.append(
                "- Note: No building regions in the graph; run building_assess first."
            )
        return "\n".join(lines)

    def run(self, stpg: STPG) -> str:
        print(f"\n  → 建筑物受灾统计工具（灾后感知图 post_graph）")

        summary = self._build_english_summary(stpg)
        print(f"    统计完成")
        return summary


# ============================================================
# 8. 道路损伤像素占比
# ============================================================
@register_tool
class RoadDamagePixelRatioTool(BaseTool):
    """
    道路损伤像素占比

    输入：STPG（仅使用灾后感知图 post_graph）
    输出：英文结构化文本（各类道路像素占整幅图像像素的比例）
    """

    name = "road_damage_pixel_ratio"
    description = (
        "根据 STPG 灾后感知图中道路区域的 RLE 掩膜，按整幅灾后图像的宽高 H×W 为分母，"
        "计算各类道路像素占全图像素的比例（Intact、Flood-submerged、Debris-covered、Unknown）。"
        "返回英文多行摘要。"
    )

    def _road_damage_labels(self) -> Dict[int, str]:
        """从配置读取道路损伤等级 → 英文名称。"""
        return dict(ROAD_DAMAGE_CONFIG.get("damage_labels", {}))

    def _build_english_summary(self, stpg: STPG) -> str:
        """
        在灾后感知图（post_graph）上统计各类道路像素占整幅图像的比例，返回多行英文摘要。

        分母为灾后图像 H×W（与 mask 尺寸一致）；多道路实例重叠处取较大 damage_level；
        道路并集内无效等级像素计入 Unknown。
        """
        graph = stpg.post_graph
        labels_cfg = self._road_damage_labels()

        road_regions = [
            r
            for r in graph.regions.values()
            if normalize_semantic_label(r.semantic) == "road"
        ]

        def _hw_from_stpg(sample: Optional[dict]) -> Optional[tuple[int, int]]:
            if sample and "size" in sample:
                return int(sample["size"][0]), int(sample["size"][1])
            ipath = (getattr(stpg, "post_image_path", None) or "").strip()
            if not ipath:
                return None
            try:
                from PIL import Image

                with Image.open(ipath) as im:
                    ww, hh = im.size
                return hh, ww
            except Exception:
                return None

        if not road_regions:
            hw = _hw_from_stpg(None)
            if hw is None:
                return "\n".join(
                    [
                        "[Road damage pixel ratios]",
                        "- Error: No road regions and cannot read image size "
                        "(need road masks or valid stpg.post_image_path).",
                    ]
                )
            h, w = hw
            image_px = h * w
            zlines = [
                "[Road damage pixel ratios]",
                f"- Source: STPG post-disaster perception graph; "
                f"denominator = full image {h}×{w} = {image_px} px",
            ]
            for _, name_en in sorted(labels_cfg.items()):
                zlines.append(
                    f"- Share of full image — {name_en}: 0.00% (0 px)"
                )
            zlines.append(
                "- Share of full image — Unknown (invalid or missing damage_level): "
                "0.00% (0 px)"
            )
            zlines.append(
                "- Note: No road regions; run road_damage_analysis first."
            )
            return "\n".join(zlines)

        first_mask = road_regions[0].mask
        if not first_mask or "size" not in first_mask:
            return (
                "[Road damage pixel ratios]\n"
                "- Error: Road regions lack RLE masks; cannot compute ratios."
            )

        h, w = int(first_mask["size"][0]), int(first_mask["size"][1])
        image_px = h * w
        if image_px <= 0:
            return (
                "[Road damage pixel ratios]\n"
                "- Error: Invalid image dimensions from mask."
            )

        label_map = np.zeros((h, w), dtype=np.uint8)
        any_road = np.zeros((h, w), dtype=bool)

        for region in road_regions:
            if not region.mask or "size" not in region.mask:
                continue
            if (
                int(region.mask["size"][0]) != h
                or int(region.mask["size"][1]) != w
            ):
                continue
            m = mask_utils.decode(region.mask)
            if m.shape != label_map.shape:
                continue
            sel = m > 0
            any_road |= sel
            dl = 0
            if region.properties:
                raw = region.properties.get("damage_level", 0)
                try:
                    dl = int(raw)
                except (TypeError, ValueError):
                    dl = 0
            level_to_write = dl if dl in (1, 2, 3) else 0
            if not np.any(sel):
                continue
            prev = label_map[sel]
            if level_to_write > 0:
                label_map[sel] = np.maximum(prev, np.uint8(level_to_write))
            else:
                label_map[sel] = np.where(prev > 0, prev, label_map[sel])

        inside_unknown = any_road & (label_map == 0)
        unknown_px = int(np.sum(inside_unknown))

        lines: List[str] = [
            "[Road damage pixel ratios]",
            f"- Source: STPG post-disaster perception graph; "
            f"denominator = full image {h}×{w} = {image_px} px",
        ]

        for level, name_en in sorted(labels_cfg.items()):
            cnt = int(np.sum((label_map == level) & any_road))
            pct = 100.0 * cnt / image_px if image_px else 0.0
            lines.append(
                f"- Share of full image — {name_en}: {pct:.2f}% ({cnt} px)"
            )

        uk_pct = 100.0 * unknown_px / image_px if image_px else 0.0
        lines.append(
            f"- Share of full image — Unknown (invalid or missing damage_level): "
            f"{uk_pct:.2f}% ({unknown_px} px)"
        )

        return "\n".join(lines)

    def run(self, stpg: STPG) -> str:
        print(f"\n  → 道路损伤像素占比工具（灾后感知图 post_graph）")

        summary = self._build_english_summary(stpg)
        print(f"    占比计算完成")
        return summary
