"""
演化边 (EvolutionEdge) 定义
连接灾前感知层与灾后感知层的时间证据
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import json
import re

from .utils import BoundingBox, compute_iou, compute_region_iou, compute_region_area


# ============================================================
# 演化类型枚举
# ============================================================

class EvolutionType(Enum):
    """
    演化边类型
    """
    MAINTAINED = "maintained"               # 存续: 位置重叠 + 语义相同 (物体还在，检查外观是否受损)
    TRANSITIONED_TO = "transitioned_to"     # 转变: 位置重叠 + 语义改变 (地物性质发生质变)
    DISAPPEARED = "disappeared"             # 语义消失: 灾前的语义概念在灾后对应位置不再存在
    APPEARED = "appeared"                   # 语义新生: 灾后出现了灾前对应位置不存在的语义概念
    EXPANDED = "expanded"                   # 扩张: 语义相同 + 面积显著增大
    SHRUNK = "shrunk"                       # 收缩: 语义相同 + 面积显著减小


# ============================================================
# 演化边数据类
# ============================================================

@dataclass
class EvolutionEdge:
    """
    单条演化边 —— 连接灾前区域与灾后区域的时间证据
    
    根据 evolution_type 的不同，关键属性不同：
    - MAINTAINED:      assessment (损伤等级标签等评估量)
    - TRANSITIONED_TO: from_label, to_label (语义发生质变，如 Building -> River)
    - DISAPPEARED:     overlap_ratio (语义消失的残留度，即该位置灾后最大IoU)
    - APPEARED:        overlap_ratio (语义新生的重叠度，即该位置灾前最大IoU)
    - EXPANDED/SHRUNK: area_ratio (面积比 = post_area / pre_area)
    """
    id: str                                                  # 演化边唯一ID (如 "evo_1")
    pre_id: Optional[str]                                   # 灾前区域ID (APPEARED时为None)
    post_id: Optional[str]                                  # 灾后区域ID (DISAPPEARED时为None)
    evolution_type: EvolutionType                           # 演化类型
    iou: float = 0.0                                        # IoU值

    # --- MAINTAINED 属性 ---
    assessment: Optional[Union[float, str]] = None    # 浮点数(如DINO余弦相似度) 或 字符串(如损伤等级标签)

    # --- TRANSITIONED_TO 属性 ---
    from_label: Optional[str] = None                        # 灾前语义标签
    to_label: Optional[str] = None                          # 灾后语义标签

    # --- DISAPPEARED / APPEARED 属性 ---
    overlap_ratio: Optional[float] = None                   # 语义消失的残留度 / 语义新生的重叠度

    # --- EXPANDED / SHRUNK 属性 ---
    area_ratio: Optional[float] = None                      # 面积比 (post / pre)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（只保留非None字段）"""
        d = {
            "id": self.id,
            "pre_id": self.pre_id,
            "post_id": self.post_id,
            "evolution_type": self.evolution_type.value,
            "iou": round(self.iou, 4),
        }
        if self.assessment is not None:
            d["assessment"] = round(self.assessment, 4) if isinstance(self.assessment, (int, float)) else self.assessment
        if self.from_label is not None:
            d["from_label"] = self.from_label
        if self.to_label is not None:
            d["to_label"] = self.to_label
        if self.overlap_ratio is not None:
            d["overlap_ratio"] = round(self.overlap_ratio, 4)
        if self.area_ratio is not None:
            d["area_ratio"] = round(self.area_ratio, 4)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvolutionEdge:
        """从字典恢复"""
        return cls(
            id=d["id"],
            pre_id=d["pre_id"],
            post_id=d.get("post_id"),
            evolution_type=EvolutionType(d["evolution_type"]),
            iou=d.get("iou", 0.0),
            assessment=d.get("assessment", d.get("similarity_score")),
            from_label=d.get("from_label"),
            to_label=d.get("to_label"),
            overlap_ratio=d.get("overlap_ratio"),
            area_ratio=d.get("area_ratio"),
        )

    def describe(self) -> str:
        """生成人类可读的描述"""
        if self.evolution_type == EvolutionType.MAINTAINED:
            if self.assessment is None:
                assessment_str = "N/A"
            elif isinstance(self.assessment, str):
                assessment_str = self.assessment
            else:
                assessment_str = f"{self.assessment:.4f}"
            return f"({self.id}) [存续] {self.pre_id} -> {self.post_id} | IoU={self.iou:.4f}, 评估={assessment_str}"
        
        elif self.evolution_type == EvolutionType.TRANSITIONED_TO:
            return f"({self.id}) [转变] {self.pre_id} -> {self.post_id} | IoU={self.iou:.4f}, {self.from_label} -> {self.to_label}"
        
        elif self.evolution_type == EvolutionType.DISAPPEARED:
            olr = f"{self.overlap_ratio:.4f}" if self.overlap_ratio is not None else "0.0"
            return f"({self.id}) [语义消失] {self.pre_id} -> ∅ | 残留度={olr}"
        
        elif self.evolution_type == EvolutionType.APPEARED:
            olr = f"{self.overlap_ratio:.4f}" if self.overlap_ratio is not None else "0.0"
            return f"({self.id}) [语义新生] ∅ -> {self.post_id} | 重叠度={olr}"
        
        elif self.evolution_type == EvolutionType.EXPANDED:
            return f"({self.id}) [扩张] {self.pre_id} -> {self.post_id} | IoU={self.iou:.4f}, 面积比={self.area_ratio:.4f}"
        
        elif self.evolution_type == EvolutionType.SHRUNK:
            return f"({self.id}) [收缩] {self.pre_id} -> {self.post_id} | IoU={self.iou:.4f}, 面积比={self.area_ratio:.4f}"


# ============================================================
# 演化边集合管理类
# ============================================================

class EvolutionEdgeSet:
    """
    演化边集合 —— 管理所有灾前→灾后的演化边
    
    职责：
    1. 基于灾前/灾后感知图自动构建演化边
    2. 按类型统计和查询
    3. JSON 序列化/反序列化
    """
    
    def __init__(self):
        self.edges: List[EvolutionEdge] = []
        self._edge_counter = 0

    def clear(self):
        """清空所有边"""
        self.edges.clear()
        self._edge_counter = 0

    # --------------------------------
    # 核心方法：从两个感知图构建演化边
    # --------------------------------
    def build_from_graphs(
        self,
        pre_graph,   # PerceptionGraph
        post_graph,  # PerceptionGraph
        min_iou: float = 0.3,
        top_k: int = 1,
        expand_thresh: float = 1.5,
        shrink_thresh: float = 0.667,
    ) -> List[EvolutionEdge]:
        """
        基于灾前/灾后感知图，自动分类并构建演化边
        
        Args:
            pre_graph:      灾前感知图 (PerceptionGraph)
            post_graph:     灾后感知图 (PerceptionGraph)
            min_iou:        IoU 最低阈值，低于此值不视为位置重叠
            top_k:          每个灾前区域最多匹配 top_k 个灾后区域
            expand_thresh:  面积比 > 此值 判定为 EXPANDED  (默认 1.5 即面积增长50%)
            shrink_thresh:  面积比 < 此值 判定为 SHRUNK    (默认 0.667 即面积缩小33%)
            
        Returns:
            构建好的演化边列表
        """
        self.clear()
        
        pre_regions = list(pre_graph.regions.items())
        post_regions = list(post_graph.regions.items())
        
        # if not pre_regions or not post_regions:
        #     print("[演化边构建] 灾前或灾后图为空，跳过")
        #     return []
        
        # 记录已匹配的灾后区域（用于后续扩展 APPEARED 类型）
        matched_post_ids = set()
        
        for pre_id, pre_region in pre_regions:
            # 1) 计算与所有灾后区域的 IoU (优先 mask，退化为 bbox)
            candidates = []
            max_iou_overall = 0.0   # 记录最大IoU（用于 DISAPPEARED 的 overlap_ratio）
            
            for post_id, post_region in post_regions:
                iou = compute_region_iou(
                    pre_region.bbox, post_region.bbox,
                    pre_region.mask, post_region.mask,
                )
                max_iou_overall = max(max_iou_overall, iou)
                if iou >= min_iou:
                    candidates.append((post_id, post_region, iou))
            
            # 按 IoU 降序
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # 2) 无匹配 → 语义消失 (该语义概念在灾后对应位置不再存在)
            if not candidates:
                self._edge_counter += 1
                edge = EvolutionEdge(
                    id=f"evo_{self._edge_counter}",
                    pre_id=pre_id,
                    post_id=None,
                    evolution_type=EvolutionType.DISAPPEARED,
                    iou=0.0,
                    overlap_ratio=round(max_iou_overall, 4),
                )
                self.edges.append(edge)
                #print(f"  {edge.describe()}")
                continue
            
            # 3) 有匹配 → 根据语义和面积分类
            for post_id, post_region, iou in candidates[:top_k]:
                matched_post_ids.add(post_id)
                
                self._edge_counter += 1
                evo_id = f"evo_{self._edge_counter}"
                
                if pre_region.semantic != post_region.semantic:
                    # --- TRANSITIONED_TO ---
                    edge = EvolutionEdge(
                        id=evo_id,
                        pre_id=pre_id,
                        post_id=post_id,
                        evolution_type=EvolutionType.TRANSITIONED_TO,
                        iou=iou,
                        from_label=pre_region.semantic,
                        to_label=post_region.semantic,
                    )
                else:
                    # 语义相同 → 判断面积变化 (优先 mask 面积，退化为 bbox 面积)
                    pre_area = compute_region_area(pre_region.bbox, pre_region.mask)
                    post_area = compute_region_area(post_region.bbox, post_region.mask)
                    area_ratio = post_area / pre_area if pre_area > 0 else 1.0
                    
                    if area_ratio > expand_thresh:
                        # --- EXPANDED ---
                        edge = EvolutionEdge(
                            id=evo_id,
                            pre_id=pre_id,
                            post_id=post_id,
                            evolution_type=EvolutionType.EXPANDED,
                            iou=iou,
                            area_ratio=area_ratio,
                        )
                    elif area_ratio < shrink_thresh:
                        # --- SHRUNK ---
                        edge = EvolutionEdge(
                            id=evo_id,
                            pre_id=pre_id,
                            post_id=post_id,
                            evolution_type=EvolutionType.SHRUNK,
                            iou=iou,
                            area_ratio=area_ratio,
                        )
                    else:
                        # --- MAINTAINED ---
                        # 从 region properties 读取 assessment（优先 post，其次 pre）
                        assessment = None
                        post_props = getattr(post_region, 'properties', None) or {}
                        pre_props = getattr(pre_region, 'properties', None) or {}
                        assessment = post_props.get("damage_label") or pre_props.get("damage_label")
                        
                        edge = EvolutionEdge(
                            id=evo_id,
                            pre_id=pre_id,
                            post_id=post_id,
                            evolution_type=EvolutionType.MAINTAINED,
                            iou=iou,
                            assessment=assessment,
                        )
                
                self.edges.append(edge)
                #print(f"  {edge.describe()}")
        
        # 4) 检查灾后未匹配的区域 → 语义新生 (灾前对应位置不存在该语义概念)
        for post_id, post_region in post_regions:
            if post_id not in matched_post_ids:
                # 计算该灾后区域与所有灾前区域的最大 IoU (作为 overlap_ratio / 重叠度)
                max_iou_to_pre = 0.0
                for _, pre_region in pre_regions:
                    iou = compute_region_iou(
                        post_region.bbox, pre_region.bbox,
                        post_region.mask, pre_region.mask,
                    )
                    max_iou_to_pre = max(max_iou_to_pre, iou)
                
                self._edge_counter += 1
                edge = EvolutionEdge(
                    id=f"evo_{self._edge_counter}",
                    pre_id=None,
                    post_id=post_id,
                    evolution_type=EvolutionType.APPEARED,
                    iou=0.0,
                    overlap_ratio=round(max_iou_to_pre, 4),
                )
                self.edges.append(edge)
                #print(f"  {edge.describe()}")
        
        # 统计
        self._print_summary()
        return self.edges

    # --------------------------------
    # 查询方法
    # --------------------------------
    def get_by_id(self, edge_id: str) -> Optional[EvolutionEdge]:
        """按ID查询演化边"""
        for e in self.edges:
            if e.id == edge_id:
                return e
        return None

    def get_by_type(self, etype: EvolutionType) -> List[EvolutionEdge]:
        """按类型筛选演化边"""
        return [e for e in self.edges if e.evolution_type == etype]

    def count_by_type(self) -> Dict[str, int]:
        """按类型统计数量"""
        counts = {}
        for etype in EvolutionType:
            c = len(self.get_by_type(etype))
            if c > 0:
                counts[etype.value] = c
        return counts

    # --------------------------------
    # 序列化 / 反序列化
    # --------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "evolution_edges": [e.to_dict() for e in self.edges],
            "metadata": {
                "total_edges": len(self.edges),
                "count_by_type": self.count_by_type(),
            }
        }

    def save_json(self, filepath: str):
        """保存到 JSON 文件"""
        data = self.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"  [保存成功] {filepath} (演化边: {len(self.edges)})")

    @classmethod
    def load_json(cls, filepath: str) -> EvolutionEdgeSet:
        """从 JSON 文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        edge_set = cls()
        for edge_data in data["evolution_edges"]:
            edge_set.edges.append(EvolutionEdge.from_dict(edge_data))
        
        # 恢复计数器
        if edge_set.edges:
            max_id = max(
                (int(e.id.split("_")[1]) for e in edge_set.edges if e.id.startswith("evo_")),
                default=0
            )
            edge_set._edge_counter = max_id
        
        print(f"  [加载成功] {filepath} (演化边: {len(edge_set.edges)})")
        return edge_set

    # --------------------------------
    # 展示
    # --------------------------------
    def show(self):
        """显示所有演化边"""
        print(f"[演化边集合] 共 {len(self.edges)} 条")
        for edge in self.edges:
            print(f"  {edge.describe()}")
    
    def _print_summary(self):
        """打印构建摘要"""
        counts = self.count_by_type()
        parts = [f"{k}: {v}" for k, v in counts.items()]
        print(f"[演化边构建完成] 共 {len(self.edges)} 条 ({', '.join(parts)})")
