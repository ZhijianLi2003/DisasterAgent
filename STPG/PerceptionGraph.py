from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json
import re
from datetime import datetime
from .utils import BoundingBox, SpatialRelation, compute_spatial_relation, get_spatial_relation_text,compute_distance
    
@dataclass
class RegionNode:
    """区域节点 - 第一层节点"""
    id: str                                                     # 节点唯一ID
    semantic: str                                               # 语义标签
    bbox: BoundingBox                                           # 边界框
    mask: Optional[Dict[str, Any]] = None                       # RLE掩码 {"size": [H,W], "counts": [...]}
    confidence: float = 0.0                                     # 置信度
    properties: Dict[str, Any] = field(default_factory=dict)    # 附加属性


#证据图的主类
class PerceptionGraph:
    """
    节点：对象 (包含 bbox, mask, 语义)
    边：对象之间的空间位置关系
    """
    def __init__(self,image_path: str = ""):
        #对应的图像的路径
        self.image_path: str = image_path
        # 区域节点(实体)存储 - python对象
        self.regions: Dict[str, RegionNode] = {}
        # 空间关系边(关系)存储 - 三元组格式: (source_id, relation, target_id)
        self.spatial_edges: List[Tuple[str, str, str]] = []
        # 节点计数器
        self._node_counter = 0
    
      
    def add_region(
        self,
        semantic: str,
        bbox: tuple,  # (x, y, width, height)
        mask: Dict[str, Any] = None,  # RLE格式: {"size": [H,W], "counts": [...]}
        confidence: float = 0.0,
        properties: Dict = None,
        k: int = 5   # K近邻参数
    ) -> str:
        """
        添加区域节点，并自动计算与最近K个区域的空间关系（K近邻）
        
        Args:
            semantic: 语义标签（如 "建筑物", "河流"）
            bbox: 边界框 (x, y, width, height)
            mask: 分割掩码（可选）
            confidence: 置信度
            properties: 附加属性
            k: 只与最近的K个区域建立空间关系
            
        Returns:
            新区域的ID
        """
        # 生成节点ID
        self._node_counter += 1
        region_id = f"region_{self._node_counter}"
        
        # 创建边界框对象
        bbox_obj = BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        
        # 创建区域节点
        region = RegionNode(
            id=region_id,
            semantic=semantic,
            bbox=bbox_obj,
            mask=mask,
            confidence=confidence,
            properties=properties or {}
        )
        
        # 计算与已有区域的距离
        new_center = bbox_obj.center()
        distances: List[Tuple[str, float]] = []
        
        for existing_id, existing_region in self.regions.items():
            existing_center = existing_region.bbox.center()
            dist = compute_distance(new_center, existing_center)
            distances.append((existing_id, dist))
        
        # 按距离排序，取最近的K个
        distances.sort(key=lambda x: x[1])
        k_nearest = distances[:k]
        
        # 只与最近的K个区域建立空间关系
        for existing_id, dist in k_nearest:
            existing_region = self.regions[existing_id]
            existing_center = existing_region.bbox.center()
            
            # 计算空间关系
            relation = compute_spatial_relation(existing_center, new_center)
            
            # 添加空间关系边 (三元组格式)
            edge = (existing_id, relation.value, region_id)
            self.spatial_edges.append(edge)
            
            # 打印关系
            desc = get_spatial_relation_text(existing_region.semantic, semantic, relation)
            #print(f"  [添加关系] {desc} (距离: {dist:.1f})")
        
        # 添加到存储
        self.regions[region_id] = region
        neighbor_count = len(k_nearest)
        #print(f"  [添加区域] {region_id}: {semantic} @ bbox{bbox} (关联{neighbor_count}个邻居)")
        
        return region_id
    
    def to_dict(self) -> Dict:
        """
        将图转换为字典格式
        
        Returns:
            字典格式的图数据
        """
        return {
            "image_path": self.image_path,
            "regions": {
                region_id: {
                    "id": region.id,
                    "semantic": region.semantic,
                    "bbox": [region.bbox.x, region.bbox.y, region.bbox.width, region.bbox.height],  # [x, y, w, h]
                    "mask": region.mask,
                    "confidence": region.confidence,
                    "properties": region.properties
                }
                for region_id, region in self.regions.items()
            },
            "spatial_edges": [list(edge) for edge in self.spatial_edges],  # 转为列表
            "metadata": {
                "total_regions": len(self.regions),
                "total_edges": len(self.spatial_edges),
                "created_at": datetime.now().isoformat()
            }
        }
    
    def save_json(self, filepath: str):
        """
        保存图到 JSON 文件
        
        Args:
            filepath: 文件路径（如 "graph.json"）
        """
        data = self.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        # 将短列表压缩到一行（如 bbox 和 spatial_edges 中的三元组）
        # 匹配只包含简单值的短列表
        def compact_list(match):
            content = match.group(0)
            # 移除换行和多余空格，压缩为一行
            compact = re.sub(r'\s+', ' ', content)
            compact = compact.replace('[ ', '[').replace(' ]', ']')
            return compact
        
        # 匹配短列表（不包含嵌套对象的列表）
        json_str = re.sub(
            r'\[\s*\n\s+[\d\.\-]+,\s*\n\s+[\d\.\-]+,\s*\n\s+[\d\.\-]+,\s*\n\s+[\d\.\-]+\s*\n\s*\]',
            compact_list, json_str
        )  # bbox: [x, y, w, h]
        
        json_str = re.sub(
            r'\[\s*\n\s+"[^"]+",\s*\n\s+"[^"]+",\s*\n\s+"[^"]+"\s*\n\s*\]',
            compact_list, json_str
        )  # spatial_edges: ["id", "relation", "id"]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
        #print(f"  [保存成功] {filepath} (区域: {len(self.regions)}, 空间关系: {len(self.spatial_edges)})")
    
    @classmethod
    def load_json(cls, filepath: str) -> "PerceptionGraph":
        """
        从 JSON 文件加载图
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载后的 EviGraph 对象
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建新的图实例
        graph = cls()
        
        # 恢复图像路径
        graph.image_path = data.get("image_path", "")
        
        # 恢复区域节点
        for region_id, region_data in data["regions"].items():
            bbox_data = region_data["bbox"]
            # bbox 现在是列表格式 [x, y, width, height]
            bbox = BoundingBox(
                x=bbox_data[0],
                y=bbox_data[1],
                width=bbox_data[2],
                height=bbox_data[3]
            )
            region = RegionNode(
                id=region_data["id"],
                semantic=region_data["semantic"],
                bbox=bbox,
                mask=region_data["mask"],
                confidence=region_data["confidence"],
                properties=region_data["properties"]
            )
            graph.regions[region_id] = region
        
        # 恢复空间关系边
        graph.spatial_edges = [tuple(t) for t in data["spatial_edges"]]
        
        # 恢复节点计数器
        if graph.regions:
            max_id = max(int(rid.split("_")[1]) for rid in graph.regions.keys())
            graph._node_counter = max_id
        
        print(f"  [加载成功] {filepath} (区域: {len(graph.regions)}, 空间关系: {len(graph.spatial_edges)})")
        
        return graph
