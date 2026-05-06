"""
STPG (Spatiotemporal Perception Graph) 时空感知图容器类
用于管理多层图结构：灾前感知图、灾后感知图、演化边集
"""

from typing import Optional, List, Tuple, Dict
from .PerceptionGraph import PerceptionGraph
from .EvolutionEdgeSet import EvolutionEdgeSet, EvolutionType


class STPG:
    """
    STPG (Spatiotemporal Perception Graph) 时空感知图容器类
    
    组织和管理多层图结构：
    - pre_graph:       灾前感知图 (空间证据)
    - post_graph:      灾后感知图 (空间证据)
    - evolution_edges: 演化边集   (时间证据)
    - disaster_type:   灾害类型   (全局属性)
    """
    
    def __init__(self, pre_image_path: str = "", post_image_path: str = ""):
        # 灾前/灾后图像路径
        self.pre_image_path: str = pre_image_path
        self.post_image_path: str = post_image_path
        # 灾前感知图
        self.pre_graph: PerceptionGraph = PerceptionGraph(pre_image_path)
        # 灾后感知图
        self.post_graph: PerceptionGraph = PerceptionGraph(post_image_path)
        # 演化边集 (连接灾前↔灾后的时间证据)
        self.evolution_edges: EvolutionEdgeSet = EvolutionEdgeSet()
        # 灾害类型 (全局属性)
        self.disaster_type: Optional[str] = None
    
    # --------------------------------
    # 构建演化边
    # --------------------------------
    def build_evolution_edges(
        self,
        min_iou: float = 0.3,
        top_k: int = 1,
        expand_thresh: float = 1.5,
        shrink_thresh: float = 0.667,
    ):
        """
        基于灾前/灾后感知图，自动构建演化边
        
        Args:
            min_iou:        最小 IoU 阈值
            top_k:          每个灾前区域最多匹配数量
            expand_thresh:  面积比 > 此值 → EXPANDED
            shrink_thresh:  面积比 < 此值 → SHRUNK
        """
        print("[构建演化边]")
        self.evolution_edges.build_from_graphs(
            pre_graph=self.pre_graph,
            post_graph=self.post_graph,
            min_iou=min_iou,
            top_k=top_k,
            expand_thresh=expand_thresh,
            shrink_thresh=shrink_thresh,
        )

    # --------------------------------
    # 序列化 / 反序列化
    # --------------------------------
    def save_json(self, pre_path: str, post_path: str, evo_path: str):
        """
        保存所有子图到 JSON 文件
        
        Args:
            pre_path:  灾前图保存路径
            post_path: 灾后图保存路径
            evo_path:  演化边保存路径
        """
        print("[保存灾前感知图]")
        self.pre_graph.save_json(pre_path)
        print("[保存灾后感知图]")
        self.post_graph.save_json(post_path)
        print("[保存演化边]")
        self.evolution_edges.save_json(evo_path)
    
    @classmethod
    def load_json(cls, pre_path: str, post_path: str, evo_path: str) -> "STPG":
        """
        从 JSON 文件加载所有子图
        
        Args:
            pre_path:  灾前图文件路径
            post_path: 灾后图文件路径
            evo_path:  演化边文件路径
            
        Returns:
            加载后的 STPG 对象
        """
        evidence_graph = cls()
        print("[加载灾前感知图]")
        evidence_graph.pre_graph = PerceptionGraph.load_json(pre_path)
        print("[加载灾后感知图]")
        evidence_graph.post_graph = PerceptionGraph.load_json(post_path)
        print("[加载演化边]")
        evidence_graph.evolution_edges = EvolutionEdgeSet.load_json(evo_path)
        return evidence_graph
    
    # --------------------------------
    # 证据简报
    # --------------------------------
    def _build_perception_brief(self, graph, label: str) -> str:
        """
        构建单个感知图的简报文本
        
        Args:
            graph: PerceptionGraph 实例
            label: 图的标签 (如 "Pre-disaster" / "Post-disaster")
            
        Returns:
            格式化的感知图简报文本
        """
        regions = graph.regions
        lines = [f"[{label} Perception Graph] Total regions: {len(regions)}"]
        
        if not regions:
            lines.append("  (No region data)")
            return "\n".join(lines)
        
        # 按语义分组统计
        semantic_groups: Dict[str, list] = {}
        for rid, region in regions.items():
            sem = region.semantic
            if sem not in semantic_groups:
                semantic_groups[sem] = []
            semantic_groups[sem].append(region)
        
        # 按语义分组输出（不含 bbox 等定位信息）
        for sem, group in sorted(semantic_groups.items()):
            lines.append(f"  [{sem}] {len(group)} regions:")
            for region in group:
                # 基本信息
                info_parts = [f"{region.id}: semantic={region.semantic}"]
                # 添加 damage_label 等业务属性（如果存在）
                if region.properties:
                    damage_label = region.properties.get("damage_label")
                    if damage_label:
                        info_parts.append(f"damage_label={damage_label}")
                lines.append(f"    {', '.join(info_parts)}")
        
        return "\n".join(lines)

    def get_evidence_brief(self) -> str:
        """
        获取完整的证据简报 (供推理智能体使用)
        
        包含两部分:
        1. 灾害类型 (全局属性)
        2. 演化边简报 (按类型分组，包含语义和损伤评估)
        
        Returns:
            格式化的证据简报文本
        """
        sections = []
        
        # ---------- 灾害类型简报 ----------
        if self.disaster_type:
            sections.append(f"[Disaster Type] {self.disaster_type}")
        
        # ---------- 演化边简报 ----------
        edges = self.evolution_edges.edges
        
        if not edges:
            sections.append("[Evolution Edges] Total: 0\n  (No evolution edge data)")
            return "\n\n".join(sections)
        
        # ---------- 类型英文标签 ----------
        type_labels = {
            EvolutionType.MAINTAINED: "MAINTAINED",
            EvolutionType.TRANSITIONED_TO: "TRANSITIONED_TO",
            EvolutionType.DISAPPEARED: "DISAPPEARED",
            EvolutionType.APPEARED: "APPEARED",
            EvolutionType.EXPANDED: "EXPANDED",
            EvolutionType.SHRUNK: "SHRUNK",
        }
        
        # ---------- 辅助: 获取区域语义标签 ----------
        def _get_semantic(graph, region_id: str) -> str:
            """从感知图中获取区域的语义标签"""
            if region_id and region_id in graph.regions:
                return graph.regions[region_id].semantic
            return "unknown"
        
        # ---------- 按类型分组 ----------
        grouped: Dict[EvolutionType, list] = {}
        for etype in EvolutionType:
            grouped[etype] = []
        for edge in edges:
            grouped[edge.evolution_type].append(edge)
        
        # ---------- 构建演化边简报文本 ----------
        lines = [f"[Evolution Edges] Total: {len(edges)}"]
        
        for etype in EvolutionType:
            group = grouped[etype]
            if not group:
                continue
            
            label = type_labels[etype]
            lines.append(f"  [{label}] {len(group)} edges:")
            
            for edge in group:
                pre_sem = _get_semantic(self.pre_graph, edge.pre_id) if edge.pre_id else "null"
                post_sem = _get_semantic(self.post_graph, edge.post_id) if edge.post_id else "null"
                
                if etype == EvolutionType.MAINTAINED:
                    # 存续: 显示语义、评估量
                    if edge.assessment is None:
                        assessment_str = "N/A"
                    elif isinstance(edge.assessment, str):
                        assessment_str = edge.assessment
                    else:
                        assessment_str = f"{edge.assessment:.4f}"
                    lines.append(
                        f"    {edge.id}: {edge.pre_id}(pre,{pre_sem}) -> {edge.post_id}(post,{post_sem}), "
                        f"assessment={assessment_str}"
                    )
                
                elif etype == EvolutionType.TRANSITIONED_TO:
                    # 转变: 显示语义变化
                    lines.append(
                        f"    {edge.id}: {edge.pre_id}(pre,{edge.from_label}) -> {edge.post_id}(post,{edge.to_label})"
                    )
                
                elif etype == EvolutionType.DISAPPEARED:
                    # 语义消失
                    lines.append(
                        f"    {edge.id}: {edge.pre_id}(pre,{pre_sem}) -> null"
                    )
                
                elif etype == EvolutionType.APPEARED:
                    # 语义新生
                    lines.append(
                        f"    {edge.id}: null -> {edge.post_id}(post,{post_sem})"
                    )
                
                elif etype == EvolutionType.EXPANDED:
                    # 扩张: 显示面积比
                    lines.append(
                        f"    {edge.id}: {edge.pre_id}(pre,{pre_sem}) -> {edge.post_id}(post,{post_sem}), "
                        f"area_ratio={edge.area_ratio:.4f}"
                    )
                
                elif etype == EvolutionType.SHRUNK:
                    # 收缩: 显示面积比
                    lines.append(
                        f"    {edge.id}: {edge.pre_id}(pre,{pre_sem}) -> {edge.post_id}(post,{post_sem}), "
                        f"area_ratio={edge.area_ratio:.4f}"
                    )
        
        sections.append("\n".join(lines))
        
        return "\n\n".join(sections)

    # --------------------------------
    # 展示
    # --------------------------------
    def show(self):
        """显示图的统计信息"""
        print("=" * 60)
        print("STPG 时空感知图统计信息")
        print("=" * 60)
        print(f"[灾前图像]   {self.pre_image_path or '未指定'}")
        print(f"[灾后图像]   {self.post_image_path or '未指定'}")
        if self.disaster_type:
            print(f"[灾害类型]   {self.disaster_type}")
        print(f"[灾前感知图] 区域数: {len(self.pre_graph.regions)}, 空间关系数: {len(self.pre_graph.spatial_edges)}")
        print(f"[灾后感知图] 区域数: {len(self.post_graph.regions)}, 空间关系数: {len(self.post_graph.spatial_edges)}")
        evo_counts = self.evolution_edges.count_by_type()
        evo_parts = [f"{k}: {v}" for k, v in evo_counts.items()]
        print(f"[演化边集]   总数: {len(self.evolution_edges.edges)} ({', '.join(evo_parts) if evo_parts else '无'})")
        print("=" * 60)
