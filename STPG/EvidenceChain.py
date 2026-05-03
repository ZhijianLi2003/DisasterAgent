"""
EvidenceChain - 推理层证据链
推理智能体的输出：声称节点 + 证据引用节点 + 支持关系边
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json


# ============================================================
# 证据引用类型枚举
# ============================================================

class EvidenceRefType(Enum):
    """证据引用类型"""
    EVOLUTION_EDGE = "evolution_edge"    # 引用演化边 (时间证据)
    PRE_REGION = "pre_region"            # 引用灾前感知图区域 (空间证据)
    POST_REGION = "post_region"          # 引用灾后感知图区域 (空间证据)


# ============================================================
# 声称节点
# ============================================================

@dataclass
class ClaimNode:
    """
    声称节点 —— 推理智能体生成的可验证结论
    
    - id:         声称唯一ID
    - text:       声称文本 (短句，可被 VLM 验证)
    - confidence: 置信度 (由验证智能体填写，推理阶段为 None)
    """
    id: str                                     # 声称ID (如 "claim_1")
    text: str                                   # 声称文本
    confidence: Optional[float] = None          # 置信度 (验证智能体填写)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ClaimNode:
        """从字典恢复"""
        return cls(
            id=d["id"],
            text=d["text"],
            confidence=d.get("confidence"),
        )

    def describe(self) -> str:
        """生成可读描述"""
        conf_str = f"{self.confidence:.4f}" if self.confidence is not None else "待验证"
        return f"[{self.id}] {self.text} (置信度: {conf_str})"


# ============================================================
# 证据引用节点
# ============================================================

@dataclass
class EvidenceRefNode:
    """
    证据引用节点 —— 引用感知层/演化边的证据
    
    不复制原始数据，只通过 ref_type + ref_id 指向 STPG 中的已有实体
    
    - id:       证据引用ID (如 "ev_1")
    - ref_type: 引用类型 (evolution_edge / pre_region / post_region)
    - ref_id:   被引用对象的ID (如 "evo:0", "pre:region_12", "post:region_9")
    - note:     可选注释 (1句话说明为什么它支持声称)
    """
    id: str                                     # 引用ID (如 "ev_1")
    ref_type: EvidenceRefType                   # 引用类型
    ref_id: str                                 # 被引用对象的ID
    note: Optional[str] = None                  # 可选注释

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = {
            "id": self.id,
            "ref_type": self.ref_type.value,
            "ref_id": self.ref_id,
        }
        if self.note is not None:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvidenceRefNode:
        """从字典恢复"""
        return cls(
            id=d["id"],
            ref_type=EvidenceRefType(d["ref_type"]),
            ref_id=d["ref_id"],
            note=d.get("note"),
        )

    def describe(self) -> str:
        """生成可读描述"""
        note_str = f" | {self.note}" if self.note else ""
        return f"[{self.id}] {self.ref_type.value} → {self.ref_id}{note_str}"


# ============================================================
# 证据链主类
# ============================================================

class EvidenceChain:
    """
    证据链 —— 推理智能体的输出结构
    
    组成:
    - claims:        声称节点列表 (推理智能体生成的可验证结论)
    - evidence_refs: 证据引用节点列表 (指向感知图/演化边的引用)
    - links:         支持关系边 (三元组: claim_id, "supported_by", evidence_ref_id)
    """

    def __init__(self):
        self.claims: Dict[str, ClaimNode] = {}
        self.evidence_refs: Dict[str, EvidenceRefNode] = {}
        self.links: List[Tuple[str, str, str]] = []     # (claim_id, "supported_by", evidence_ref_id)
        self.verified: bool = False                      # 是否已通过验证智能体验证
        # 计数器
        self._claim_counter = 0
        self._evidence_counter = 0

    def clear(self):
        """清空所有数据"""
        self.claims.clear()
        self.evidence_refs.clear()
        self.links.clear()
        self.verified = False
        self._claim_counter = 0
        self._evidence_counter = 0

    # --------------------------------
    # 添加节点
    # --------------------------------
    def add_claim(self, text: str, claim_id: str = None) -> str:
        """
        添加声称节点
        
        Args:
            text:     声称文本
            claim_id: 可选，自定义ID；若不传则自动生成
            
        Returns:
            声称节点ID
        """
        if claim_id is None:
            self._claim_counter += 1
            claim_id = f"claim_{self._claim_counter}"
        
        claim = ClaimNode(id=claim_id, text=text)
        self.claims[claim_id] = claim
        return claim_id

    def add_evidence_ref(
        self,
        ref_type: EvidenceRefType,
        ref_id: str,
        note: str = None,
        evidence_id: str = None,
    ) -> str:
        """
        添加证据引用节点
        
        Args:
            ref_type:    引用类型 (EvidenceRefType)
            ref_id:      被引用对象ID (如 "evo:0", "pre:region_12")
            note:        可选注释
            evidence_id: 可选，自定义ID；若不传则自动生成
            
        Returns:
            证据引用节点ID
        """
        if evidence_id is None:
            self._evidence_counter += 1
            evidence_id = f"ev_{self._evidence_counter}"
        
        ref = EvidenceRefNode(
            id=evidence_id,
            ref_type=ref_type,
            ref_id=ref_id,
            note=note,
        )
        self.evidence_refs[evidence_id] = ref
        return evidence_id

    # --------------------------------
    # 添加支持关系边
    # --------------------------------
    def link(self, claim_id: str, evidence_id: str):
        """
        添加支持关系: claim ← evidence
        
        Args:
            claim_id:    声称节点ID
            evidence_id: 证据引用节点ID
        """
        self.links.append((claim_id, "supported_by", evidence_id))

    # --------------------------------
    # 查询方法
    # --------------------------------
    def get_evidence_for_claim(self, claim_id: str) -> List[EvidenceRefNode]:
        """获取某条声称的所有支持证据"""
        ev_ids = [link[2] for link in self.links if link[0] == claim_id]
        return [self.evidence_refs[eid] for eid in ev_ids if eid in self.evidence_refs]

    def get_claims_by_evidence(self, evidence_id: str) -> List[ClaimNode]:
        """获取某条证据支持的所有声称"""
        claim_ids = [link[0] for link in self.links if link[2] == evidence_id]
        return [self.claims[cid] for cid in claim_ids if cid in self.claims]

    def get_unverified_claims(self) -> List[ClaimNode]:
        """获取所有未验证的声称 (confidence 为 None)"""
        return [c for c in self.claims.values() if c.confidence is None]

    def update_confidence(self, claim_id: str, confidence: float):
        """
        验证智能体回写置信度
        
        Args:
            claim_id:   声称节点ID
            confidence: 验证后的置信度 (0.0 ~ 1.0)
        """
        if claim_id in self.claims:
            self.claims[claim_id].confidence = confidence

    # --------------------------------
    # 序列化 / 反序列化
    # --------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "verified": self.verified,
            "claims": [c.to_dict() for c in self.claims.values()],
            "evidence_refs": [e.to_dict() for e in self.evidence_refs.values()],
            "links": [list(link) for link in self.links],
            "metadata": {
                "total_claims": len(self.claims),
                "total_evidence_refs": len(self.evidence_refs),
                "total_links": len(self.links),
                "verified_claims": sum(1 for c in self.claims.values() if c.confidence is not None),
                "unverified_claims": sum(1 for c in self.claims.values() if c.confidence is None),
            }
        }

    def save_json(self, filepath: str):
        """保存到 JSON 文件"""
        data = self.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"  [保存成功] {filepath} (声称: {len(self.claims)}, 证据引用: {len(self.evidence_refs)}, 链接: {len(self.links)})")

    @classmethod
    def load_json(cls, filepath: str) -> EvidenceChain:
        """从 JSON 文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chain = cls()
        
        # 恢复验证状态
        chain.verified = data.get("verified", False)
        
        # 恢复声称节点
        for claim_data in data["claims"]:
            claim = ClaimNode.from_dict(claim_data)
            chain.claims[claim.id] = claim
        
        # 恢复证据引用节点
        for ref_data in data["evidence_refs"]:
            ref = EvidenceRefNode.from_dict(ref_data)
            chain.evidence_refs[ref.id] = ref
        
        # 恢复链接
        chain.links = [tuple(link) for link in data["links"]]
        
        # 恢复计数器
        if chain.claims:
            max_claim = max(
                (int(cid.split("_")[1]) for cid in chain.claims if cid.startswith("claim_")),
                default=0
            )
            chain._claim_counter = max_claim
        if chain.evidence_refs:
            max_ev = max(
                (int(eid.split("_")[1]) for eid in chain.evidence_refs if eid.startswith("ev_")),
                default=0
            )
            chain._evidence_counter = max_ev
        
        print(f"  [加载成功] {filepath} (声称: {len(chain.claims)}, 证据引用: {len(chain.evidence_refs)}, 链接: {len(chain.links)})")
        return chain

    # --------------------------------
    # 展示
    # --------------------------------
    def show(self):
        """显示证据链的完整内容"""
        print("=" * 60)
        print("EvidenceChain 证据链")
        print("=" * 60)
        
        # 逐条声称展示，附带其证据
        for claim in self.claims.values():
            print(f"  {claim.describe()}")
            evidences = self.get_evidence_for_claim(claim.id)
            if evidences:
                for ev in evidences:
                    print(f"    └── {ev.describe()}")
            else:
                print(f"    └── (无证据引用)")
        
        # 统计
        verified = sum(1 for c in self.claims.values() if c.confidence is not None)
        total = len(self.claims)
        print("-" * 60)
        print(f"  总计: {total} 条声称, {len(self.evidence_refs)} 条证据引用, {len(self.links)} 条链接")
        print(f"  已验证: {verified}/{total}")
        print("=" * 60)
