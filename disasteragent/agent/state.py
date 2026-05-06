"""
LangGraph 共享状态定义

所有节点通过读写此 State 进行数据流转。
相比旧版 (example_old) 的主要变化:
  - EvidenceGraph → STPG
  - 新增三类证据字段 (experience / grounding / changing)
  - disaster_type 现在由感知阶段的 DisasterTypeRecognitionTool 写入 STPG
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class WorkflowState(TypedDict, total=False):
    """DisasterAgent 工作流的共享状态"""

    # ==================== 输入 ====================
    task_id: str
    task_description: str
    pre_image_path: str
    post_image_path: str

    # ==================== Skill 路由 ====================
    skill_name: str

    # ==================== Perception 阶段 ====================
    stpg: Any                           # STPG 实例
    perception_brief: str               # stpg.get_evidence_brief() 输出
    statistics_summary: str             # 统计工具输出的英文摘要（供 VQA 使用）

    # ==================== Evidence Retrieval 阶段 ====================
    experience_evidence: list
    grounding_evidence: dict  # pre_image / post_image / color_description（str 路径与说明）
    # 语义变化证据 (推理 → claims → 验证)
    evidence_chain: Any                 # EvidenceChain 实例
    changing_evidence: list             # 已验证的高置信度 claims

    # ==================== Generation 阶段 ====================
    report: str                         # 向后兼容：单阶段输出或多阶段合并
    references: list
    generation_outputs: dict            # 多阶段输出：{output_key: content, ...}

    # ==================== 元信息 ====================
    errors: list
    current_stage: str
    output_dir: str
