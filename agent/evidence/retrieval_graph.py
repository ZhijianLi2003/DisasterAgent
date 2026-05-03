"""
证据检索编排节点

对应架构图中 Evidence Retrieval 部分，将三类证据检索统一编排：
  1. Experts-Experience Evidence (离线)
  2. Visual-Grounding Evidence (在线)
  3. Semantic-Changing Evidence (在线)

通过 skill config.yaml 中的 evidence 配置节控制开关：
  evidence:
    enable_experience: true/false
    enable_grounding: true/false
    enable_changing: true/false
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.brain import Brain

from agent.state import WorkflowState
from agent.evidence.experience import experience_evidence_node
from agent.evidence.grounding import grounding_evidence_node
from agent.evidence.semantic_change import semantic_change_evidence_node
from agent.skill_router.skills import get_skill


def _get_evidence_config(state: WorkflowState) -> dict:
    """
    从当前 skill 配置中获取 evidence 开关配置
    
    Returns:
        evidence 配置字典，默认全部开启
    """
    default_config = {
        "enable_experience": True,
        "enable_grounding": True,
        "enable_changing": True,
    }
    
    skill_name = state.get("skill_name")
    if not skill_name:
        return default_config
    
    try:
        skill = get_skill(skill_name)
        evidence_config = skill.get_stage_config("evidence")
        return {**default_config, **evidence_config}
    except (KeyError, Exception) as e:
        print(f"[EvidenceRetrieval] 获取 skill 配置失败: {e}，使用默认配置")
        return default_config


def evidence_retrieval_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    证据检索总节点：根据 skill 配置执行三类证据检索，合并结果。

    三类证据之间相互独立，可通过 config.yaml 单独开关：
      - experience_evidence: 离线经验证据
      - grounding_evidence:  在线视觉定位证据
      - changing_evidence:   在线语义变化证据 (含推理+验证)
    """
    print(f"\n{'='*60}")
    print(f"[EvidenceRetrieval] 开始证据检索阶段")
    print(f"{'='*60}")

    # 读取 skill 配置中的证据开关
    evidence_cfg = _get_evidence_config(state)
    enable_experience = evidence_cfg.get("enable_experience", True)
    enable_grounding = evidence_cfg.get("enable_grounding", True)
    enable_changing = evidence_cfg.get("enable_changing", True)

    print(f"[EvidenceRetrieval] 证据开关: experience={enable_experience}, "
          f"grounding={enable_grounding}, changing={enable_changing}")

    merged_result = {
        "current_stage": "evidence_retrieval",
        "experience_evidence": [],
        "grounding_evidence": {},
        "changing_evidence": [],
    }

    # 1. 经验证据 (离线)
    if enable_experience:
        exp_result = experience_evidence_node(state, brain)
        merged_result.update(exp_result)
    else:
        print("[EvidenceRetrieval] 跳过经验证据（已禁用）")

    # 2. 视觉定位证据 (在线)
    if enable_grounding:
        grd_result = grounding_evidence_node(state, brain)
        merged_result.update(grd_result)
    else:
        print("[EvidenceRetrieval] 跳过视觉定位证据（已禁用）")

    # 更新 state 供语义变化节点使用（它需要前面的结果）
    updated_state = {**state, **merged_result}

    # 3. 语义变化证据 (在线，含推理+验证)
    if enable_changing:
        sem_result = semantic_change_evidence_node(updated_state, brain)
        merged_result.update(sem_result)
    else:
        print("[EvidenceRetrieval] 跳过语义变化证据（已禁用）")

    exp_count = len(merged_result.get("experience_evidence", []))
    _ge = merged_result.get("grounding_evidence") or {}
    grd_count = len([k for k in _ge if k.endswith("_image")])
    chg_count = len(merged_result.get("changing_evidence", []))

    print(f"\n[EvidenceRetrieval] 证据检索完成:")
    print(f"  经验证据:   {exp_count} 条 {'(已禁用)' if not enable_experience else ''}")
    print(f"  定位证据:   {grd_count} 条 {'(已禁用)' if not enable_grounding else ''}")
    print(f"  变化证据:   {chg_count} 条 {'(已禁用)' if not enable_changing else ''}")

    return merged_result
