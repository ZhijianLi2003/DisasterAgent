"""
经验证据检索 (Experience Evidence) — 离线

对应架构图中 Evidence Retrieval 的第一路：
  Type → Multimodal Embedding → similarity search → disaster case data → rank Top K

实现思路：
  1. 根据灾害类型 + 图像，通过 RAG API 检索最相关的历史灾害案例
  2. 返回 Top-K 经验证据
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.brain import Brain

from agent.skill_router.skills import get_skill
from agent.state import WorkflowState


def experience_evidence_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    经验证据检索节点：通过 RAG API 检索历史灾害案例。
    """
    print(f"\n[ExperienceEvidence] 开始经验证据检索")

    # 从 STPG 获取灾害类型（感知阶段由 DisasterTypeRecognitionTool 写入）
    stpg = state.get("stpg")
    disaster_type = getattr(stpg, "disaster_type", None) if stpg else None
    if not disaster_type:
        disaster_type = "unknown"
    print(f"[ExperienceEvidence] 灾害类型: {disaster_type}")

    # 获取图像路径
    pre_image_path = state.get("pre_image_path")
    post_image_path = state.get("post_image_path")

    if not post_image_path:
        print("[ExperienceEvidence] 缺少灾后图像，跳过检索")
        return {"experience_evidence": []}

    # 获取 RAG 配置
    skill = get_skill(state["skill_name"])
    evidence_config = skill.get_stage_config("evidence")
    rag_config = evidence_config.get("rag", {})

    api_url = rag_config.get("api_url", "http://localhost:8008")
    top_k = rag_config.get("top_k", 3)
    timeout = rag_config.get("timeout", 30)

    print(f"[ExperienceEvidence] RAG 服务: {api_url}")
    print(f"[ExperienceEvidence] 检索 Top-{top_k} 案例")

    # 调用 RAG API
    experience_evidence = []

    try:
        from rag.rag_client import RAGClient

        client = RAGClient(base_url=api_url)

        # 检查服务是否可用
        if not client.is_available():
            print(f"[ExperienceEvidence] RAG 服务不可用: {api_url}")
            return {"experience_evidence": []}

        # 检索相似案例
        results = client.retrieve(
            pre_image_path=pre_image_path or "",
            post_image_path=post_image_path,
            disaster_type=disaster_type,
            top_k=top_k,
        )

        # 转换为 experience_evidence 格式
        for result in results:
            experience_evidence.append({
                "id": result.get("id", ""),
                "similarity": result.get("similarity", 0.0),
                "description": result.get("caption_answer", ""),
                "advice": result.get("advice_answer", ""),
            })

        print(f"[ExperienceEvidence] 检索到 {len(experience_evidence)} 条经验证据")

        # 打印检索结果摘要
        for i, ev in enumerate(experience_evidence, 1):
            print(f"  [{i}] {ev['id']} (similarity: {ev['similarity']:.4f})")
            desc_preview = ev['description'][:80] + "..." if len(ev['description']) > 80 else ev['description']
            print(f"      {desc_preview}")

    except ImportError:
        print("[ExperienceEvidence] 无法导入 RAGClient，跳过检索")
    except Exception as e:
        print(f"[ExperienceEvidence] RAG 检索失败: {e}")

    # 保存到 experience_evidence 文件夹
    if experience_evidence:
        output_dir = Path(state.get("output_dir", "output/unknown"))
        evidence_dir = output_dir / "experience_evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        
        cases_path = evidence_dir / "retrieved_cases.json"
        with open(cases_path, "w", encoding="utf-8") as f:
            json.dump(experience_evidence, f, ensure_ascii=False, indent=2)
        print(f"[ExperienceEvidence] 案例已保存: {cases_path}")

    return {
        "experience_evidence": experience_evidence,
    }
