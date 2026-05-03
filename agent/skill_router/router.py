"""
Skill 路由节点

VLM 根据用户任务描述和灾前/灾后图像，选择最合适的 skill。
对应架构图中第①步。

注：灾害类型识别在感知阶段通过 DisasterTypeRecognitionTool 完成。
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.brain import Brain

from agent.skill_router.skills import SKILL_REGISTRY, list_skills, get_skill
from agent.state import WorkflowState


def skill_router_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    Skill 路由节点：VLM 选择合适的 skill。

    输入: state 中的 task_description, pre_image_path, post_image_path
    输出: skill_name
    """
    print(f"\n[SkillRouter] 开始技能路由")

    task_description = state.get("task_description", "")
    pre_image_path = state.get("pre_image_path")
    post_image_path = state.get("post_image_path")

    skills = list_skills()
    skills_info = "\n".join(
        f"  - {s['name']}: {s['description']}"
        for s in skills
    )

    system_prompt = (
        "You are a skill selection assistant for remote sensing disaster analysis.\n"
        "Based on the user's task description and the provided images, "
        "select the most appropriate skill.\n\n"
        "Return ONLY a JSON object, no other text."
    )

    user_text = (
        f"Available skills:\n{skills_info}\n\n"
        f"Task description:\n{task_description}\n\n"
        "Return in the following JSON format:\n"
        "{\n"
        '  "skill_name": "selected skill name"\n'
        "}"
    )

    # 如果有图像，使用多模态对话让 VLM 同时看图
    if post_image_path:
        from agent.brain import Brain as B
        content_parts = []
        if pre_image_path:
            content_parts.append(B.text("Pre-disaster image:"))
            content_parts.append(B.encode_image(pre_image_path))
        content_parts.append(B.text("Post-disaster image:"))
        content_parts.append(B.encode_image(post_image_path))
        content_parts.append(B.text(user_text))

        result = brain.chat_with_content_json(
            content_parts=content_parts,
            system_prompt=system_prompt,
        )
    else:
        result = brain.chat_json(
            user_prompt=user_text,
            system_prompt=system_prompt,
        )

    skill_name = result.get("skill_name", "")

    if skill_name not in SKILL_REGISTRY:
        available = list(SKILL_REGISTRY.keys())
        print(f"[SkillRouter] VLM 返回无效 skill: '{skill_name}'，可用: {available}")
        skill_name = available[0] if available else ""

    print(f"[SkillRouter] skill -> {skill_name}")

    return {
        "skill_name": skill_name,
        "current_stage": "routed",
    }
