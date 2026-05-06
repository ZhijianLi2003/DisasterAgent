"""
Generation 节点 — 报告生成阶段（支持多阶段输出）

对应架构图中第②步的 Generation 部分：
  experience_evidence + grounding_evidence + changing_evidence → 带引用的最终报告

流程：
  1. 从 skill config 读取 output_stages 配置
  2. 对每个 stage：
     - 加载对应的 prompt section
     - 根据 experience_fields 配置组装经验证据
     - 调用 VLM 生成该阶段的输出
  3. 保存所有输出到 generation_outputs 和 report
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from disasteragent.agent.brain import Brain

from disasteragent.agent.skill_router.skills import get_skill
from disasteragent.agent.state import WorkflowState


# ==================== 默认单阶段配置（向后兼容） ====================
DEFAULT_OUTPUT_STAGE = {
    "name": "default",
    "prompt_section": "generation",
    "experience_fields": {"include_description": True, "include_advice": False},
    "output_key": "report",
}


def _grounding_has_content(grounding_evidence: dict) -> bool:
    """是否存在非空的 grounding 证据（dict：路径 + color_description）"""
    if not grounding_evidence:
        return False
    return bool(
        grounding_evidence.get("pre_image")
        or grounding_evidence.get("post_image")
        or str(grounding_evidence.get("color_description", "")).strip()
    )


def _build_evidence_references(changing_evidence: list, threshold: float) -> str:
    """从已验证 claims 中构建可信声称引用列表"""
    if not changing_evidence:
        return "(No verified claims available)"

    lines = []
    for idx, claim in enumerate(changing_evidence, 1):
        text = claim.get("text", "")
        conf = claim.get("confidence", 0.0)
        lines.append(f"[{idx}] {text} (confidence: {conf:.2f})")
    return "\n".join(lines)


def _build_grounding_desc_text(grounding_evidence: dict) -> str:
    """
    GROUNDING-ENHANCED DESCRIPTION 正文（不含外层 === 标题）：
    几何说明 + 颜色含义。图片紧跟在该文本块之后，无需写路径。
    """
    if not grounding_evidence:
        return "(No fine-grained assessment available)"

    lines = [
        "Grounding annotated images: buildings = axis-aligned bounding boxes; "
        "roads = polygon masks from segmentation.",
    ]
    color_desc = (grounding_evidence.get("color_description") or "").strip()
    if color_desc:
        lines.append(color_desc)

    if not color_desc and not grounding_evidence.get("pre_image") and not grounding_evidence.get("post_image"):
        return "(No fine-grained assessment available)"
    return "\n".join(lines)


def _build_experience_text(
    experience_evidence: list,
    *,
    include_description: bool = True,
    include_advice: bool = True,
) -> str:
    """从经验证据构建 VLM 参考文本（字段由 stage experience_fields 控制）"""
    if not experience_evidence or (not include_description and not include_advice):
        return ""

    lines = ["=== EXPERIENCE EVIDENCE (Historical Cases) ==="]
    for idx, case in enumerate(experience_evidence, 1):
        desc = (case.get("description") or "").strip()
        advice = (case.get("advice") or "").strip()
        similarity = case.get("similarity", 0.0)
        case_parts: list[str] = []
        if include_description and desc:
            case_parts.append(f"Description: {desc}")
        if include_advice and advice:
            case_parts.append(f"Recovery guidance (from similar case): {advice}")
        if case_parts:
            lines.append(
                f"[Case {idx}] (similarity: {similarity:.2f})\n" + "\n".join(case_parts)
            )

    lines.append("=== END OF EXPERIENCE EVIDENCE ===")
    if len(lines) == 2:
        return ""
    return "\n".join(lines)


def _generate_single_stage(
    state: WorkflowState,
    brain: "Brain",
    skill,
    stage: dict,
    changing_evidence: list,
    grounding_evidence: dict,
    experience_evidence: list,
    confidence_threshold: float,
) -> str:
    """
    执行单个生成阶段，返回生成的文本。
    
    Args:
        state: 工作流状态
        brain: VLM 大脑实例
        skill: Skill 实例
        stage: 阶段配置 dict (name, prompt_section, experience_fields, output_key)
        changing_evidence: 语义变化证据
        grounding_evidence: 视觉定位证据
        experience_evidence: 经验证据
        confidence_threshold: 置信度阈值
    
    Returns:
        生成的文本内容
    """
    from agent.brain import Brain as B

    stage_name = stage.get("name", "unknown")
    prompt_section = stage.get("prompt_section", "generation")
    exp_fields = stage.get("experience_fields", {})
    exp_inc_desc = exp_fields.get("include_description", True)
    exp_inc_adv = exp_fields.get("include_advice", False)

    print(f"\n[Generation] 执行阶段: {stage_name} (prompt: {prompt_section})")

    # 加载该阶段对应的 prompt
    try:
        gen_prompt = skill.get_prompt(prompt_section)
    except KeyError:
        print(f"[Generation] 警告: 未找到 prompt section '{prompt_section}'，使用 'generation'")
        gen_prompt = skill.get_prompt("generation")

    # 构建证据文本
    evidence_ref_text = _build_evidence_references(changing_evidence, confidence_threshold)
    grounding_desc_body = _build_grounding_desc_text(grounding_evidence)
    experience_text = _build_experience_text(
        experience_evidence,
        include_description=exp_inc_desc,
        include_advice=exp_inc_adv,
    )

    verified_text = "\n".join([
        "=== VERIFIED EVIDENCE (Semantic-Changing References) ===",
        evidence_ref_text,
        "=== END OF EVIDENCE ===",
    ])

    # 构建多模态内容
    pre_image_path = state.get("pre_image_path")
    post_image_path = state.get("post_image_path")

    content_parts = []

    if pre_image_path:
        content_parts.append(B.text("Pre-disaster image:"))
        content_parts.append(B.encode_image(pre_image_path))
        content_parts.append(B.text("Post-disaster image:"))
        content_parts.append(B.encode_image(post_image_path))
    else:
        content_parts.append(B.text("Post-disaster image:"))
        content_parts.append(B.encode_image(post_image_path))

    # 验证证据
    content_parts.append(B.text(verified_text))

    # VISUAL GROUNDING EVIDENCE
    g_pre = grounding_evidence.get("pre_image")
    g_post = grounding_evidence.get("post_image")
    if g_pre or g_post:
        content_parts.append(B.text(
            "\n=== VISUAL GROUNDING EVIDENCE ===\n" + grounding_desc_body
        ))
        if g_pre:
            content_parts.append(B.text("Pre-disaster (annotated):"))
            content_parts.append(B.encode_image(g_pre))
        if g_post:
            content_parts.append(B.text("Post-disaster (annotated):"))
            content_parts.append(B.encode_image(g_post))
        content_parts.append(B.text("=== END OF VISUAL GROUNDING EVIDENCE ==="))

    # 尾部（经验证据，可选）
    if experience_text.strip():
        content_parts.append(B.text("\n" + experience_text))

    # 分离 system_prompt 和 output_format
    main_prompt_parts = re.split(r'^##\s+', gen_prompt, maxsplit=1, flags=re.MULTILINE)
    system_prompt = main_prompt_parts[0].strip()

    if len(main_prompt_parts) > 1:
        output_format_section = "## " + main_prompt_parts[1].strip()
        content_parts.append(B.text("\n" + output_format_section))

    # 调用 Brain 生成
    result = brain.chat_with_content(
        content_parts=content_parts,
        system_prompt=system_prompt,
    )

    print(f"\n[Generation] 阶段 '{stage_name}' 输出:")
    print("-" * 40)
    print(result[:500] + ("..." if len(result) > 500 else ""))
    print("-" * 40)

    return result


def generation_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    生成节点：支持多阶段输出
    
    根据 skill config 中的 output_stages 配置，依次执行每个生成阶段。
    每个阶段可以有不同的 prompt 和经验证据字段配置。
    """
    task_id = state.get("task_id", "N/A")
    print(f"\n[Generation] 开始生成阶段 — task_id={task_id}")

    # 0. 加载 skill 配置
    skill = get_skill(state["skill_name"])
    generation_config = skill.get_stage_config("generation")
    evidence_config = skill.get_stage_config("evidence")
    semantic_change_config = evidence_config.get("semantic_change", {})
    confidence_threshold = semantic_change_config.get("confidence_threshold", 0.7)

    # 获取 output_stages，如果没有则使用默认单阶段
    output_stages = generation_config.get("output_stages", [])
    if not output_stages:
        # 向后兼容：没有 output_stages 配置时使用旧的单阶段逻辑
        exp_prompt_cfg = evidence_config.get("experience_in_prompt", {})
        output_stages = [{
            "name": "default",
            "prompt_section": "generation",
            "experience_fields": {
                "include_description": exp_prompt_cfg.get("include_description", True),
                "include_advice": exp_prompt_cfg.get("include_advice", False),
            },
            "output_key": "report",
        }]
        print("[Generation] 使用默认单阶段配置")
    else:
        print(f"[Generation] 检测到 {len(output_stages)} 个输出阶段")

    # 1. 获取三类证据
    changing_evidence = state.get("changing_evidence", [])
    grounding_evidence = state.get("grounding_evidence") or {}
    if not isinstance(grounding_evidence, dict):
        grounding_evidence = {}
    experience_evidence = state.get("experience_evidence", [])

    # 2. 检查必要输入
    post_image_path = state.get("post_image_path")
    if not post_image_path:
        print("[Generation] 缺少灾后图像，无法生成报告")
        return {
            "current_stage": "generation",
            "report": "",
            "generation_outputs": {},
            "errors": state.get("errors", []) + ["[generation] post_image_path is missing"],
        }

    # 3. 依次执行每个生成阶段
    generation_outputs: dict[str, str] = {}

    for idx, stage in enumerate(output_stages, 1):
        stage_name = stage.get("name", f"stage_{idx}")
        output_key = stage.get("output_key", stage_name)

        print(f"\n[Generation] ===== 阶段 {idx}/{len(output_stages)}: {stage_name} =====")

        try:
            result = _generate_single_stage(
                state=state,
                brain=brain,
                skill=skill,
                stage=stage,
                changing_evidence=changing_evidence,
                grounding_evidence=grounding_evidence,
                experience_evidence=experience_evidence,
                confidence_threshold=confidence_threshold,
            )
            generation_outputs[output_key] = result
        except Exception as e:
            print(f"[Generation] 阶段 '{stage_name}' 执行失败: {e}")
            generation_outputs[output_key] = f"[ERROR] {e}"

    # 4. 构建 references 列表
    references = [
        {"index": idx + 1, "text": c.get("text", ""), "confidence": c.get("confidence", 0.0)}
        for idx, c in enumerate(changing_evidence)
    ]

    # 5. 合并报告（向后兼容 report 字段）
    # 如果只有一个阶段，report = 该阶段输出
    # 如果有多个阶段，report = 所有阶段输出的合并
    if len(generation_outputs) == 1:
        report = list(generation_outputs.values())[0]
    else:
        report_parts = []
        for key, content in generation_outputs.items():
            report_parts.append(f"=== {key.upper()} ===\n{content}")
        report = "\n\n".join(report_parts)

    # 6. 保存输出
    output_dir = Path(state.get("output_dir", "output/unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)

    changing_dir = output_dir / "changing_evidence"
    changing_dir.mkdir(parents=True, exist_ok=True)

    # 保存合并报告
    report_path = str(output_dir / "final_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 保存每个阶段的单独输出
    for key, content in generation_outputs.items():
        stage_path = output_dir / f"{key}.txt"
        with open(stage_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[Generation] 阶段输出已保存: {stage_path}")

    # 保存证据引用
    evidence_ref_text = _build_evidence_references(changing_evidence, confidence_threshold)
    evidence_path = str(changing_dir / "evidence_references.txt")
    with open(evidence_path, "w", encoding="utf-8") as f:
        f.write(evidence_ref_text)

    print(f"\n[Generation] 合并报告已保存: {report_path}")
    print(f"[Generation] 语义变化引用已保存: {evidence_path}")
    print(f"[Generation] 生成阶段完成 — 共 {len(generation_outputs)} 个输出")

    return {
        "current_stage": "generation",
        "report": report,
        "generation_outputs": generation_outputs,
        "references": references,
    }
