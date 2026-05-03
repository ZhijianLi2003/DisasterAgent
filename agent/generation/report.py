"""
Generation 节点 — 报告生成阶段

对应架构图中第②步的 Generation 部分：
  experience_evidence + grounding_evidence + changing_evidence → 带引用的最终报告

流程：
  1. 从三类证据中构建引用文本
  2. 结合灾前/灾后图像 + 证据引用 → VLM 生成最终报告
  3. 保存报告和引用
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.brain import Brain

from agent.skill_router.skills import get_skill
from agent.state import WorkflowState


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
    """从经验证据构建 VLM 参考文本（字段由 skill config experience_in_prompt 控制）"""
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
    # 仅有标题无有效案例内容时不向 VLM 注入空块
    if len(lines) == 2:
        return ""
    return "\n".join(lines)


def generation_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    生成节点：整合三类证据 → 生成带引用的灾害评估报告
    """
    task_id = state.get("task_id", "N/A")
    print(f"\n[Generation] 开始生成阶段 — task_id={task_id}")

    # 0. 加载 skill 配置
    skill = get_skill(state["skill_name"])
    gen_prompt = skill.get_prompt("generation")
    evidence_config = skill.get_stage_config("evidence")
    semantic_change_config = evidence_config.get("semantic_change", {})
    confidence_threshold = semantic_change_config.get("confidence_threshold", 0.7)
    exp_prompt_cfg = evidence_config.get("experience_in_prompt", {})
    exp_inc_desc = exp_prompt_cfg.get("include_description", True)
    exp_inc_adv = exp_prompt_cfg.get("include_advice", True)

    # 1. 构建三类证据的引用文本
    changing_evidence = state.get("changing_evidence", [])
    grounding_evidence = state.get("grounding_evidence") or {}
    if not isinstance(grounding_evidence, dict):
        grounding_evidence = {}
    experience_evidence = state.get("experience_evidence", [])

    evidence_ref_text = _build_evidence_references(changing_evidence, confidence_threshold)
    grounding_desc_body = _build_grounding_desc_text(grounding_evidence)
    experience_text = _build_experience_text(
        experience_evidence,
        include_description=exp_inc_desc,
        include_advice=exp_inc_adv,
    )

    # 2. verified 证据块与尾部经验文本
    verified_text = "\n".join(
        [
            "=== VERIFIED EVIDENCE (Semantic-Changing References) ===",
            evidence_ref_text,
            "=== END OF EVIDENCE ===",
        ]
    )

    tail_parts: list[str] = []
    if experience_text.strip():
        tail_parts.extend(["", experience_text])
    tail_text = "\n".join(tail_parts)

    # 3. 构建多模态内容
    pre_image_path = state.get("pre_image_path")
    post_image_path = state.get("post_image_path")

    if not post_image_path:
        print("[Generation] 缺少灾后图像，无法生成报告")
        return {
            "current_stage": "generation",
            "report": "",
            "errors": state.get("errors", []) + ["[generation] post_image_path is missing"],
        }

    from agent.brain import Brain as B

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

    # VISUAL GROUNDING EVIDENCE（说明 + 图片 包在一个块里）
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
    if tail_text.strip():
        content_parts.append(B.text(tail_text))

    # 分离 system_prompt 和 output_format
    # 取 gen_prompt 中第一个 ## 之前的主体部分作为 system_prompt
    # ## Output Format 及之后的内容作为 user content 的最后部分
    main_prompt_parts = re.split(r'^##\s+', gen_prompt, maxsplit=1, flags=re.MULTILINE)
    system_prompt = main_prompt_parts[0].strip()

    # 把 ## Output Format 部分加到 user content 最后
    if len(main_prompt_parts) > 1:
        output_format_section = "## " + main_prompt_parts[1].strip()
        content_parts.append(B.text("\n" + output_format_section))

    # 4. 调用 Brain 生成报告
    report = brain.chat_with_content(
        content_parts=content_parts,
        system_prompt=system_prompt,
    )

    print(f"\n[Generation] 灾害评估报告:")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # 5. 构建 references 列表
    references = [
        {"index": idx + 1, "text": c.get("text", ""), "confidence": c.get("confidence", 0.0)}
        for idx, c in enumerate(changing_evidence)
    ]

    # 6. 保存
    output_dir = Path(state.get("output_dir", "output/unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)

    changing_dir = output_dir / "changing_evidence"
    changing_dir.mkdir(parents=True, exist_ok=True)

    report_path = str(output_dir / "final_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    evidence_path = str(changing_dir / "evidence_references.txt")
    with open(evidence_path, "w", encoding="utf-8") as f:
        f.write(evidence_ref_text)

    print(f"[Generation] 报告已保存: {report_path}")
    print(f"[Generation] 生成阶段完成")

    return {
        "current_stage": "generation",
        "report": report,
        "references": references,
    }
