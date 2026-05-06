"""
语义变化证据 (Semantic-Changing Evidence) — 在线

对应架构图中 Evidence Retrieval 的第三路：
  perception brief → reasoning → Graph node → evolution edges 
  → claims → confidence score → evidence → verification

实现思路（沿用旧版 reasoning + verification 的核心逻辑）：
  1. 从 STPG 获取 evidence_brief (perception brief)
  2. VLM 推理：生成 claims + evidence_refs → 构建 EvidenceChain
  3. VLM 验证：逐条 claim + 原始图像 → 独立验证 → 回写 confidence
  4. 筛选高置信度 claims 作为 changing_evidence
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from disasteragent.agent.brain import Brain

from disasteragent.agent.skill_router.skills import get_skill
from disasteragent.agent.state import WorkflowState
from disasteragent.STPG.EvidenceChain import EvidenceChain, EvidenceRefType


_REF_TYPE_MAP = {
    "evolution_edge": EvidenceRefType.EVOLUTION_EDGE,
    "pre_region": EvidenceRefType.PRE_REGION,
    "post_region": EvidenceRefType.POST_REGION,
}


def _build_evidence_chain(llm_json: dict) -> EvidenceChain:
    """将 VLM 输出的 JSON 解析为 EvidenceChain 数据结构"""
    chain = EvidenceChain()

    for claim_data in llm_json.get("claims", []):
        claim_id = chain.add_claim(
            text=claim_data["text"],
            claim_id=claim_data.get("claim_id"),
        )
        for ref_data in claim_data.get("evidence_refs", []):
            ref_type_str = ref_data["ref_type"]
            ref_type = _REF_TYPE_MAP.get(ref_type_str)
            if ref_type is None:
                print(f"  [Warning] 未知引用类型: {ref_type_str}, 跳过")
                continue
            ev_id = chain.add_evidence_ref(
                ref_type=ref_type,
                ref_id=ref_data["ref_id"],
            )
            chain.link(claim_id, ev_id)

    return chain


def _reasoning_step(
    state: WorkflowState,
    brain: "Brain",
) -> EvidenceChain:
    """推理步骤：evidence_brief → VLM → claims → EvidenceChain"""
    print(f"[SemanticChange] 推理: 生成 claims...")

    evidence_brief = state.get("perception_brief", "")
    if not evidence_brief:
        print("[SemanticChange] evidence_brief 为空，跳过推理")
        return EvidenceChain()

    # 打印灾害简报
    print("\n" + "=" * 60)
    print("PERCEPTION BRIEF (灾害简报)")
    print("=" * 60)
    print(evidence_brief)
    print("=" * 60 + "\n")

    skill = get_skill(state["skill_name"])
    system_prompt = skill.get_prompt_section("evidence retrieval", "semantic change reasoning")

    user_prompt = (
        f"## Evidence Brief\n\n"
        f"{evidence_brief}\n\n"
        f"Please analyze the above Evidence Brief and generate your claims as a JSON object."
    )

    claims_json = brain.chat_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    num_claims = len(claims_json.get("claims", []))
    print(f"[SemanticChange] VLM 返回 {num_claims} 条 claims")

    chain = _build_evidence_chain(claims_json)
    chain.show()
    return chain


def _verification_step(
    state: WorkflowState,
    brain: "Brain",
    chain: EvidenceChain,
) -> EvidenceChain:
    """验证步骤：逐条 claim + 原始图像 → VLM 独立验证"""
    print(f"[SemanticChange] 验证: 逐条验证 claims...")

    unverified = chain.get_unverified_claims()
    if not unverified:
        print("[SemanticChange] 无待验证 claims")
        return chain

    skill = get_skill(state["skill_name"])
    system_prompt = skill.get_prompt_section("evidence retrieval", "semantic change verification")

    pre_image_path = state.get("pre_image_path")
    post_image_path = state.get("post_image_path")

    if not post_image_path:
        print("[SemanticChange] 缺少灾后图像，无法验证")
        return chain

    from agent.brain import Brain as B

    for idx, claim in enumerate(unverified, 1):
        print(f"  [{idx}/{len(unverified)}] {claim.text[:80]}...")

        try:
            content_parts = []
            if pre_image_path:
                content_parts.append(B.text("Image 1 — PRE-DISASTER:"))
                content_parts.append(B.encode_image(pre_image_path))
                content_parts.append(B.text("Image 2 — POST-DISASTER:"))
                content_parts.append(B.encode_image(post_image_path))
            else:
                content_parts.append(B.text("POST-DISASTER image:"))
                content_parts.append(B.encode_image(post_image_path))

            user_text = (
                "Please verify the following claim by ONLY comparing "
                "what you see in the images.\n\n"
                f"[CLAIM]\n{claim.text}\n\n"
                "Assess whether this claim is true and provide your "
                "confidence score as JSON."
            )
            content_parts.append(B.text(user_text))

            result = brain.chat_with_content_json(
                content_parts=content_parts,
                system_prompt=system_prompt,
            )

            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
            chain.update_confidence(claim.id, confidence)
            print(f"    confidence: {confidence:.4f} | {reasoning[:100]}")

        except Exception as e:
            print(f"    FAILED: {e}")
            chain.update_confidence(claim.id, 0.0)

    chain.verified = True
    return chain


def semantic_change_evidence_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    语义变化证据节点：推理 → 构建证据链 → 验证 → 筛选高置信度 claims

    整合了旧版 reasoning_node + verification_node 的逻辑。
    """
    print(f"\n[SemanticChange] 开始语义变化证据检索")

    stpg = state.get("stpg")
    if stpg is None:
        print("[SemanticChange] STPG 为空，跳过")
        return {
            "evidence_chain": None,
            "changing_evidence": [],
        }

    # Step 1: 推理 → EvidenceChain
    chain = _reasoning_step(state, brain)

    # Step 2: 验证
    chain = _verification_step(state, brain, chain)

    # Step 2.5: 添加灾害类型声称（置信度 1.00）
    disaster_type = stpg.disaster_type
    if disaster_type:
        disaster_claim_id = chain.add_claim(
            text=f"The type of disaster that occurred in this area is: {disaster_type}.",
            claim_id="claim_disaster_type",
        )
        chain.update_confidence(disaster_claim_id, 1.00)
        print(f"[SemanticChange] 添加灾害类型声称: {disaster_type} (confidence: 1.00)")

    # Step 3: 保存到 changing_evidence 文件夹
    output_dir = Path(state.get("output_dir", "output/unknown"))
    evidence_dir = output_dir / "changing_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存灾害简报
    perception_brief = state.get("perception_brief", "")
    if perception_brief:
        brief_path = evidence_dir / "perception_brief.txt"
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(perception_brief)
        print(f"[SemanticChange] 灾害简报已保存: {brief_path}")
    
    # 保存证据链
    chain_path = str(evidence_dir / "evidence_chain.json")
    chain.save_json(chain_path)
    print(f"[SemanticChange] 证据链已保存: {chain_path}")

    # Step 4: 筛选高置信度 claims
    skill = get_skill(state["skill_name"])
    evidence_config = skill.get_stage_config("evidence")
    semantic_change_config = evidence_config.get("semantic_change", {})
    threshold = semantic_change_config.get("confidence_threshold", 0.7)

    changing_evidence = [
        c.to_dict()
        for c in chain.claims.values()
        if c.confidence is not None and c.confidence >= threshold
    ]
    changing_evidence.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    print(f"[SemanticChange] 完成 — "
          f"总 claims: {len(chain.claims)}, "
          f"高置信度: {len(changing_evidence)}")

    return {
        "evidence_chain": chain,
        "changing_evidence": changing_evidence,
    }
