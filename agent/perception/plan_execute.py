"""
Perception 节点 — 感知阶段 (Plan-and-Execute)

对应架构图中第②步的 Perception 部分：
  Plan-and-Execute Agent → Function Calling → 构建 STPG

流程：
  1. 加载 skill 的感知阶段 prompt 和 config
  2. 组装上下文（state 中的图像路径 + config 中的参数）
  3. Brain 输出结构化 JSON 工具调用计划
  4. 按计划依次执行工具，构建 STPG
  5. 从执行结果中提取 STPG
  6. 保存工具调用计划和执行日志 (perception_plan.json)
  7. 生成 evidence_brief 并保存 STPG
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.brain import Brain

from agent.skill_router.skills import get_skill
from agent.state import WorkflowState

from tool.toolkit.perception_tool import (
    CreateSTPGTool,
    SpatialPerceptionTool,
    TemporalChangePerceptionTool,
    BuildingAssessTool,
    RoadDamageAnalysisTool,
    DisasterTypeRecognitionTool,
)
from STPG.STPG import STPG


def perception_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    感知节点：Brain 规划工具调用 → 自动执行 → 产出 STPG + evidence_brief
    """
    task_id = state.get("task_id", "N/A")
    print(f"\n[Perception] 开始感知阶段 — task_id={task_id}")

    # 1. 加载 skill 配置
    skill = get_skill(state["skill_name"])
    perception_prompt = skill.get_prompt("perception")
    perception_config = skill.get_stage_config("perception")

    print(f"[Perception] skill={skill.name}, "
          f"concepts={len(perception_config.get('concepts', []))}")

    # 2. 组装上下文
    context = (
        f"## Current task\n"
        f"{state.get('task_description', '')}\n\n"
        f"## Workflow state values\n"
        f"- pre_image_path: {state.get('pre_image_path', 'None')}\n"
        f"- post_image_path: {state.get('post_image_path', '')}\n\n"
        f"## Skill config (perception stage)\n"
        f"- perception.concepts: {perception_config.get('concepts', [])}\n"
        f"- perception.evolution_min_iou: {perception_config.get('evolution_min_iou', 0.3)}\n"
        f"- perception.evolution_top_k: {perception_config.get('evolution_top_k', 1)}\n\n"
        f"## Perception instructions\n"
        f"{perception_prompt}"
    )

    # 3. 创建工具实例
    tools = [
        CreateSTPGTool(),
        SpatialPerceptionTool(),
        TemporalChangePerceptionTool(),
        BuildingAssessTool(),
        RoadDamageAnalysisTool(),
        DisasterTypeRecognitionTool(),
    ]

    # 4. Brain 规划 + 执行
    plan, step_results, execution_log = brain.plan_and_execute(
        tools=tools,
        user_prompt=context,
        system_prompt=(
            "You are a perception planner for remote sensing disaster analysis. "
            "Plan the tool calls based on the instructions and available values."
        ),
    )

    # 5. 从执行结果中提取 STPG
    stpg = None
    for step_num in sorted(step_results.keys(), reverse=True):
        if isinstance(step_results[step_num], STPG):
            stpg = step_results[step_num]
            break

    # 收集错误
    errors = [log for log in execution_log if log["status"] == "error"]
    if errors:
        print(f"[Perception] 有 {len(errors)} 个步骤执行失败")
        for err in errors:
            print(f"  Step {err['step']} ({err['tool']}): {err['error']}")

    # 6. 保存工具调用计划和执行日志
    output_dir = Path(state.get("output_dir", "output/unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)

    perception_plan_log = {
        "plan": plan,
        "execution_log": execution_log,
    }
    plan_log_path = output_dir / "perception_plan.json"
    with open(plan_log_path, "w", encoding="utf-8") as f:
        json.dump(perception_plan_log, f, ensure_ascii=False, indent=2)
    print(f"[Perception] 工具调用计划已保存到: {plan_log_path}")

    # 7. 生成 evidence_brief + 保存 STPG
    perception_brief = ""
    if stpg is not None:
        perception_brief = stpg.get_evidence_brief()

        stpg_dir = output_dir / "STPG"
        stpg_dir.mkdir(parents=True, exist_ok=True)
        stpg.save_json(
            pre_path=str(stpg_dir / "pre_perception_graph.json"),
            post_path=str(stpg_dir / "post_perception_graph.json"),
            evo_path=str(stpg_dir / "evolution_edges.json"),
        )
        print(f"[Perception] STPG 已保存到: {stpg_dir}")
    else:
        print(f"[Perception] STPG 为空，跳过保存")

    print(f"[Perception] 感知阶段完成 — STPG: {type(stpg).__name__}")

    return {
        "current_stage": "perception",
        "stpg": stpg,
        "perception_brief": perception_brief,
        "errors": state.get("errors", []) + [
            f"[perception] step {e['step']} {e['tool']}: {e['error']}"
            for e in errors
        ],
    }
