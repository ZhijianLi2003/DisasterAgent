"""
VQA 回答节点 — 简单问答阶段

用于 pipeline=vqa_short 的任务（如建筑物计数、道路评估），
跳过证据检索与长报告生成，直接将感知阶段统计工具的摘要送给 VLM 生成简短回答。

流程：
  1. 从 state["statistics_summary"] 读取感知阶段统计工具的输出
  2. 加载 skill 的 vqa prompt
  3. 单次 VLM 调用生成简短回答
  4. 写入 state["report"] 并保存到文件
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.brain import Brain

from disasteragent.agent.skill_router.skills import get_skill
from disasteragent.agent.state import WorkflowState


def vqa_answer_node(state: WorkflowState, brain: "Brain") -> dict:
    """
    VQA 回答节点：读取统计摘要 + VLM 单次调用 → 简短回答
    """
    task_id = state.get("task_id", "N/A")
    skill_name = state.get("skill_name", "")
    print(f"\n[VQA Answer] 开始 VQA 回答阶段 — task_id={task_id}, skill={skill_name}")

    # 1. 加载 skill 配置和 vqa prompt
    skill = get_skill(skill_name)
    try:
        vqa_prompt = skill.get_prompt("vqa")
    except KeyError:
        vqa_prompt = (
            "You are a concise disaster assessment assistant. "
            "Based on the statistical evidence provided, answer the user's question briefly and accurately. "
            "Use quantitative data when available. Keep your answer under 50 words."
        )
        print(f"[VQA Answer] skill 中无 # vqa section，使用默认 prompt")

    # 2. 从 state 读取感知阶段统计工具的输出
    statistics_summary = state.get("statistics_summary", "")
    if not statistics_summary:
        print(f"[VQA Answer] ⚠️ statistics_summary 为空，感知阶段可能未调用统计工具")
        return {
            "report": "Error: No statistics data available from perception stage.",
            "current_stage": "vqa_answer",
        }

    print(f"[VQA Answer] 已读取统计摘要（长度: {len(statistics_summary)} 字符）")

    # 3. 组装 VLM 输入
    task_description = state.get("task_description", "")
    user_prompt = (
        f"## User Question\n{task_description}\n\n"
        f"## Statistical Evidence\n{statistics_summary}"
    )

    # 4. VLM 单次调用
    print(f"[VQA Answer] 调用 VLM 生成回答...")
    answer = brain.chat(
        user_prompt=user_prompt,
        system_prompt=vqa_prompt,
    )
    print(f"[VQA Answer] VLM 回答完成")

    # 5. 保存到文件
    output_dir = Path(state.get("output_dir", "output/unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存统计摘要
    stats_path = output_dir / "statistics_summary.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(statistics_summary)
    print(f"[VQA Answer] 统计摘要已保存到: {stats_path}")

    # 保存最终回答
    answer_path = output_dir / "final_report.txt"
    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(answer)
    print(f"[VQA Answer] 最终回答已保存到: {answer_path}")

    print(f"[VQA Answer] VQA 回答阶段完成")

    return {
        "report": answer,
        "current_stage": "vqa_answer",
    }
