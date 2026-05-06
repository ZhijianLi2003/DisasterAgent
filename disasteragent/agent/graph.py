"""
顶层 LangGraph 工作流编排

构建 DisasterAgent 工作流图，支持两种流水线：
  - full（默认）: skill_router → perception → evidence_retrieval → generation → END
  - vqa_short:    skill_router → perception → vqa_answer → END

Brain 实例由外部传入，通过 functools.partial 注入到每个节点。
"""

from __future__ import annotations
from functools import partial

from langgraph.graph import StateGraph, START, END

from disasteragent.agent.brain import Brain
from disasteragent.agent.state import WorkflowState
from disasteragent.agent.skill_router.router import skill_router_node
from disasteragent.agent.skill_router.skills import get_skill
from disasteragent.agent.perception.plan_execute import perception_node
from disasteragent.agent.evidence.retrieval_graph import evidence_retrieval_node
from disasteragent.agent.generation.report import generation_node
from disasteragent.agent.generation.vqa_answer import vqa_answer_node


def _route_after_perception(state: WorkflowState) -> str:
    """
    感知后路由：根据 skill 配置的 pipeline 字段决定下一步。

    - pipeline == "vqa_short" → vqa_answer（简单问答）
    - 否则 → evidence_retrieval（完整流水线）
    """
    skill_name = state.get("skill_name", "")
    if not skill_name:
        return "evidence_retrieval"

    try:
        skill = get_skill(skill_name)
        pipeline = skill.get_config().get("pipeline", "full")
    except KeyError:
        pipeline = "full"

    if pipeline == "vqa_short":
        print(f"[Workflow] pipeline=vqa_short → 跳转 vqa_answer")
        return "vqa_answer"
    else:
        print(f"[Workflow] pipeline=full → 继续 evidence_retrieval")
        return "evidence_retrieval"


def build_workflow(brain: Brain):
    """
    构建并编译 DisasterAgent 工作流图。

    流程（含条件分支）:
      START → skill_router → perception → [条件路由]
                                             ├─ vqa_short → vqa_answer → END
                                             └─ full → evidence_retrieval → generation → END

    Args:
        brain: 统一的 VLM 大脑实例，注入到所有节点

    Returns:
        已编译的 LangGraph Runnable
    """
    graph = StateGraph(WorkflowState)

    # 注册节点（通过 partial 注入 brain）
    graph.add_node("skill_router",       partial(skill_router_node, brain=brain))
    graph.add_node("perception",         partial(perception_node, brain=brain))
    graph.add_node("evidence_retrieval", partial(evidence_retrieval_node, brain=brain))
    graph.add_node("generation",         partial(generation_node, brain=brain))
    graph.add_node("vqa_answer",         partial(vqa_answer_node, brain=brain))

    # 定义边
    graph.add_edge(START, "skill_router")
    graph.add_edge("skill_router", "perception")

    # 感知后条件路由
    graph.add_conditional_edges(
        "perception",
        _route_after_perception,
        {
            "vqa_answer": "vqa_answer",
            "evidence_retrieval": "evidence_retrieval",
        },
    )

    # 完整流水线后续
    graph.add_edge("evidence_retrieval", "generation")
    graph.add_edge("generation", END)

    # VQA 短流水线直接结束
    graph.add_edge("vqa_answer", END)

    compiled = graph.compile()
    print("[Workflow] DisasterAgent 工作流已编译")
    print("[Workflow] 支持 full / vqa_short 两种流水线")
    return compiled
