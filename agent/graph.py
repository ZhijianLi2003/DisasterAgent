"""
顶层 LangGraph 工作流编排

构建 DisasterAgent 四阶段流水线：
  Skill Router → Perception → Evidence Retrieval → Generation

Brain 实例由外部传入，通过 functools.partial 注入到每个节点。
"""

from __future__ import annotations
from functools import partial

from langgraph.graph import StateGraph, START, END

from agent.brain import Brain
from agent.state import WorkflowState
from agent.skill_router.router import skill_router_node
from agent.perception.plan_execute import perception_node
from agent.evidence.retrieval_graph import evidence_retrieval_node
from agent.generation.report import generation_node


def build_workflow(brain: Brain):
    """
    构建并编译 DisasterAgent 工作流图。

    流程:
      START → skill_router → perception → evidence_retrieval → generation → END

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

    # 定义边
    graph.add_edge(START, "skill_router")
    graph.add_edge("skill_router", "perception")
    graph.add_edge("perception", "evidence_retrieval")
    graph.add_edge("evidence_retrieval", "generation")
    graph.add_edge("generation", END)

    compiled = graph.compile()
    print("[Workflow] DisasterAgent 工作流已编译")
    print("[Workflow] 流程: skill_router → perception → evidence_retrieval → generation")
    return compiled
