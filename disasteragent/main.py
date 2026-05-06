"""
DisasterAgent 主入口

用法:
    # 设置环境变量
    set DASHSCOPE_API_KEY=your_api_key

    # 运行
    python main.py
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from disasteragent.config.settings import get_settings, Settings
from disasteragent.agent.brain import Brain
from disasteragent.agent.graph import build_workflow


def run_task(
    pre_image_path: str,
    post_image_path: str,
    task_description: str = "Generate a comprehensive disaster report",
    task_id: Optional[str] = None,
    brain: Optional[Brain] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    运行一次完整的 DisasterAgent 工作流。

    Args:
        pre_image_path:   灾前图像路径
        post_image_path:  灾后图像路径
        task_description: 任务描述
        task_id:          任务ID（不传则自动生成）
        brain:            外部传入的 Brain 实例
        output_dir:       输出目录

    Returns:
        最终状态字典
    """
    settings = get_settings()

    if task_id is None:
        task_id = f"task_{uuid.uuid4().hex[:8]}"

    if output_dir is None:
        output_dir = str(settings.output_dir / task_id)

    if brain is None:
        brain = Brain(
            model=settings.model,
            api_key=settings.api_key,
            api_base=settings.api_base,
            temperature=settings.temperature,
            top_p=settings.top_p,
            enable_thinking=settings.enable_thinking,
        )

    print(f"Brain: {brain}")

    workflow = build_workflow(brain)

    initial_state = {
        "task_id": task_id,
        "task_description": task_description,
        "pre_image_path": pre_image_path,
        "post_image_path": post_image_path,
        "errors": [],
        "current_stage": "",
        "output_dir": output_dir,
    }

    print("=" * 60)
    print(f"启动 DisasterAgent 工作流")
    print(f"  task_id:     {task_id}")
    print(f"  灾前图像:    {pre_image_path}")
    print(f"  灾后图像:    {post_image_path}")
    print(f"  输出目录:    {output_dir}")
    print("=" * 60)

    final_state = workflow.invoke(initial_state)

    print("\n" + "=" * 60)
    print("工作流执行完毕")
    report = final_state.get("report", "(无)")
    print(f"最终报告:\n{report[:500]}")
    if final_state.get("errors"):
        print(f"错误记录: {final_state['errors']}")
    print("=" * 60)

    return final_state


if __name__ == "__main__":
    # result = run_task(
    #     pre_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/nepal_flooding_00000548_pre_disaster.png",
    #     post_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/nepal_flooding_00000548_post_disaster.png",
    #     task_description="Generate a comprehensive disaster report, including the description of the disaster and recovery suggestions.",
    # )

    # result = run_task(
    #     pre_image_path="/local_data/lizhijian/dataset/RSCC_benchmark/lower-puna-volcano/images/lower-puna-volcano_00000002_pre_disaster_part3.png",
    #     post_image_path="/local_data/lizhijian/dataset/RSCC_benchmark/lower-puna-volcano/images/lower-puna-volcano_00000002_post_disaster_part3.png",
    #     task_description="Describe the disaster change on this area.",
    # )

    # result = run_task(
    #     pre_image_path="/local_data/lizhijian/dataset/RSCC_benchmark/nepal-flooding/images/nepal-flooding_00000002_pre_disaster_part2.png",
    #     post_image_path="/local_data/lizhijian/dataset/RSCC_benchmark/nepal-flooding/images/nepal-flooding_00000002_post_disaster_part2.png",
    #     task_description="Describe the disaster change on this area.",
    # )

    # result = run_task(
    #     pre_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/hurricane_florence_00000099_pre_disaster.png",
    #     post_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/hurricane_florence_00000099_post_disaster.png",
    #     #task_description="Analyze both the pre-disaster and post-disaster images to answer the following question. Choose the best option from the candidate options provided. Question: What percentage of the entire image is occupied by undamaged roads? Options: A. 1.84%, B. 2.46%, C. 3.29%, D. 5.27%, E. 8.98%. Your task is to respond with ONLY the capital letter of the correct option (e.g., C). Do not include any explanation or other text.",
    #     task_description="Analyze both the pre-disaster and post-disaster images to answer the following question. Question: What percentage of the entire image is occupied by undamaged roads? ",
    # )

    result = run_task(
        pre_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/joplin_tornado_00000041_pre_disaster.png",
        post_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/joplin_tornado_00000041_post_disaster.png",
        task_description="Analyze both the pre-disaster and post-disaster images to answer the following question. Choose the best option from the candidate options provided. Question: What is the total number of completely destroyed buildings? Options: A:3, B:7, C:14, D:17, E:5. Your task is to respond with ONLY the capital letter of the correct option (e.g., C). Do not include any explanation or other text.",
        #task_description="Analyze both the pre-disaster and post-disaster images to answer the following question. Question: What is the total number of completely destroyed buildings? ",
    )
