"""
DisasterAgent — 灾害遥感图像分析智能体

用法:
    import disasteragent

    result = disasteragent.run(
        pre_image_path="path/to/pre.png",
        post_image_path="path/to/post.png",
        task_description="...",
    )
"""

__all__ = ["run"]


def __getattr__(name):
    """延迟导入，避免 import disasteragent 时立即触发全部依赖"""
    if name == "run":
        from disasteragent.main import run_task
        return run_task
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
