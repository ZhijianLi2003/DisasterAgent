"""
全局配置

集中管理 API、模型、路径等配置。
优先读取环境变量，其次使用默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# disasteragent 包根目录
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# 项目根目录 (DisasterAgent/)
PROJECT_ROOT = PACKAGE_ROOT.parent

@dataclass
class Settings:
    """全局配置"""

    # ---- 项目路径 ----
    project_root: Path = PROJECT_ROOT
    package_root: Path = PACKAGE_ROOT
    skills_dir: Path = PACKAGE_ROOT / "skills"
    output_dir: Path = PROJECT_ROOT / "output" / "qwen3_5_27b"


    # ---- VLM / LLM ----
    model: str = "qwen3.5-27b"
    api_key: str = "sk-bc3daa33c369427a96c22e7987869664"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.0
    top_p: float = 0.1
    enable_thinking: bool = False

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


_settings: Settings | None = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
