"""
Skill 定义与注册表

每个 Skill 是一个子文件夹（包含 config.yaml + prompt.md），
由 BaseSkill 统一加载，registry 自动扫描注册。

沿用旧版的 BaseSkill + 自动扫描模式，经过验证稳定可靠。
"""

from __future__ import annotations

import re
import yaml
from pathlib import Path
from typing import Optional


# ==================== BaseSkill ====================

class BaseSkill:
    """
    通用 Skill 加载器。
    根据 skill 文件夹路径，自动加载 config.yaml 和 prompt.md。
    """

    def __init__(self, skill_dir: str | Path):
        self.skill_dir = Path(skill_dir)
        if not self.skill_dir.is_dir():
            raise FileNotFoundError(f"Skill 目录不存在: {self.skill_dir}")

        self._config = self._load_config()
        self.name: str = self._config.get("name", self.skill_dir.name)
        self.description: str = self._config.get("description", "")
        self._prompts = self._load_prompts()

    def _load_config(self) -> dict:
        config_path = self.skill_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"缺少 config.yaml: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_prompts(self) -> dict[str, str]:
        """加载 prompt.md，按一级标题 (# xxx) 分段"""
        prompt_path = self.skill_dir / "prompt.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"缺少 prompt.md: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections: dict[str, str] = {}
        pattern = r'^#\s+(.+)$'
        parts = re.split(pattern, content, flags=re.MULTILINE)

        for i in range(1, len(parts), 2):
            title = parts[i].strip().lower()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections[title] = body

        return sections

    def get_prompt(self, stage: str) -> str:
        """获取指定阶段的 prompt"""
        key = stage.strip().lower()
        if key not in self._prompts:
            available = list(self._prompts.keys())
            raise KeyError(
                f"Skill '{self.name}' 中没有阶段 '{stage}' 的 prompt。可用: {available}"
            )
        return self._prompts[key]

    def get_prompt_section(self, stage: str, section: str) -> str:
        """从指定阶段的 prompt 中提取某个二级标题 (## xxx) 的内容"""
        full_prompt = self.get_prompt(stage)
        parts = re.split(r'^##\s+(.+)$', full_prompt, flags=re.MULTILINE)
        target = section.strip().lower()
        for i in range(1, len(parts), 2):
            if parts[i].strip().lower() == target:
                return parts[i + 1].strip() if i + 1 < len(parts) else ""
        return ""

    def get_config(self) -> dict:
        return self._config

    def get_stage_config(self, stage: str) -> dict:
        return self._config.get(stage.strip().lower(), {})

    def __repr__(self) -> str:
        stages = list(self._prompts.keys())
        return f"<Skill name={self.name!r} stages={stages}>"


# ==================== 注册表 ====================

SKILL_REGISTRY: dict[str, BaseSkill] = {}

_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


def _auto_discover():
    """扫描 skills/ 目录，自动注册所有合法的 skill 文件夹"""
    if not _SKILLS_DIR.exists():
        print(f"[SkillRegistry] skills 目录不存在: {_SKILLS_DIR}")
        return

    for item in sorted(_SKILLS_DIR.iterdir()):
        if not item.is_dir():
            continue
        if (item / "config.yaml").exists() and (item / "prompt.md").exists():
            try:
                skill = BaseSkill(item)
                SKILL_REGISTRY[skill.name] = skill
                print(f"[SkillRegistry] 已注册: {skill.name}")
            except Exception as e:
                print(f"[SkillRegistry] 加载失败 ({item.name}): {e}")


def get_skill(name: str) -> BaseSkill:
    """根据名称获取已注册的 Skill 实例"""
    _ensure_loaded()
    if name not in SKILL_REGISTRY:
        available = list(SKILL_REGISTRY.keys())
        raise KeyError(f"Skill '{name}' 未注册。可用: {available}")
    return SKILL_REGISTRY[name]


def list_skills() -> list[dict]:
    """列出所有已注册的 skill"""
    _ensure_loaded()
    return [
        {"name": s.name, "description": s.description}
        for s in SKILL_REGISTRY.values()
    ]


def _ensure_loaded():
    if not SKILL_REGISTRY:
        _auto_discover()


_auto_discover()
