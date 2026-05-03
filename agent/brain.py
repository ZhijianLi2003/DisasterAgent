"""
统一 VLM 大脑

封装全局 VLM/LLM 客户端，贯穿 skill 路由、感知、推理、验证、生成所有阶段。
所有与大模型的交互（文本对话、多模态理解、工具调用规划、JSON 解析）统一由 Brain 完成。

相比旧版的改进:
  - API Key 改为环境变量 / 外部传入，不再硬编码
  - plan_and_execute 保持不变，继续使用 JSON 计划 + 本地执行模式
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class Brain:
    """
    统一的 VLM 大脑。
    一个 Brain 实例 = 一个模型配置，贯穿整个工作流。
    """

    def __init__(
        self,
        model: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.0,
        top_p: float = 0.1,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.api_base = api_base
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking

        if not self.api_key:
            raise ValueError(
                "未提供 API Key。请设置环境变量 DASHSCOPE_API_KEY 或通过 api_key 参数传入。"
            )

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            top_p=top_p,
            api_key=self.api_key,
            base_url=api_base,
            extra_body={"enable_thinking": enable_thinking},
        )

    # ==================== 图像编码工具 ====================

    @staticmethod
    def encode_image(image_path: str) -> dict:
        """将图像文件编码为 LangChain 多模态内容块"""
        img_data = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_data).decode("utf-8")
        suffix = Path(image_path).suffix.lstrip(".").lower()
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "tif": "image/tiff",
            "tiff": "image/tiff",
        }.get(suffix, "image/png")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    @staticmethod
    def text(content: str) -> dict:
        """构建文本内容块"""
        return {"type": "text", "text": content}

    # ==================== 纯文本对话 ====================

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """纯文本对话，返回模型回复文本"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        response = self.llm.invoke(messages)
        return response.content

    # ==================== 多模态对话 ====================

    def chat_with_content(
        self,
        content_parts: list[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        """多模态对话：支持任意顺序的文本与图像组合"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=content_parts))
        response = self.llm.invoke(messages)
        return response.content

    # ==================== JSON 解析 ====================

    @staticmethod
    def parse_json(text: str) -> dict | list:
        """从模型回复中提取并解析 JSON（处理 markdown 代码块等干扰）"""
        text = text.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"无法从模型回复中解析 JSON:\n{text[:500]}")

    # ==================== JSON 对话快捷方法 ====================

    def chat_json(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict | list:
        """纯文本对话 → 自动解析 JSON 回复"""
        raw = self.chat(user_prompt=user_prompt, system_prompt=system_prompt)
        return self.parse_json(raw)

    def chat_with_content_json(
        self,
        content_parts: list[dict],
        system_prompt: Optional[str] = None,
    ) -> dict | list:
        """多模态对话 → 自动解析 JSON 回复"""
        raw = self.chat_with_content(
            content_parts=content_parts, system_prompt=system_prompt
        )
        return self.parse_json(raw)

    # ==================== 工具调用（计划-执行模式） ====================

    @staticmethod
    def _build_tool_descriptions(tools: list) -> str:
        """从工具实例列表自动生成工具描述文本（供模型阅读）"""
        lines = []
        for t in tools:
            sig = inspect.signature(t.run)
            params = []
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                ann = param.annotation
                type_str = (
                    ann.__name__
                    if hasattr(ann, "__name__")
                    else str(ann)
                    if ann != inspect.Parameter.empty
                    else "Any"
                )
                params.append(f"{name}: {type_str}")
            params_str = ", ".join(params)
            lines.append(f"  - {t.name}({params_str}): {t.description}")
        return "\n".join(lines)

    def plan_tool_calls(
        self,
        tools: list,
        user_prompt: Optional[str] = None,
        content_parts: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """
        让模型阅读 prompt + 可用工具描述，以 JSON 输出有序的工具调用计划。
        模型不会真的执行工具，只是「规划」。
        """
        tools_text = self._build_tool_descriptions(tools)

        plan_instruction = (
            "\n\n## Available Tools\n"
            f"{tools_text}\n\n"
            "## Output Requirements\n"
            "Based on the task, plan the tool call order and parameters.\n"
            "Output ONLY the following JSON format:\n"
            "```json\n"
            '{"steps": [\n'
            '  {"step": 1, "tool": "tool_name", "args": {"param1": "value1"}},\n'
            '  {"step": 2, "tool": "tool_name", "args": {"param1": "$step_1", "param2": "value2"}}\n'
            "]}\n"
            "```\n"
            "Rules:\n"
            '- Use "$step_N" to reference the return value of step N\n'
            "- Output ONLY JSON, no explanation\n"
        )

        if content_parts:
            full_parts = list(content_parts) + [Brain.text(plan_instruction)]
            raw = self.chat_with_content(full_parts, system_prompt=system_prompt)
        elif user_prompt:
            raw = self.chat(
                user_prompt=user_prompt + plan_instruction,
                system_prompt=system_prompt,
            )
        else:
            raise ValueError("必须提供 user_prompt 或 content_parts")

        plan_data = self.parse_json(raw)
        steps = (
            plan_data.get("steps", [])
            if isinstance(plan_data, dict)
            else plan_data
        )
        print(f"  [Brain] 模型生成了 {len(steps)} 步工具调用计划")
        return steps

    @staticmethod
    def execute_plan(
        plan: list[dict],
        tools: list,
    ) -> tuple[dict[int, Any], list[dict]]:
        """按计划依次执行工具，支持 "$step_N" 引用前序步骤的返回值"""
        tool_map = {t.name: t for t in tools}
        step_results: dict[int, Any] = {}
        execution_log: list[dict] = []

        for step_info in plan:
            step_num = step_info["step"]
            tool_name = step_info["tool"]
            raw_args = step_info.get("args", {})

            resolved_args: dict[str, Any] = {}
            for k, v in raw_args.items():
                if isinstance(v, str) and v.startswith("$step_"):
                    ref_num = int(v.split("_", 1)[1])
                    if ref_num not in step_results:
                        raise ValueError(
                            f"Step {step_num} 引用了尚未执行的 $step_{ref_num}"
                        )
                    resolved_args[k] = step_results[ref_num]
                else:
                    resolved_args[k] = v

            print(f"  [Brain] Step {step_num}: {tool_name}(...)")

            try:
                tool = tool_map[tool_name]
                result = tool.run(**resolved_args)
                step_results[step_num] = result
                execution_log.append({
                    "step": step_num,
                    "tool": tool_name,
                    "status": "success",
                    "result_type": type(result).__name__,
                })
                print(f"           -> {type(result).__name__}")
            except Exception as e:
                execution_log.append({
                    "step": step_num,
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e),
                })
                print(f"           -> FAILED: {e}")

        return step_results, execution_log

    def plan_and_execute(
        self,
        tools: list,
        user_prompt: Optional[str] = None,
        content_parts: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> tuple[list[dict], dict[int, Any], list[dict]]:
        """
        计划-执行一体化：模型生成 JSON 调用计划 → 代码按顺序执行。

        Returns:
            (plan, step_results, execution_log)
        """
        plan = self.plan_tool_calls(
            tools=tools,
            user_prompt=user_prompt,
            content_parts=content_parts,
            system_prompt=system_prompt,
        )
        step_results, execution_log = self.execute_plan(plan, tools)
        return plan, step_results, execution_log

    def __repr__(self) -> str:
        return (
            f"<Brain model={self.model!r} "
            f"temperature={self.temperature} "
            f"top_p={self.top_p}>"
        )
