"""
工具注册表 (Tool Register)

提供工具的统一注册、查询和实例化机制。
所有工具通过 @register_tool 装饰器自动注册到全局 ToolRegistry。

使用方式：
    from Tool.tookit.tool_register import ToolRegistry, register_tool

    # 注册工具（装饰器方式）
    @register_tool
    class MyTool(BaseTool):
        name = "my_tool"
        description = "..."
        def run(self, **kwargs): ...

    # 查询 / 创建工具
    ToolRegistry.list_names()               # ['sam3_segment', 'dino_extract', 'my_tool']
    tool = ToolRegistry.create("sam3_segment", service_url="http://...")
"""
from typing import Any, Dict, List, Type


class BaseTool:
    """
    工具基类

    所有工具必须继承此类，并实现：
      - name:        工具唯一标识名
      - description: 工具功能的简要描述
      - run():       执行工具逻辑
    """

    name: str = ""
    description: str = ""

    def run(self, **kwargs) -> Any:
        raise NotImplementedError("子类必须实现 run() 方法")

    def __call__(self, **kwargs) -> Any:
        return self.run(**kwargs)

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"


class ToolRegistry:
    """工具注册表 — 管理所有已注册的工具类"""

    _registry: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        """注册一个工具类"""
        if not tool_cls.name:
            raise ValueError(f"{tool_cls.__name__} 缺少 name 属性")
        cls._registry[tool_cls.name] = tool_cls
        return tool_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseTool]:
        """按名称获取工具类（未实例化）"""
        if name not in cls._registry:
            raise KeyError(f"工具 '{name}' 未注册。已注册: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTool:
        """按名称创建工具实例"""
        tool_cls = cls.get(name)
        return tool_cls(**kwargs)

    @classmethod
    def list_tools(cls) -> List[Dict[str, str]]:
        """列举所有已注册工具（名称 + 描述）"""
        return [
            {"name": t.name, "description": t.description}
            for t in cls._registry.values()
        ]

    @classmethod
    def list_names(cls) -> List[str]:
        """列举所有已注册工具名称"""
        return list(cls._registry.keys())

    @classmethod
    def has(cls, name: str) -> bool:
        """判断工具是否已注册"""
        return name in cls._registry


def register_tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """装饰器 — 自动将工具类注册到 ToolRegistry"""
    ToolRegistry.register(cls)
    return cls
