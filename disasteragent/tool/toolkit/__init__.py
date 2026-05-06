"""
Tool Toolkit — 工具注册表与工具定义 (离线版本)
"""
from disasteragent.tool.toolkit.tool_register import BaseTool, ToolRegistry, register_tool  # noqa: F401
from disasteragent.tool.toolkit.perception_tool import (  # noqa: F401
    CreateSTPGTool,
    SpatialPerceptionTool,
    TemporalChangePerceptionTool,
    BuildingAssessTool,
    RoadDamageAnalysisTool,
    DisasterTypeRecognitionTool,
)
from disasteragent.tool.toolkit.clac_tool import (  # noqa: F401
    BuildingDisasterCountTool,
    RoadDamagePixelRatioTool,
)