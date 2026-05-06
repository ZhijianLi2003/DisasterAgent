"""
工具构造参数配置 (Toolkit Config)

集中定义每个工具的构造参数，供实例化工具时使用。
修改工具参数只需改此文件，无需改动 tool.py 中的工具逻辑。

使用方式：
    from Tool.tookit.config import TOOL_CONFIGS
    from Tool.tookit.tool_register import ToolRegistry

    # 按名称创建工具，自动传入构造参数
    tool = ToolRegistry.create("spatial_perception", **TOOL_CONFIGS["spatial_perception"])
"""
from pathlib import Path


# ========================== 路径基准 =============================
TOOL_ROOT = Path(__file__).resolve().parent.parent  # 指向 tool/
OFFLINE_DATA_DIR = TOOL_ROOT / "server" / "off_line"  # 离线数据目录


# ========================== 离线数据文件路径 =============================
OFFLINE_DATA_PATHS = {
    "sam3_segment":         OFFLINE_DATA_DIR / "sam3_concept_segment.json",
    "dino_dtr":             OFFLINE_DATA_DIR / "DINO_DTR.json",
    "changeos_4level":      OFFLINE_DATA_DIR / "Changeos_Building_4level.json",
    "changemamba_3level":   OFFLINE_DATA_DIR / "ChangeMamba_Building_3level.json",
    "changemamba_road":     OFFLINE_DATA_DIR / "ChangeMamba_Road.json",
}


# ==================== CreateSTPGTool ====================
# 该工具无构造参数，run() 时直接传入灾前/灾后图像路径
CREATE_STPG_CONFIG = {}


# ==================== SpatialPerceptionTool ====================
SPATIAL_PERCEPTION_CONFIG = {
    "offline_json_path":    OFFLINE_DATA_PATHS["sam3_segment"],
}


# ==================== TemporalChangePerceptionTool ====================
TEMPORAL_CHANGE_CONFIG = {
    "min_iou":  0.3,                    # 演化边匹配的最小 IoU 阈值
    "top_k":    1,                      # 每个区域最多保留的匹配数
}

# ==================== BuildingAssessTool ====================
BUILDING_ASSESS_CONFIG = {
    "offline_json_4level":  OFFLINE_DATA_PATHS["changeos_4level"],
    "offline_json_3level":  OFFLINE_DATA_PATHS["changemamba_3level"],

    # ---------- 标签映射 (0 不算有效分类，标记为 unknown) ----------
    "dam4_labels": {1: "No-damage", 2: "Minor-damage", 3: "Major-damage", 4: "Destroyed"},
    "dam3_labels": {1: "Intact", 2: "Damaged", 3: "Destroyed"},
}


# ==================== RoadDamageAnalysisTool ====================
ROAD_DAMAGE_CONFIG = {
    "offline_json_path":    OFFLINE_DATA_PATHS["changemamba_road"],

    # ---------- 标签映射（0 未使用；1 完好 / 2 洪水淹没 / 3 碎片覆盖）----------
    "damage_labels": {1: "Intact", 2: "Flood-submerged", 3: "Debris-covered"},
}


# ==================== DisasterTypeRecognitionTool ====================
DISASTER_TYPE_CONFIG = {
    "offline_json_path":    OFFLINE_DATA_PATHS["dino_dtr"],
}



# ==================== 汇总：工具名 → 构造参数字典 ====================
TOOL_CONFIGS = {
    "create_stpg":                  CREATE_STPG_CONFIG,
    "spatial_perception":           SPATIAL_PERCEPTION_CONFIG,
    "temporal_change_perception":   TEMPORAL_CHANGE_CONFIG,
    "building_assess":              BUILDING_ASSESS_CONFIG,
    "road_damage_analysis":         ROAD_DAMAGE_CONFIG,
    "disaster_type_recognition":    DISASTER_TYPE_CONFIG,
}
