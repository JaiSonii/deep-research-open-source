from typing import Any
from datetime import datetime

def get_today_str():
    """return todays date in windows, different method for other os"""
    return datetime.now().strftime("%Y -%m -%d")

def format_tool_instructions(tools: list[Any]) -> str:
    tool_str = ""
    for index, tool in enumerate(tools):
        tool_str += "\n<tool_info>\n"
        tool_str += f"<tool_id> {index} <tool_id>\n"
        tool_str += f"<tool_name> {tool.name} <tool_name>\n"   # Use .name
        tool_str += f"<tool_description>{tool.description}<tool_description>\n"
        tool_str += "<tool_info>\n"
    return tool_str