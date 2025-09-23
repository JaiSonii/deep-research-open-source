
"""
This module includes the Research State and other states requried by the Research Agent
"""

from typing_extensions import TypedDict, Sequence, Annotated, List, Dict, Any, Optional
import uuid
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolCall
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import operator

class ResearchState(TypedDict):
    researcher_messages : Annotated[Sequence[List[BaseMessage]], add_messages]
    tool_call_iterations : int
    research_topic : str
    compressed_research : str
    raw_notes : Annotated[List[str], operator.add]

class ResearchOutput(TypedDict):
    compressed_research : str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages : Annotated[Sequence[List[BaseMessage]], add_messages]

class ToolFunction(BaseModel):
    name: str = Field(..., description="Name of the tool to invoke")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call")


class LLMOutput(BaseModel):
    """
    Schema for LLM responses in the research agent.

    - tool_calls: Full list of tool calls (name, id, args) to execute next.
    - research_message: The reasoning, plan, or final answer message.
    """
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Structured instructions for which tools to call next"
    )
    research_message: Optional[str] = Field(
        default=None,
        description="The reasoning, plan, or partial/final research message."
    )


class Summary(BaseModel):
    """Schema for webpage content summarization."""
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")
