
from typing_extensions import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage
from langchain_core.messages.tool import ToolCall
from langgraph.graph.message import add_messages
import operator
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SupervisorState(TypedDict):
    """
    This class represents the state of the research supervisor
    """
    supervisor_messages : Annotated[Sequence[BaseMessage], add_messages]
    research_brief : str
    notes : Annotated[list[str], operator.add] = []
    research_iterations : int = 0
    raw_notes : Annotated[list[str], operator.add] = []

class SupervisorOutput(BaseModel):
    """
    Structured output model, to force LLM to provide message and tools calls
    """
    message : str = Field(
        content="supervisor message"
    )
    tool_calls : List[ToolCall] = Field(
        default_factory=list,
        description="Structured instructions for which tools to call next"
    )

@tool
class ConductResearch(BaseModel):
    """
    Tool for delegating a research task to a specialized sub-agent.

    Args:
        research_topic : The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).
    """

    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
    pass
