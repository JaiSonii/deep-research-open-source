
from langgraph.graph import MessagesState
from typing_extensions import Optional, Annotated, Sequence
import operator
from langgraph.graph.message import BaseMessage,add_messages
from pydantic import BaseModel, Field

class AgentInputState(MessagesState):
    """Input State for the Agent. Only contains the user's input message"""
    pass

class AgentState(MessagesState):
    """State of the Agent while Agent researches on the topic"""
    research_brief : Optional[str]
    supervisor_messages : Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add] = []
    notes: Annotated[list[str], operator.add] = []
    final_report : str

class ClarifyWithUser(BaseModel):
    """Clarification Schema for clarifying before starting the researching"""
    need_clarification : bool = Field(
        description="Wether the user needs to be asked for clarifying question"
    )

    question : str = Field(
        description= "A question to ask the user to clarify the research scope"
    )

    verification : str = Field(
        description="Verify message that we will start research, after the user has provided the necessary information"
    )

class ResearchQuestion(BaseModel):
    """The Actual Search question to be provided for research"""
    research_brief : str = Field(
        description="A research question that will be used to guide the research"
    )

