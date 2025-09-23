
"""
Contains Nodes and the graph flow to clarify with user and finally have determine the proper research scope
"""

from deep_research.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState
from typing import Literal
from langgraph.types import Command
import os
from deep_research.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from langchain_core.messages import HumanMessage, get_buffer_string, AIMessage
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from deep_research.openrouter import init_chat_model

load_dotenv()

def get_today_str():
    """return todays date in windows, different method for other os"""
    return datetime.now().strftime("%Y -%m -%d")

model = init_chat_model(model = "x-ai/grok-4-fast:free", api_key=os.getenv('OPENAI_API_KEY'), temperature=0)

def clarify_with_user(state : AgentState)-> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """

    structured_model = model.with_structured_output(ClarifyWithUser)
    response = structured_model.invoke([
        HumanMessage(content = clarify_with_user_instructions.format(
            messages = get_buffer_string(messages=state['messages']),
            date = get_today_str()
        ))
    ])

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages" : AIMessage(content=response.question)}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages" : AIMessage(content=response.verification)}
        )

def write_research_brief(state : AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    structured_model = model.with_structured_output(ResearchQuestion)

    response = structured_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(messages=state.get('messages',[])),
            date=get_today_str()
        ))
    ])

    return {
        "research_brief" : response.research_brief,
        "supervisor_brief" : response.research_brief
    }


deep_research_builder = StateGraph(AgentState, input_schema = AgentInputState)

deep_research_builder.add_node('clarify_with_user', clarify_with_user)
deep_research_builder.add_node('write_research_brief', write_research_brief)

deep_research_builder.add_edge(START, 'clarify_with_user')
deep_research_builder.add_edge('write_research_brief', END)

scope_research = deep_research_builder.compile()
