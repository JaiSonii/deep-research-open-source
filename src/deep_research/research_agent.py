
from deep_research.openrouter import init_chat_model
from langchain_core.messages import SystemMessage
from deep_research.prompts import research_agent_prompt, compress_research_human_message, compress_research_system_prompt
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage, filter_messages
from deep_research.research_state import ResearchState, ResearchOutput, LLMOutput, Summary
from deep_research.tavily import tavily_search
from typing_extensions import Literal, Any
from langgraph.graph import StateGraph, START, END
from os import getenv
from datetime import datetime

model = init_chat_model(model='x-ai/grok-4-fast:free', temperature=0.3, api_key=getenv('OPENROUTER_API_KEY'))
compress_model = init_chat_model(model='x-ai/grok-4-fast:free', temperature=0.3, api_key=getenv('OPENROUTER_API_KEY'))
tools = [tavily_search]
tools_by_name = {tool.name : tool for tool in tools}

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

def llm_call(state : ResearchState):
    """
    Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    structured_model = model.with_structured_output(LLMOutput)

    messages = [SystemMessage(content=research_agent_prompt.format(date=get_today_str(), tools_info=format_tool_instructions(tools)))] \
               + state['researcher_messages']

    result = structured_model.invoke(messages)
    print('-----------------------------------LLM CALL-------------------------------------')
    print(state)

    ai_message = AIMessage(
        content=result.research_message or "",
        tool_calls=result.tool_calls
    )
    print('------------------------------------AI message----------------------------------')
    print(ai_message.tool_calls)

    return {"researcher_messages": [ai_message]}


def tool_node(state : ResearchState):
    print('-----------------------------------Tool Node-------------------------------------')
    print(state)
    tool_calls = state['researcher_messages'][-1].tool_calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call['name']]
        observations.append(tool.invoke(tool_call['args']))

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call['name'],
            tool_call_id=tool_call['id']
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {'researcher_messages' : tool_outputs}

def should_continue(state : ResearchState) -> Literal['llm_call','compress_research']:
    print('-----------------------------------Should continue-------------------------------------')
    print(state)
    last_message = state['researcher_messages'][-1]
    if last_message.tool_calls:
        return 'tool_node'
    return 'compress_research'

def compress_research(state: ResearchState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """
    print('-----------------------------------compress research-------------------------------------')
    print(state)
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }


graph_builder = StateGraph(ResearchState, output_schema=ResearchOutput)

graph_builder.add_node('llm_call', llm_call)
graph_builder.add_node('tool_node', tool_node)
graph_builder.add_node('compress_research', compress_research)

graph_builder.add_edge(START, "llm_call")
graph_builder.add_edge("tool_node", "llm_call")  # back to LLM after tool

graph_builder.add_conditional_edges(
    'llm_call',
    should_continue,
    {
        "tool_node" : "tool_node",
        "compress_research" : "compress_research"
    }
)
graph_builder.add_edge("compress_research", END)

research_agent = graph_builder.compile()
