
from langchain_core.messages import BaseMessage, filter_messages, SystemMessage, AIMessage, HumanMessage, ToolMessage
from deep_research.state_multi_agent_supervisor import SupervisorState, ConductResearch, ResearchComplete, SupervisorOutput
from deep_research.openrouter import init_chat_model
from langgraph.types import Command
from typing_extensions import Literal
from deep_research.prompts import lead_researcher_prompt
from deep_research.utils import get_today_str, format_tool_instructions
from langgraph.graph import END, START, StateGraph
from deep_research.research_agent import research_agent
from os import getenv
import asyncio

def get_notes_from_tool_calls(messages : list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.

    Args:
        messages: List of messages from supervisor's conversation history

    Returns:
        List of research note strings extracted from ToolMessage objects
    """
    return [msg.content for msg in filter_messages(messages, include_types='tool')]

# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it

tools = [ConductResearch, ResearchComplete]
supervisor_model = init_chat_model(model='x-ai/grok-4-fast:free', temperature=0.3, api_key=getenv('OPENROUTER_API_KEY'))

max_concurrent_researchers = 3
max_researcher_iterations = 6

async def supervisor(state : SupervisorState) -> Command[Literal['supervisor_tools']]:
    """Coordinate research activities.

    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with updated state
    """
    print('---------------------------------------STATE-----------------------------------------------------')
    print(state)
    structured_model = supervisor_model.with_structured_output(SupervisorOutput)
    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        tool_info=format_tool_instructions(tools),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )

    messages = [SystemMessage(content=system_message)] + state.get('supervisor_messages', [])

    result = await structured_model.ainvoke(messages)
    print('-------------------------------------------supervisor_result---------------------------------------------')
    print(result)
    ai_message = AIMessage(content=result.message, tool_calls=result.tool_calls)
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages" : [ai_message],
            "research_iterations" : state.get('research_iterations', 0) + 1
        }
    )

async def supervisor_tools(state : SupervisorState) -> Command[Literal['supervisor', '__end__']]:
    """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process, or handle errors
    """

    supervisor_messages = state.get('supervisor_messages',[])
    research_iterations = state.get('research_iterations',0)
    most_recent_message = supervisor_messages[-1]

    tool_messages = []
    all_raw_notes = []
    next_step = 'supervisor'
    should_end = False

    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call['name'] == 'ResearchComplete'
        for tool_call in most_recent_message.tool_calls
    )
    if exceeded_iterations or no_tool_calls or research_complete:
        next_step = END
        should_end = True
    else:
        try:
            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call['name'] == 'ConductResearch'
            ]

            if conduct_research_calls:
                coros = [
                    research_agent.ainvoke({
                        "researcher_messages" : HumanMessage(content = tool_call['args']['research_topic']),
                        "research_topic" : tool_call['args']['research_topic']
                    })
                    for tool_call in most_recent_message.tool_calls
                ]

                tool_results = await asyncio.gather(*coros)
                print('----------------------------------------------Tool Results-----------------------------------------')
                print(tool_results)

                research_tool_messages = [
                    ToolMessage(
                        content = result.get('compressed_research', "Error Synthesizing research report"),
                        id=tool_call['id'],
                        name=tool_call['name']
                    ) for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                all_raw_notes = [
                    '\n'.join(result.get('raw_notes', []))
                    for result in tool_results
                ]
        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes
            }
        )

supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node('supervisor', supervisor)
supervisor_builder.add_node('supervisor_tools', supervisor_tools)
supervisor_builder.add_edge(START, 'supervisor')
supervisor_agent = supervisor_builder.compile() 
