from typing_extensions import Literal

from langchain_core.messages import HumanMessage, get_buffer_string
from langgraph.graph import StateGraph, START,END
from langgraph.types import Command

from deep_research import logging as dr_logging
from deep_research.llm import get_chat_model
from deep_research.prompts import RESEARCH_BRIEF_PROMPT, DRAFT_REPORT_PROMPT  
from deep_research.states import AgentState, ResearchQuestion, AgentInputState, DraftReport
from deep_research.utils import get_today_str 

draft_model = get_chat_model("draft")

def write_research_brief(state:AgentState)->Command[Literal["write_draft_report"]]:
    
    prompt = RESEARCH_BRIEF_PROMPT.format(
        messages=get_buffer_string(state.get("messages",[])),
        date=get_today_str()
    )

    structured_output_model = draft_model.with_structured_output(ResearchQuestion)
    response = structured_output_model.invoke([HumanMessage(content=prompt)])

    return Command(
        goto="write_draft_report",
        update={"research_brief":response.research_brief}
    )

def write_draft_report(state:AgentState)-> Command[Literal["__end__"]]:

    research_brief = state.get("research_brief","")

    draft_report_prompt = DRAFT_REPORT_PROMPT.format(
        research_brief=research_brief,
        date=get_today_str()
    )

    structured_output_model = draft_model.with_structured_output(DraftReport)
    response = structured_output_model.invoke([
        HumanMessage(content=draft_report_prompt)
    ])

    return {
        "research_brief":research_brief,
        "draft_report":response.draft_report,
        "supervisor_messages":[
            "Here is the draft report: "+ response.draft_report,
            research_brief
        ]
    }