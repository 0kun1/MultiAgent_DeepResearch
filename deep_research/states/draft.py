import operator

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

from typing_extensions import Optional,Annotated

from pydantic import BaseModel,Field

class AgentInputState(MessagesState):
    pass

class AgentState(MessagesState):
    research_brief: Optional[str]
    supervisor_messages: Annotated[list[str],add_messages]
    raw_notes: Annotated[list[str], operator.add] = []
    notes: Annotated[list[str], operator.add] = []
    draft_report: str
    final_report: str

class ResearchQuestion(BaseModel):
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class DraftReport(BaseModel):
    draft_report: str = Field(
        description="A draft report that will be used to guide the research.",
    )