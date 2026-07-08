#***********************************************
#      Filename: state_research.py
#   Description: 研究智能体结构化字段定义
#***********************************************

from langchain_core.messages import BaseMessage
import operator
from langgraph.graph.message import add_messages
from typing_extensions import Annotated,List,TypedDict,Sequence
from pydantic import BaseModel,Field

class ResearcherState(TypedDict):
    researcher_messages:Annotated[Sequence[BaseMessage],add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str],operator.add]

class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

class Summary(BaseModel):
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")