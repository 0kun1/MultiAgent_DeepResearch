#***********************************************
#      Filename: state_supervisor.py
#   Description: Supervisor智能体的结构化字段定义
#***********************************************

"""
多智能体Supervisor的State定义
本文件定义了多智能体Supervisor工作流程中使用的State对象和tools字段定义。
"""

import operator

from typing_extensions import Annotated,List,TypedDict,Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from pydantic import BaseModel,Field

from deep_research.states.critique import Critique 
from deep_research.states.quality import QualityMetric

class SupervisorState(TypedDict):
    supervisor_messages: Annotated[Sequence[BaseMessage],add_messages]
    research_brief: str
    notes: Annotated[List[str],operator.add]
    research_iterations: int = 0
    critique_nums: int = 0
    raw_notes: Annotated[list[str], operator.add] = []
    draft_report: str
    active_critiques: Annotated[List[Critique], operator.add]
    quality_history: Annotated[List[QualityMetric], operator.add]
    needs_quality_repair: bool

@tool
class ConductResearch(BaseModel):
    """用于将研究任务委派给专业子Agent (specialized sub-agent) 的工具。"""
    research_topic: str = Field(
        description="研究主题。每次委派的任务应该为单一主题，并需详细描述（至少一个段落）。",
    )

@tool
class ResearchComplete(BaseModel):
    """用于指示研究过程已完成的工具。"""
    pass

