import asyncio
from langchain_core.messages import HumanMessage
from deep_research.agent_builder import deep_researcher_builder


async def main():
    agent = deep_researcher_builder.compile()

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="简单介绍一下 AI Agent Memory。")
            ]
        },
        config={
            "recursion_limit": 50
        }
    )

    print(result.keys())
    print(result.get("final_report", "没有 final_report"))


if __name__ == "__main__":
    asyncio.run(main())