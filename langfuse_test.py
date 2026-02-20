import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.tools import AgentTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from langfuse_tracer import trace

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

math_expert = AssistantAgent(
    name="math_expert",
    model_client=model_client,
    system_message="You are a math expert. Solve problems step by step.",
    description="Expert in mathematical problems.",
)

orchestrator = AssistantAgent(
    name="orchestrator",
    model_client=model_client,
    system_message="You are a general assistant. Delegate to the math expert when needed.",
    tools=[AgentTool(math_expert, return_value_as_last_message=True)],
    max_tool_iterations=5,
)

@trace(
    name="orchestrator_with_tools",
    session_id="session-003",
    tags=["autogen", "agent-tool", "orchestration"],
)
async def run_orchestrator(task: str):
    result = await orchestrator.run(task=task)
    return result

async def main():
    result = await run_orchestrator("What is the integral of x^3 + 2x from 0 to 5?")
    print(result.messages[-1].content)
    await model_client.close()

asyncio.run(main())