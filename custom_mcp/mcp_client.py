import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode




async def main():
    model = ChatOllama(model="phi4-mini:3.8b")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["/home/sudamasharma/agentic_ai/mcp/custom_mcp/custom_mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    model_with_tools  = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
        

    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
    

    graph = StateGraph(MessagesState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", should_continue)
    graph.add_edge("tools", "call_model")

    workflow = graph.compile()

    result = await workflow.ainvoke({"messages": "what's (3 + 5) x 12?"})
    print(result['messages'][-1].content)


if __name__=="__main__":
    asyncio.run(main())