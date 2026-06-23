from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END, 
    ACT: ACT}
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

def main():
    print("Hello ReAct LangGraph with Function Calling\n")

    # Test 1: Math tool (triple)
    # print("Test 1: Triple of 7")
    # res = app.invoke({"messages": [HumanMessage(content="What is the triple of 7?")]})
    #print(res["messages"][LAST].content)
    #print()

    # Test 2: Web search tool (TavilySearch)
    #print("Test 2: Weather in London")
    #res = app.invoke({"messages": [HumanMessage(content="What is the weather today in London?")]})
    #print(res["messages"][LAST].content)
    #print()

    print("Test 3: Weather in London and triple it")
    res = app.invoke({"messages": [HumanMessage(content="What is the celsius temperature today in London? List it and triple it")]})
    print(res["messages"][LAST].content)
    print()


if __name__ == "__main__":
    main()
