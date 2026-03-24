from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests

load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-120b")

@tool
def get_stock_price(ticker: str) -> dict:
    """Get the current stock price for a given ticker symbol."""
    
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey=X0K7FQOLBBQ9820R"
    )

    r = requests.get(url)
    data = r.json()
    return data


@tool
def purchase_stock(ticker: str, quantity: int) -> dict:
    """Purchase a given quantity of stock for a given ticker symbol.
    
    Human-In-the-loop (HITL) function that simulates purchasing stock. 
    Before executing the purchase, it would ask for human confirmation. 
    ("yes" to confirm, "no" to cancel).

    """

    decision = interrupt(f"Approve purchase of {quantity} shares of {ticker}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        # Simulate stock purchase logic here
        return {"status": "success", 
                "message": f"Purchased {quantity} shares of {ticker}",
                "ticker": ticker,
                "quantity": quantity
        }
    
    else:
        return {"status": "cancelled", 
                "message": f"Purchase of {quantity} shares of {ticker} cancelled",
                "ticker": ticker,
                "quantity": quantity
        }
    

tools = [get_stock_price, purchase_stock]

llm_with_tool = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """Node that handles the chat interaction with the user."""
    message = state["messages"]

    result = llm_with_tool.invoke(message)

    return {"messages": [result]}

tool_node = ToolNode(tools)

memory = InMemorySaver()


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=memory)

if __name__ == "__main__":

    thread_id = "demo_thread"

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break

        state = {"messages": [HumanMessage(content=user_input)]}

        result = chatbot.invoke(state, config={"configurable": {"thread_id": thread_id}})


        interrupts = result.get("__interrupt__", [])

        if interrupts:
            prompt_to_user = interrupts[0].value
            print(f"HITL: {prompt_to_user}")
            decision = input("Your decision: ").strip().lower()


            result = chatbot.invoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})


        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")


        