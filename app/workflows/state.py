from typing import Annotated
import os
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from IPython.display import Image, display
import matplotlib.pyplot as plt


openai_api_key = os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatOpenAI(openai_api_key="sk-proj-kf9yknjQHXCyqASw4jfMT3BlbkFJd3zhvmKFUC414OEaHyZM", model="gpt-4o-mini", temperature=0)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()


# Draw the graph
try:
    img = graph.get_graph().draw_mermaid_png()
    
    # Save the image to a file
    with open('graph_image.png', 'wb') as f:
        f.write(img)

    # Display the image using matplotlib
    img_display = plt.imread('graph_image.png')
    plt.imshow(img_display)
    plt.axis('off')  # Hide axes
    plt.show()
    
except Exception as e:
    print(f"An error occurred: {e}")