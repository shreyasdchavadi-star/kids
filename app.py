# ==========================================================
# Google Colab - Single Cell LangGraph + Groq + Gradio
# Assignment:
# 1. Process Node
# 2. PrintResponse Node
# 3. Display Response
# 4. Store Chat History
# 5. Gradio ChatInterface
# 6. Enhanced Prompt (AI Expert for Kids 5-15)
# ==========================================================

# Install Libraries


import gradio as gr
from typing import TypedDict
from google.colab import userdata
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# ==========================================================
# Groq API Key
# ==========================================================

GROQ_API_KEY = userdata.get("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

# ==========================================================
# State Definition
# ==========================================================

class ChatState(TypedDict):
    question: str
    response: str
    history: list

# ==========================================================
# Process Node
# ==========================================================

def process_node(state: ChatState):

    question = state["question"]

    prompt = f"""
You are an Expert Artificial Intelligence Teacher with over 20 years of experience.

Your mission is to teach Artificial Intelligence to children between 5 and 15 years old.

Instructions:

• Detect the student's approximate age from the question.
• If no age is mentioned, assume the student is 10 years old.

Age Groups

Age 5-7
- Use very simple words.
- Use short sentences.
- Explain like a bedtime story.
- Use emojis.
- Use toys, animals and cartoons as examples.

Age 8-10
- Use easy English.
- Give daily life examples.
- Explain step by step.
- Make learning fun.

Age 11-13
- Introduce AI terms slowly.
- Explain every technical word.
- Use examples like YouTube, ChatGPT, Alexa, Google Maps, Robots and Games.

Age 14-15
- Explain like an introductory AI course.
- Cover Machine Learning, Deep Learning, Neural Networks and LLMs using beginner-friendly language.

General Rules

1. Always answer in simple English.
2. Never use difficult words without explaining them.
3. Use bullet points whenever possible.
4. Give at least one real-life example.
5. Make learning interesting and interactive.
6. Encourage curiosity.
7. Use emojis where appropriate.
8. End every answer with:

🎯 Fun Fact:
❓ Quiz Question:

Student Question:
{question}
"""

    result = llm.invoke(prompt)

    state["response"] = result.content

    return state

# ==========================================================
# PrintResponse Node
# ==========================================================

def print_response_node(state: ChatState):

    print("\n" + "="*70)
    print("USER QUESTION")
    print(state["question"])
    print("\nAI RESPONSE")
    print(state["response"])
    print("="*70)

    history = state.get("history", [])

    history.append({
        "user": state["question"],
        "assistant": state["response"]
    })

    state["history"] = history

    return state

# ==========================================================
# Build LangGraph Workflow
# ==========================================================

graph = StateGraph(ChatState)

graph.add_node("Process", process_node)
graph.add_node("PrintResponse", print_response_node)

graph.set_entry_point("Process")

graph.add_edge("Process", "PrintResponse")
graph.add_edge("PrintResponse", END)

app = graph.compile()

# ==========================================================
# Chat History
# ==========================================================

chat_history = []

# ==========================================================
# Chat Function
# ==========================================================

def chatbot(message, history):

    global chat_history

    state = {
        "question": message,
        "response": "",
        "history": chat_history
    }

    result = app.invoke(state)

    chat_history = result["history"]

    return result["response"]

# ==========================================================
# Gradio UI
# ==========================================================

demo = gr.ChatInterface(
    fn=chatbot,
    title="🤖 AI Teacher for Kids (Age 5–15)",
    description="""
This chatbot uses LangGraph with:
✅ Process Node
✅ PrintResponse Node
✅ Chat History
✅ Expert AI Teacher Prompt
""",
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="Ask anything about AI...",
        container=False,
        scale=7
    ),
    theme=gr.themes.Soft(),
    type="messages"
)

# ==========================================================
# Launch Application
# ==========================================================

demo.launch(share=True)
