
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing from the .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    timeout=30,
    max_retries=2
)

system_prompt = SystemMessage(
    content=(
        "You are a helpful Python and AI learning assistant. "
        "Give accurate, clear, beginner-friendly explanations. "
        "Maintain context from the conversation and ask clarifying "
        "questions when necessary."
    )
)

his = [system_prompt]
MAX_MESSAGES = 10

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    his.append(HumanMessage(content=user_input))

    try:
        result = llm.invoke(his)
        his.append(AIMessage(content=result.content))
        print("AI:", result.content)

    except Exception as e:
        print("AI: Sorry, the request failed.")
        print("Reason:", str(e))

        # Remove the failed user message so history stays consistent
        his.pop()

    # Keep the system prompt and the latest 10 conversation messages
    if len(his) > MAX_MESSAGES + 1:
        his = [system_prompt] + his[-MAX_MESSAGES:]

print("Chat ended.")
