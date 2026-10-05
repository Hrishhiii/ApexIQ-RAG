"""Command-line entry point for the enterprise RAG application."""



from rag import rag_pipeline
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
import uuid

session_id = str(uuid.uuid4())

history = []

while True:

    query = input("Enter your query: ")

    if query.lower() in ["exit","Bye","quit","goodbye"]:
        print("Goodbye. Have a great day!")
        break

    history.append(
        HumanMessage(content=query)
    )

    # Keep only the recent conversation within token budget
    recent_history = trim_messages(
        history,
        max_tokens=1000,
        strategy="last",
        token_counter=count_tokens_approximately,
        start_on="human",
        include_system=True
    )

    answer = rag_pipeline(query, recent_history, session_id=session_id)
    print("Answer:", answer)

    history.append(
        AIMessage(content=answer)
    )
