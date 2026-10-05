import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from graph import graph


st.set_page_config(
    page_title="ApexIQ - Enterprise Knowledge Assistant",
    page_icon="💬",
    layout="wide",
)

st.title("ApexIQ - Enterprise Knowledge Assistant")
st.caption("Ask questions about the company knowledge base.")


# -----------------------------
# Session state
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Conversation")

    if st.button("New chat", use_container_width=True):
        clear_chat()
        st.rerun()

    st.caption("Session ID")
    st.code(st.session_state.session_id, language=None)


# -----------------------------
# Display existing messages
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

# -----------------------------
# Chat input
# -----------------------------
query = st.chat_input(
    "Ask a question about the knowledge base..."
)

if query:

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    st.session_state.history.append(
        HumanMessage(content=query)
    )

    recent_history = trim_messages(
        st.session_state.history,
        max_tokens=1000,
        strategy="last",
        token_counter=count_tokens_approximately,
        start_on="human",
        include_system=True,
    )

    with st.chat_message("assistant"):

        with st.spinner("Searching the knowledge base..."):

            try:

                result = graph.invoke({
                    "query": query,
                    "history": recent_history,
                    "session_id": st.session_state.session_id
                })

                # Display answer
                st.markdown(result["answer"])

                # Display sources separately
                if result["sources"]:
                    with st.expander("📄 Sources"):
                        for source in result["sources"]:
                            st.write(f"• {source}")
                else:
                    st.info("No sources found for this answer.")

                # Save assistant message for UI history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"] ,
                    }
                )

                # Save only the answer in conversational memory
                st.session_state.history.append(
                    AIMessage(content=result["answer"])
                )

            except Exception as exc:
                st.error(
                    f"Something went wrong: "
                    f"{type(exc).__name__}: {exc}"
                )