from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

router_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

class RouteQuery(BaseModel):
    route: Literal["greeting","rag","out_of_scope"] = Field(
        description="The appropriate route for the query. Options are: 'greeting', 'rag', or 'out_of_scope'."
    )


structured_router = router_llm.with_structured_output(RouteQuery,method="json_schema",strict=True)


router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are the query router for an enterprise knowledge assistant.

Your job is ONLY to classify the user's query.
Do not answer the user's question.

Choose exactly one route:

1. greeting
   Use this when the user is only greeting, thanking,
   acknowledging, or saying goodbye.

   Examples:
   - Hi
   - Hello
   - Hey
   - Good morning
   - Thanks
   - Thank you
   - Bye

2. rag
   Use this when the user is asking a question that could
   reasonably be answered using the enterprise knowledge base.

   Examples:
   - What is the education stipend?
   - What is the leave policy?
   - How do I submit an expense?
   - Are employees eligible for certification reimbursement?

3. out_of_scope
   Use this when the query is unrelated to the enterprise
   knowledge base and cannot reasonably be answered from
   the enterprise documents.

   Examples:
   - Who is the PM of India?
   - What is the weather today?
   - Write a Python program.
   - Who won yesterday's cricket match?

Important:
- Classify based on the user's PRIMARY intent.
- A greeting combined with an enterprise question is RAG.
  Example: "Hi, can you tell me the leave policy?" → rag
- Do not use outside/general knowledge to determine an answer.
- Do not provide an answer. Return only the classification.
        """
    ),
    (
        "human",
        "{query}"
    )
])

#Chaining the prompt and structured output to create a router chain
router_chain = router_prompt | structured_router


def classify_query(query: str) -> str:
    result = router_chain.invoke({"query": query})
    return result.route
