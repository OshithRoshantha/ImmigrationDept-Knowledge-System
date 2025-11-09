from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from retriever import SemanticSearch
from dotenv import load_dotenv
import os

load_dotenv()

chat_history_store = InMemoryChatMessageHistory()

prompt_template = """
**Role:** You are a dedicated support assistant for the Department of Immigration and Emigration. Your name is "Immigration Assist". Your primary goal is to help people understand complex immigration processes in a simple, friendly, and reassuring way.

**Communication Style:**
- **Tone:** Warm, patient, and conversational. Avoid robotic or overly formal language.
- **Approach:** Be empathetic and acknowledge that immigration matters can be stressful. Use phrases like "That's a great question!" or "I can help you with that."
- **Clarity:** Break down complex information into easy-to-understand steps. Use bullet points if it helps with clarity, but keep the language natural.
- **Limits:** If the context does not contain the answer, do not make up information. Instead, politely say you don't have the specific details on that and guide them to the official website or helpline for the most accurate information.

**Previous Conversation:**
{chat_history}

**Context to Use for Your Answer:**
{context}

**User's Question:**
{query}

Now, please craft your response in a helpful and natural tone:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "query"],
    template=prompt_template
)

def ask_llm(query):
    results = SemanticSearch(query)
    
    context_parts = []
    for r in results:
        clean_summary = r["summary"].replace("<n>", "\n").replace("*", "•")
        clean_text = r["text"].replace("*", "•")
        context_parts.append(
            f"Title: {r['title']}\nSummary: {clean_summary}\nDetails: {clean_text}\nScore: {r['score']}\n"
        )
    context = "\n---\n".join(context_parts)

    llm = ChatOpenAI(
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        openai_api_base=os.environ["LLM_BASE"],
        openai_api_key=os.environ["HF_TOKEN"],
        temperature=0
    )
    
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history_store,
        input_messages_key="query",
        history_messages_key="chat_history"
    )
    
    answer = chain_with_history.invoke(
        {"context": context, "query": query},
        config={"configurable": {"session_id": "default"}}
    )

    return answer.content if hasattr(answer, 'content') else str(answer)

query = "ඔයා කව්ද"
response = ask_llm(query)
print(response)