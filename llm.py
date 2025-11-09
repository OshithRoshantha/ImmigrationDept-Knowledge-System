from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from retriever import SemanticSearch
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
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

**Important:** The user's question has been translated to English for processing. Please provide your response in English. It will be translated back to the user's language automatically.

**Previous Conversation:**
{chat_history}

**Context to Use for Your Answer:**
{context}

**User's Question (translated to English):**
{query}

Now, please craft your response in a helpful and natural tone in English:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "query"],
    template=prompt_template
)

def detect_language(text):
    try:
        if any('\u0D80' <= char <= '\u0DFF' for char in text):
            return 'si'
        if any('\u0B80' <= char <= '\u0BFF' for char in text):
            return 'ta'
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'en'

def translate_text(text, target_lang='en', source_lang='auto'):
    try:
        if source_lang == target_lang:
            return text
        lang_map = {'si': 'si', 'ta': 'ta', 'en': 'en'}
        src = lang_map.get(source_lang, 'auto')
        tgt = lang_map.get(target_lang, 'en')
  
        translated = GoogleTranslator(source=src, target=tgt).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  

def ask_llm(query):
    detected_lang = detect_language(query)

    english_query = query
    if detected_lang in ['si', 'ta']:
        english_query = translate_text(query, target_lang='en', source_lang=detected_lang)

    results = SemanticSearch(english_query)
    
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
        {"context": context, "query": english_query},
        config={"configurable": {"session_id": "default"}}
    )

    english_answer = answer.content if hasattr(answer, 'content') else str(answer)
    
    if detected_lang in ['si', 'ta']:
        final_answer = translate_text(english_answer, target_lang=detected_lang, source_lang='en')
        return final_answer
    
    return english_answer

query = "ඔයා කව්ද"
response = ask_llm(query)
print(response)