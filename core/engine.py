from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever
from dotenv import load_dotenv

load_dotenv()

# Shared prompt used by all RAG chains — keeps the instruction set consistent across build and load paths.
# {context} is filled with retrieved transcript chunks; {question} is passed through from the user.
_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert video content assistant. Answer the user's question \
based ONLY on the transcript context provided below.

If the answer is not found in the context, say: \
"I could not find this information in the transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from transcript:
{context}""",
    ),
    ("human", "{question}"),
])

def get_llm():
    # initialises the Groq-hosted Qwen model; called fresh each time to avoid stale state
    return init_chat_model("groq:llama-3.3-70b-versatile")

def _format_docs(docs):
    # flattens a list of retrieved Document objects into a single string for the prompt context
    return "\n\n".join(doc.page_content for doc in docs)

def _build_chain(retriever):
    # LCEL pipeline: retriever fetches top-k chunks → formatted into context → prompt → LLM → plain string
    # RunnablePassthrough() forwards the raw question string unchanged into the {question} slot
    return (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | get_llm()
        | StrOutputParser()
    )

def build_rag_chain(transcript: str):
    # call this on first run: embeds the transcript, saves the vector store to disk, returns the Q&A chain
    return _build_chain(get_retriever(build_vector_store(transcript), k=4))

def load_rag_chain():
    # call this on subsequent runs: skips re-embedding by loading the existing vector store from disk
    return _build_chain(get_retriever(load_vector_store(), k=4))

def ask_question(rag_chain, question: str) -> str:
    # invokes the chain with a plain question string; the retriever + prompt handle context injection internally
    return rag_chain.invoke(question)