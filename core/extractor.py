from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_llm():
    return init_chat_model("groq:qwen3.6-27b")

def build_chain(system_prompt: str):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return prompt | llm | StrOutputParser()

def extract_action_items(transcript:str)->str:
    chain = build_chain(
        "You are an expert content analyst. Extract every action item, recommendation, or to-do "
        "mentioned in the transcript — this could be from a meeting, tutorial, lecture, or any video. "
        "For each item provide: Task, Owner/Audience (who it applies to, or 'General'), "
        "and Deadline (if stated, else 'Not specified'). "
        "Format as a numbered list. If none found, say 'No action items found.'"
    )

    return chain.invoke({"text": transcript})

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Extract every key point, conclusion, or decision "
        "stated in the transcript — this could be from a meeting, documentary, talk, or any video. "
        "Only include definitive statements, not speculations or open questions. "
        "For each point state what was concluded and by whom (if mentioned). "
        "Format as a numbered list. If none found, say 'No key decisions or conclusions found.'"
    )
    return chain.invoke({"text": transcript})

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Extract all questions, unresolved topics, or areas "
        "flagged for further exploration in the transcript — from any type of video content. "
        "Only include items left unanswered or explicitly deferred. "
        "For each item state the question/topic and who raised it (if mentioned). "
        "Format as a numbered list. If none found, say 'No open questions found.'"
    )
    return chain.invoke({"text": transcript})