import hashlib
import json
import os
from utils.audio_processor import process_input
from core.transcriber import transcribe_auto
from core.summarise import summarise, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.engine import build_rag_chain, load_rag_chain, ask_question

_SESSIONS_FILE = "chroma-db/sessions.json"

def _get_session_id(source: str) -> str:
    # deterministic 8-char hash so the same source always maps to the same store
    return hashlib.md5(source.encode()).hexdigest()[:8]

def _save_session(session_id: str, source: str, title: str):
    os.makedirs("chroma-db", exist_ok=True)
    sessions = _load_sessions()
    sessions[session_id] = {"source": source, "title": title}
    with open(_SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

def _load_sessions() -> dict:
    if not os.path.exists(_SESSIONS_FILE):
        return {}
    with open(_SESSIONS_FILE) as f:
        return json.load(f)

def run_pipeline(source: str) -> dict:
    session_id = _get_session_id(source)
    print("Starting video assistant")

    chunks = process_input(source)

    transcript = transcribe_auto(chunks)
    print("Raw transcription (first 300 chars): ", transcript[:3000])

    summary = summarise(transcript)
    title = generate_title(summary)
    print("Title: ", title)
    print("Summary: ", summary)

    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript, session_id)
    _save_session(session_id, source, title)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

def chat_loop(rag_chain):
    print("\n💬 Chat with your video (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")

if __name__ == "__main__":
    print("1. Process a new video")
    print("2. Resume chat with last processed video")
    mode = input("Choose (1/2): ").strip()

    if mode == "2":
        sessions = _load_sessions()
        if not sessions:
            print("No previous sessions found. Process a video first.")
        else:
            print("\nAvailable sessions:")
            for sid, info in sessions.items():
                print(f"  [{sid}] {info['title']} — {info['source']}")
            sid = input("Enter session ID to resume: ").strip()
            if sid not in sessions:
                print("Invalid session ID.")
            else:
                rag_chain = load_rag_chain(sid)
                chat_loop(rag_chain)
    else:
        source = input("Enter YouTube URL or local file path: ").strip()
        result = run_pipeline(source)

        print("\n" + "=" * 60)
        print(f"📌 Title: {result['title']}")
        print(f"\n📋 Summary:\n{result['summary']}")
        print(f"\n✅ Action Items:\n{result['action_items']}")
        print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
        print(f"\n❓ Open Questions:\n{result['open_questions']}")
        print("=" * 60)

        chat_loop(result["rag_chain"])    