from utils.audio_processor import process_input
from core.transcriber import transcribe_auto
from core.summarise import summarise, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.engine import build_rag_chain, load_rag_chain, ask_question

def run_pipeline(source: str) -> dict:
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

    rag_chain = build_rag_chain(transcript)

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
        # skip re-processing; load the vector store that was saved during the last run
        print("Loading previous session...")
        rag_chain = load_rag_chain()
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