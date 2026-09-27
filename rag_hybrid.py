from hybrid_retriever import search
from reranker import rerank
from context_builder import build_context
from generator import generate_answer
from query_rewriter import rewrite_question

def ask(question, history):
    standalone_question = rewrite_question(
        question,
        history
    )

    print(
        f"\nSearch question: {standalone_question}"
    )

    results = search(
        question,
        top_k=5,
        alpha=0.5,
    )

    print(
        f"\nRetrieved {len(results.matches)} documents"
    )

    reranked_results = rerank(
        question,
        results,
        top_k=3,
    )

    print(
        f"Reranked to {len(reranked_results)} documents"
    )

    context = build_context(
        reranked_results
    )

    answer = generate_answer(
        question,
        context,
    )

    return answer

if __name__ == "__main__":

    print("=" * 60)
    print("Hybrid RAG Assistant")
    print("=" * 60)
    print("Type 'exit' to quit.")

    history = []

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        answer = ask(
            question,
            history
        )

        print(
            f"\nAssistant: {answer}"
        )

        history.append(
            f"User: {question}"
        )

        history.append(
            f"Assistant: {answer}"
        )
