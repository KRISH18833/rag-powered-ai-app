import sys
import argparse
from src.retrieval.retriever import retrieve
from src.llm.llm_client import generate_with_retry, rag_answer, plain_answer
from src.embeddings.embedder import embed_query

__all__ = [
    "retrieve",
    "generate_with_retry",
    "rag_answer",
    "plain_answer",
    "embed_query",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--mode", choices=["rag", "plain", "compare", "retrieve"], default="rag", help="Execution mode")
    parser.add_argument("--full", action="store_true", help="Print full chunk text instead of first 300 chars (retrieve mode)")
    args = parser.parse_args()

    if args.mode == "retrieve":
        print(f"Query: {args.question}\n")
        results = retrieve(args.question)
        for rank, chunk in enumerate(results, start=1):
            text = chunk["text"] if args.full else chunk["text"][:300].replace("\n", " ")
            print(f"[{rank}] {chunk['source']} | page {chunk['page']} | distance {chunk['distance']:.4f}")
            print(f"    {text}" + ("" if args.full else "..."))
            print()

    elif args.mode == "plain":
        answer = plain_answer(args.question)
        print(answer)

    elif args.mode == "rag":
        answer, sources = rag_answer(args.question)
        print(answer)
        if sources and "sources:" not in answer.lower():
            print("\nSources:")
            for s in sources:
                print(f"- {s}")

    elif args.mode == "compare":
        plain = plain_answer(args.question)
        rag, sources = rag_answer(args.question)

        sources_text = "\n".join(f"- {s}" for s in sources)
        print("=== Plain Answer ===")
        print(plain)
        print("\n=== RAG Answer ===")
        print(rag)
        if sources and "sources:" not in rag.lower():
            print("\nSources:")
            print(sources_text)

        markdown_entry = f"## {args.question}\n\n### Plain Answer\n{plain}\n\n### RAG Answer\n{rag}\n\n"
        if sources:
            markdown_entry += f"**Sources:**\n{sources_text}\n\n"
        markdown_entry += "---\n\n"
        with open("comparison_results.md", "a", encoding="utf-8") as f:
            f.write(markdown_entry)
