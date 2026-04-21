"""CLI: python -m app.cli "your question here" """
import sys

from retrieval.semantic import retrieve
from generation.llm import generate_answer


def answer(query: str, k: int = 5):
    hits = retrieve(query, k=k)
    ans = generate_answer(query, hits)
    return ans, hits


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m app.cli "your question"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    ans, hits = answer(q)
    print("\n=== ANSWER ===\n")
    print(ans)
    print("\n=== SOURCES ===\n")
    for h in hits:
        print(f"  [{h['score']:.3f}] {h['chunk_id']} — {h['section_name']}")


if __name__ == "__main__":
    main()
