"""Run the eval set end-to-end and report metrics.

Two metrics:
1. RETRIEVAL: did at least one retrieved chunk come from an expected section?
   This is a coarse proxy for precision@k that you can compute without
   hand-labeling every chunk. Good enough for a weekend.
2. ANSWER QUALITY: LLM-as-judge grades correctness / groundedness / refusal
   behavior on a 1-5 scale.

Output: evaluation/results.json and a printed summary.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from configs.config import JUDGE_MODEL, TOP_K
from retrieval.semantic import retrieve
from generation.llm import generate_answer

client = OpenAI()

TEST_SET = Path(__file__).parent / "test_set.json"
RESULTS = Path(__file__).parent / "results.json"


JUDGE_PROMPT = """You are grading an answer produced by a RAG system against a 10-K filing.

Question: {question}
Expected answer characteristics: {expected}
Question type: {qtype}

System answer:
---
{answer}
---

Grade on two axes, each 1-5:

CORRECTNESS: Does the answer match what's expected?
- For factual/numeric: does it give the right kind of figure/fact?
- For analytical: does it cover the expected themes?
- For should_refuse: does the system correctly say info is not in the filing?
  (If qtype is should_refuse and the system DID refuse, correctness = 5.
   If it hallucinated an answer, correctness = 1.)

GROUNDEDNESS: Does the answer cite chunk_ids in brackets, and are the claims
tied to those excerpts rather than invented?

Respond in strict JSON:
{{"correctness": <1-5>, "groundedness": <1-5>, "notes": "<one sentence>"}}
"""


def judge(question: str, expected: str, qtype: str, answer: str) -> Dict:
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, qtype=qtype, answer=answer,
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def retrieval_section_hit(hits: List[Dict], expected_sections: List[str]) -> bool:
    """True if any retrieved chunk is from an expected section.
    For should_refuse questions (empty expected_sections), this metric is N/A."""
    if not expected_sections:
        return None
    retrieved_sections = {h["section_id"] for h in hits}
    return bool(retrieved_sections & set(expected_sections))


def main():
    questions = json.loads(TEST_SET.read_text())
    results = []

    for q in questions:
        print(f"\n--- {q['id']}: {q['question']}")
        hits = retrieve(q["question"], k=TOP_K)
        answer = generate_answer(q["question"], hits)
        grade = judge(q["question"], q["expected"], q["type"], answer)
        ret_hit = retrieval_section_hit(hits, q["expected_sections"])

        print(f"  retrieved sections: {[h['section_id'] for h in hits]}")
        print(f"  section hit: {ret_hit}")
        print(f"  correctness: {grade['correctness']}/5  groundedness: {grade['groundedness']}/5")
        print(f"  notes: {grade['notes']}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "retrieved_sections": [h["section_id"] for h in hits],
            "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
            "section_hit": ret_hit,
            "answer": answer,
            "correctness": grade["correctness"],
            "groundedness": grade["groundedness"],
            "judge_notes": grade["notes"],
        })

    # Aggregate
    rated = [r for r in results if r["section_hit"] is not None]
    section_hit_rate = sum(r["section_hit"] for r in rated) / len(rated) if rated else 0
    avg_correct = sum(r["correctness"] for r in results) / len(results)
    avg_ground = sum(r["groundedness"] for r in results) / len(results)

    summary = {
        "n_questions": len(results),
        "retrieval_section_hit_rate": round(section_hit_rate, 3),
        "avg_correctness": round(avg_correct, 2),
        "avg_groundedness": round(avg_ground, 2),
    }

    RESULTS.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull results -> {RESULTS}")


if __name__ == "__main__":
    main()
