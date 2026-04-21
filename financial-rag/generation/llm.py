"""Answer generation. Grounds answers in retrieved chunks and cites them by ID.

Design choice: the prompt forces the model to say 'not found in the filing' rather
than hallucinate when evidence is thin. This is the most important guardrail for
a finance RAG — wrong numbers are worse than no answer.
"""
from __future__ import annotations
from typing import List, Dict

from openai import OpenAI

from configs.config import GEN_MODEL

client = OpenAI()


SYSTEM = """You are a financial analyst assistant answering questions about a company's 10-K filing.

Rules:
1. Base your answer ONLY on the provided excerpts. Do not use outside knowledge.
2. If the excerpts do not contain the answer, say: "The filing does not contain enough information to answer this."
3. Cite the excerpts you used by their chunk_id in brackets, e.g. [AAPL-item_1a-002].
4. Be concise. Numbers and specifics beat generic language.
5. If excerpts conflict, note it.
"""


def format_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[chunk_id: {c['chunk_id']} | section: {c['section_name']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: List[Dict]) -> str:
    ctx = format_context(chunks)
    user = f"Question: {query}\n\nExcerpts from the filing:\n\n{ctx}"
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content
