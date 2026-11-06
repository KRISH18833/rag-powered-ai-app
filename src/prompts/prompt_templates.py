"""Prompt templates for RAG and LLM querying."""

SYSTEM_PROMPT = """You are a Cloud Computing for ML course assistant.
Rules:
1. Answer ONLY from the provided context.
2. Never use outside knowledge.
3. Cite sources as [Cloud Unit N.pdf, p.X].
4. Keep commands, code and numbers exactly as in the context.
5. Be concise, use bullets for multi-part answers.
6. If the context does not contain the answer, reply exactly: "I couldn't find this in the course notes." """

# Backward compatibility alias
SYSTEM_INSTRUCTION = SYSTEM_PROMPT
RAG_SYSTEM_INSTRUCTION = SYSTEM_PROMPT

FALLBACK_RESPONSE = "I couldn't find this in the course notes."

FEW_SHOT_EXAMPLES = """--- Example 1 ---
Context:
[1] (Cloud Unit 1.pdf, p.7) Elasticity can be expressed mathematically as: E = (Delta R / R) / (Delta L / L), where R is resources allocated and L is workload demand.

Question: How is elasticity expressed as an equation in the notes?

Answer:
- Elasticity equation: E = (Delta R / R) / (Delta L / L), where R is resources allocated and L is workload demand [Cloud Unit 1.pdf, p.7]

Sources:
- Cloud Unit 1.pdf, p.7

--- Example 2 ---
Context:
[1] (Cloud Unit 4.pdf, p.5) Kubernetes manages containerized applications across a cluster of nodes using Pods, Deployments, and Services.

Question: Who won the 2018 FIFA World Cup?

Answer:
I couldn't find this in the course notes.

Sources:
"""

PLAIN_PROMPT = "You are a helpful AI assistant. Answer the following question concisely using your general knowledge."


def build_rag_prompt(question: str, chunks: list) -> str:
    """Format numbered context blocks, few-shot examples, and the question with required output format."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        if isinstance(c, dict):
            source = c.get("source", "Unknown")
            page = c.get("page", "?")
            text = c.get("text", "").strip()
            blocks.append(f"[{i}] ({source}, p.{page}) {text}")
        else:
            blocks.append(f"[{i}] {c}")

    context_str = "\n\n".join(blocks)

    return f"""Context:
{context_str}

Examples:
{FEW_SHOT_EXAMPLES}

Question: {question}

Required output format:
Answer:
<concise answer citing sources as [Cloud Unit N.pdf, p.X], using bullets for multi-part answers, or exactly "I couldn't find this in the course notes.">

Sources:
<list each source on a new line as "- Cloud Unit N.pdf, p.X", or leave empty if the answer was not found>"""
