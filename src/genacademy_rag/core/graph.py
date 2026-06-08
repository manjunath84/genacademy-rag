"""The one LangGraph graph: retrieve → grade → {answer + citations | refuse}. Dependencies
(retriever, provider) are injected so the graph is unit-tested against fakes."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from genacademy_rag.core.grader import grade_answerability
from genacademy_rag.core.types import GraphState

REFUSAL_MESSAGE = "I could not find this in the course materials."
ANSWER_SYSTEM = (
    "You answer ONLY from the provided course context. If the context does not contain the "
    "answer, say you could not find it. Never use outside knowledge. Be concise."
)


def build_graph(*, retriever, provider, cosine_threshold: float = 0.2):
    def retrieve_node(state: GraphState) -> dict:
        return {"retrieved": retriever.retrieve(state["question"])}

    def grade_node(state: GraphState) -> dict:
        g = grade_answerability(state["question"], state["retrieved"], provider,
                                cosine_threshold=cosine_threshold)
        return {"answerable": g.answerable, "confidence": g.confidence}

    def answer_node(state: GraphState) -> dict:
        context = "\n---\n".join(r.chunk.text for r in state["retrieved"])
        answer = provider.generate(
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"}],
            json_mode=False, max_tokens=512,
        )
        citations = [r.chunk.citation for r in state["retrieved"]]
        return {"answer": answer, "citations": citations, "refused": False}

    def refuse_node(state: GraphState) -> dict:
        return {"answer": REFUSAL_MESSAGE,
                "citations": [r.chunk.citation for r in state["retrieved"]], "refused": True}

    def route(state: GraphState) -> str:
        return "answer" if state["answerable"] else "refuse"

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("answer", answer_node)
    g.add_node("refuse", refuse_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"answer": "answer", "refuse": "refuse"})
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    return g.compile()
