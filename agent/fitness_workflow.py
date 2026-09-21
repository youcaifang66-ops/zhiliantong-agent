from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.intent_classifier import classify_intent
from rag.evaluable_retriever import EvaluableHybridRetriever


RED_FLAGS = ("胸痛", "昏厥", "呼吸困难", "骨折", "剧烈疼痛", "大量出血")


class FitnessState(TypedDict, total=False):
    query: str
    intent: str
    safe: bool
    citations: list[dict]
    response: str
    steps: list[str]


def build_fitness_workflow(retriever: EvaluableHybridRetriever):
    def route(state: FitnessState):
        return {"intent": classify_intent(state["query"]), "steps": ["route_intent"]}

    def safety(state: FitnessState):
        safe = not any(flag in state["query"] for flag in RED_FLAGS)
        return {"safe": safe, "steps": state["steps"] + ["safety_guard"]}

    def retrieve(state: FitnessState):
        hits = retriever.search(state["query"], top_k=3)
        return {
            "citations": [
                {"chunk_id": hit.chunk_id, "source": hit.source, "score": hit.rrf_score}
                for hit in hits
            ],
            "steps": state["steps"] + ["hybrid_retrieval"],
        }

    def compose(state: FitnessState):
        if not state["safe"]:
            response = "检测到需要优先处理的安全风险，请停止训练并及时寻求专业医疗帮助。"
        else:
            sources = "、".join(item["source"] for item in state.get("citations", [])) or "无"
            response = f"已按“{state['intent']}”场景处理；建议依据来源：{sources}。"
        return {"response": response, "steps": state["steps"] + ["compose_response"]}

    graph = StateGraph(FitnessState)
    graph.add_node("route", route)
    graph.add_node("safety", safety)
    graph.add_node("retrieve", retrieve)
    graph.add_node("compose", compose)
    graph.add_edge(START, "route")
    graph.add_edge("route", "safety")
    graph.add_conditional_edges("safety", lambda state: "retrieve" if state["safe"] else "compose")
    graph.add_edge("retrieve", "compose")
    graph.add_edge("compose", END)
    return graph.compile(checkpointer=InMemorySaver())
