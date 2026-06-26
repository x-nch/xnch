from dataclasses import dataclass

from agentmemory import create_memory, search_memory


def _compute_complexity(raw_input: str, metadata: dict) -> float:
    if 'complexity_score' in metadata:
        return float(metadata['complexity_score'])
    length_score = min(len(raw_input) / 500, 1.0)
    has_multipart = 1.0 if any(c in raw_input for c in ['and then', 'after that', 'first', 'finally', 'steps']) else 0.0
    has_code = 0.5 if any(c in raw_input for c in ['```', 'def ', 'class ', 'import ', 'kubectl']) else 0.0
    return min((length_score + has_multipart + has_code) / 3, 1.0)


@dataclass
class ModelRoute:
    model_name: str
    reason: str


def classify_request(raw_input: str, actor_role: str, metadata: dict) -> ModelRoute:
    try:
        similar = search_memory(
            "routing-decisions",
            raw_input,
            n_results=3,
            include_embeddings=False,
        )
        for item in similar:
            mem = item["document"] if isinstance(item, dict) else str(item)
            if isinstance(mem, str):
                try:
                    import json as _json
                    parsed = _json.loads(mem)
                    if parsed.get("raw_input", "").lower().strip() == raw_input.lower().strip():
                        return ModelRoute(
                            model_name=parsed["model_name"],
                            reason=f"recalled: {parsed['reason']}",
                        )
                except Exception:
                    pass
    except Exception:
        pass

    if metadata.get("privacy_sensitive"):
        route = ModelRoute(
            model_name="gemma4-local",
            reason="privacy_sensitive: routed to local model",
        )
        _persist_route(raw_input, actor_role, metadata, route)
        return route

    intent_class = metadata.get("intent_class", "")
    complexity_score = metadata.get("complexity_score", 0.0)

    if intent_class == "EXECUTION":
        route = ModelRoute(
            model_name="gemma4-local",
            reason="intent_class=EXECUTION: routed to local model for low-latency execution",
        )
        _persist_route(raw_input, actor_role, metadata, route)
        return route

    if intent_class == "DECISION":
        complexity_score = _compute_complexity(raw_input, metadata)
        if complexity_score > 0.7:
            route = ModelRoute(
                model_name="claude-judgment",
                reason=f"intent_class=DECISION complexity={complexity_score:.2f}: routed to cloud judgment model",
            )
            _persist_route(raw_input, actor_role, metadata, route)
            return route

    route = ModelRoute(
        model_name="gemma4-local",
        reason="default route: local model",
    )
    _persist_route(raw_input, actor_role, metadata, route)
    return route


def _persist_route(raw_input: str, actor_role: str, metadata: dict, route: ModelRoute) -> None:
    import json as _json
    try:
        create_memory(
            "routing-decisions",
            _json.dumps({
                "raw_input": raw_input,
                "actor_role": actor_role,
                "intent_class": metadata.get("intent_class", ""),
                "complexity_score": metadata.get("complexity_score", 0.0),
                "privacy_sensitive": metadata.get("privacy_sensitive", False),
                "model_name": route.model_name,
                "reason": route.reason,
            }),
            metadata={"model_name": route.model_name, "intent_class": metadata.get("intent_class", "")},
        )
    except Exception:
        pass
