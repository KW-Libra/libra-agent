from __future__ import annotations

from importlib import import_module

__all__ = [
    "JudgeOrchestrator",
    "AnthropicChatClient",
    "GeminiChatClient",
    "LibraLangGraphRuntime",
    "LibraDecisionStore",
    "LlamaCppServerClient",
    "LocalKnowledgeBase",
    "OllamaChatClient",
    "PortfolioHolding",
    "PortfolioSnapshot",
    "TriggerEvent",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AnthropicChatClient": (".anthropic_client", "AnthropicChatClient"),
    "GeminiChatClient": (".gemini_client", "GeminiChatClient"),
    "JudgeOrchestrator": (".libra_runtime", "JudgeOrchestrator"),
    "LibraLangGraphRuntime": (".libra_graph", "LibraLangGraphRuntime"),
    "LibraDecisionStore": (".libra_store", "LibraDecisionStore"),
    "LlamaCppServerClient": (".llama_cpp_client", "LlamaCppServerClient"),
    "LocalKnowledgeBase": (".libra_runtime", "LocalKnowledgeBase"),
    "OllamaChatClient": (".ollama_client", "OllamaChatClient"),
    "PortfolioHolding": (".libra_models", "PortfolioHolding"),
    "PortfolioSnapshot": (".libra_models", "PortfolioSnapshot"),
    "TriggerEvent": (".libra_models", "TriggerEvent"),
}


def __getattr__(name: str):
    module_name, attribute_name = _LAZY_IMPORTS.get(name, (None, None))
    if module_name is None or attribute_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, attribute_name)
