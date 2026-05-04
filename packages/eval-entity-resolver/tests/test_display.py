import pytest

from eval_entity_resolver.display import humanize_model_slug


@pytest.mark.parametrize(
    "slug,expected",
    [
        # Acronym + version: glue with hyphen
        ("gpt-5", "GPT-5"),
        ("gpt-5-mini", "GPT-5 Mini"),
        ("gpt-4o", "GPT-4o"),
        ("gpt-4o-mini", "GPT-4o Mini"),
        ("glm-4.5", "GLM-4.5"),
        ("glm-5-thinking", "GLM-5 Thinking"),
        # Acronym followed by param size: do NOT glue (size suffix `b`)
        ("deepseek-llm-67b-chat", "DeepSeek LLM 67B Chat"),
        ("qwen2.5-vl-7b-instruct", "Qwen2.5 VL 7B Instruct"),
        # 8-digit date suffix
        ("gpt-4-0613", "GPT-4 (0613)"),
        ("gpt-4o-2024-05-13", "GPT-4o (2024-05-13)"),
        ("gpt-4o-2024-08-06", "GPT-4o (2024-08-06)"),
        ("claude-opus-4-20250514", "Claude Opus 4 (2025-05-14)"),
        ("claude-3.5-sonnet-20240620", "Claude 3.5 Sonnet (2024-06-20)"),
        ("claude-3.5-sonnet-20241022", "Claude 3.5 Sonnet (2024-10-22)"),
        ("glm-4.5-2025-08-22", "GLM-4.5 (2025-08-22)"),
        # 4-digit code (MMDD) — vendor convention (Grok 2 1212, Kimi-K2 0711)
        ("grok-4-0709", "Grok 4 (0709)"),
        ("grok-2-1212", "Grok 2 (1212)"),
        ("kimi-k2-0711", "Kimi K2 (0711)"),
        # Param sizes
        ("llama-3.1-8b", "Llama 3.1 8B"),
        ("llama-3.1-8b-instruct", "Llama 3.1 8B Instruct"),
        ("mistral-7b", "Mistral 7B"),
        # MoE active-expert form
        ("qwen3-30b-a3b", "Qwen3 30B A3B"),
        ("qwen3-235b-a22b-instruct", "Qwen3 235B A22B Instruct"),
        # Mixture-of-X notation
        ("mixtral-8x7b", "Mixtral 8x7B"),
        # O-series stays lowercase
        ("o1", "o1"),
        ("o3", "o3"),
        ("o3-pro-2025-06-10", "o3 Pro (2025-06-10)"),
        # Vendor token override
        ("deepseek-r1", "DeepSeek R1"),
        ("deepseek-v3", "DeepSeek V3"),
        # Cohere `-MM-YYYY` convention
        ("command-r-08-2024", "Command R (2024-08)"),
        ("command-a-03-2025", "Command A (2025-03)"),
        # `-YYYY-MM` (year-month, no day): incomplete date snapshots
        ("openai/gpt-5-2025-08", "GPT-5 (2025-08)"),
        ("google/gemini-2.5-flash-2025-04", "Gemini 2.5 Flash (2025-04)"),
        # Full canonical id (org prefix stripped)
        ("openai/gpt-5-mini", "GPT-5 Mini"),
        ("anthropic/claude-opus-4.1-20250805", "Claude Opus 4.1 (2025-08-05)"),
    ],
)
def test_humanize_model_slug(slug: str, expected: str) -> None:
    assert humanize_model_slug(slug) == expected


def test_humanize_empty() -> None:
    assert humanize_model_slug("") == ""
