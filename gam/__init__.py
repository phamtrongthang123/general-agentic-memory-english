"""
General Agentic Memory (GAM) - 通用智能记忆系统

这个包提供了基于记忆的智能代理架构，支持：
- MemoryAgent: 记忆构建和管理
- DeepResearchAgent: 深度研究代理
- 多种LLM后端支持（OpenRouter、HuggingFace等）
"""

from .agents import (
    MemoryAgent, 
    DeepResearchAgent
)
from .llm_call import BaseLLM, OpenRouterModel, HFModel
from .utils import (
    safe_json_extract,
    build_session_chunks_from_text,
    build_pages_from_sessions_and_abstracts,
    tokenize
)
from .retrieval import BM25Sessions
from .prompts import (
    MemoryAgent_PROMPT,
    SESSION_SUMMARY_PROMPT,
    PLANNING_DEEP_RESEARCH_PROMPT,
    REPLAN_FROM_SESSIONS_PROMPT
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "MemoryAgent",
    "DeepResearchAgent", 
    "BaseLLM",
    "OpenRouterModel",
    "HFModel",
    "safe_json_extract",
    "build_session_chunks_from_text",
    "build_pages_from_sessions_and_abstracts",
    "tokenize",
    "BM25Sessions",
    "MemoryAgent_PROMPT",
    "SESSION_SUMMARY_PROMPT",
    "PLANNING_DEEP_RESEARCH_PROMPT",
    "REPLAN_FROM_SESSIONS_PROMPT"
]
