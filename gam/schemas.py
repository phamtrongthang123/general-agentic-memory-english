# schemas.py
# -*- coding: utf-8 -*-
"""
统一的数据模型定义
使用 Pydantic 实现一个定义，多种用途：
- Python 数据类
- JSON Schema 自动生成
- 数据验证
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Protocol

# =============================
# Core data models (Pydantic)
# =============================

class MemoryState(BaseModel):
    """Long-term memory: only abstracts list."""
    abstracts: List[str] = Field(default_factory=list, description="记忆摘要列表")

class Page(BaseModel):
    """页面数据"""
    header: str = Field(..., description="页面标题")
    content: str = Field(..., description="页面内容")
    meta: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class MemoryUpdate(BaseModel):
    """记忆更新"""
    new_state: MemoryState = Field(..., description="新的记忆状态")
    new_page: Page = Field(..., description="新的页面")
    debug: Dict[str, Any] = Field(default_factory=dict, description="调试信息")

class SearchPlan(BaseModel):
    """搜索计划"""
    info_needs: List[str] = Field(default_factory=list, description="信息需求列表")
    tools: List[str] = Field(default_factory=list, description="使用的工具")
    keyword_collection: List[str] = Field(default_factory=list, description="关键词集合")
    vector_queries: List[str] = Field(default_factory=list, description="向量查询")
    page_indices: List[int] = Field(default_factory=list, description="页面索引")

class ToolResult(BaseModel):
    """工具结果"""
    tool: str = Field(..., description="工具名称")
    inputs: Dict[str, Any] = Field(..., description="输入参数")
    outputs: Any = Field(..., description="输出结果")
    error: Optional[str] = Field(None, description="错误信息")

class Hit(BaseModel):
    """搜索结果命中"""
    page_index: Optional[int] = Field(None, description="页面索引")
    snippet: str = Field(..., description="文本片段")
    source: str = Field(..., description="来源类型")
    meta: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class SourceInfo(BaseModel):
    """来源信息"""
    page_index: Optional[int] = Field(None, description="页面索引")
    snippet: str = Field(..., description="文本片段")
    source: str = Field(..., description="来源类型")

class Result(BaseModel):
    """搜索结果"""
    content: str = Field("", description="整合的内容")
    sources: List[SourceInfo] = Field(default_factory=list, description="来源列表")

class ReflectionDecision(BaseModel):
    """反思决策"""
    enough: bool = Field(..., description="信息是否足够")
    new_request: Optional[str] = Field(None, description="新的请求")

class ResearchOutput(BaseModel):
    """研究输出"""
    integrated_memory: str = Field(..., description="整合的记忆")
    raw_memory: Dict[str, Any] = Field(..., description="原始记忆数据")

class GenerateRequests(BaseModel):
    """生成请求"""
    new_requests: List[str] = Field(..., description="新的请求列表")

# =============================
# Protocols (接口定义)
# =============================

class MemoryStore(Protocol):
    def load(self) -> MemoryState: ...
    def save(self, state: MemoryState) -> None: ...
    def add(self, abstract: str) -> None: ...

class PageStore(Protocol):
    def add(self, page: Page) -> None: ...
    def get(self, index: int) -> Optional[Page]: ...
    def list_all(self) -> List[Page]: ...

class Retriever(Protocol):
    """Unified interface for keyword / vector / page-id retrievers."""
    name: str
    def build(self, pages: List[Page]) -> None: ...
    def search(self, query: str, top_k: int = 10) -> List[Hit]: ...

class Tool(Protocol):
    name: str
    def run(self, **kwargs) -> ToolResult: ...

class ToolRegistry(Protocol):
    def run_many(self, tool_inputs: Dict[str, Dict[str, Any]]) -> List[ToolResult]: ...

# =============================
# In-memory default stores
# =============================

class InMemoryMemoryStore:
    def __init__(self, init_state: Optional[MemoryState] = None) -> None:
        self._state = init_state or MemoryState()

    def load(self) -> MemoryState:
        return self._state

    def save(self, state: MemoryState) -> None:
        self._state = state

    def add(self, abstract: str) -> None:
        """Add a new abstract to memory if it doesn't already exist."""
        if abstract and abstract not in self._state.abstracts:
            self._state.abstracts.append(abstract)

class InMemoryPageStore:
    """
    Simple append-only list store for Page.
    Uses index-based access.
    """
    def __init__(self) -> None:
        self._pages: List[Page] = []

    def add(self, page: Page) -> None:
        self._pages.append(page)

    def get(self, index: int) -> Optional[Page]:
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def list_all(self) -> List[Page]:
        return list(self._pages)

# =============================
# 自动生成 JSON Schema
# =============================

# 为 LLM 调用生成 JSON Schema
PLANNING_SCHEMA = SearchPlan.model_json_schema()
INTEGRATE_SCHEMA = Result.model_json_schema()
INFO_CHECK_SCHEMA = ReflectionDecision.model_json_schema()
GENERATE_REQUESTS_SCHEMA = GenerateRequests.model_json_schema()