import json
import re
import unicodedata
from typing import List


def safe_json_extract(text: str) -> dict:
    """安全JSON解析（带兜底）"""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def build_session_chunks_from_text(text: str, max_tokens: int = 2000, tokenizer=None, model: str = "gpt-4o-mini") -> List[str]:
    """
    从文本构建session chunks
    
    Args:
        text: 输入文本
        max_tokens: 每个session的最大token数
        tokenizer: 分词器对象，如果为None则根据model自动选择tiktoken编码器
        model: 模型名称，用于选择tokenizer（当tokenizer为None时使用）
        
    Returns:
        List[str]: 构建的session chunks
    """
    if not text.strip():
        return []
    
    # 如果提供了tokenizer，直接使用
    if tokenizer is not None:
        try:
            # 尝试使用提供的tokenizer
            if hasattr(tokenizer, 'encode'):
                tokens = tokenizer.encode(text)
            else:
                # 如果是transformers的tokenizer
                tokens = tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) <= max_tokens:
                return [f"[Session 1]\n{text}"]
            
            # 按max_tokens切分tokens
            chunks = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i + max_tokens]
                if hasattr(tokenizer, 'decode'):
                    chunk_text = tokenizer.decode(chunk_tokens)
                else:
                    # 如果是transformers的tokenizer
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                session_id = len(chunks) + 1
                chunks.append(f"[Session {session_id}]\n{chunk_text}")
            
            return chunks
            
        except Exception as e:
            print(f"Warning: 提供的tokenizer切分失败: {e}，使用字符切分作为fallback")
            return _build_chunks_by_char(text, max_tokens * 4)
    
    # 如果没有提供tokenizer，使用tiktoken（适用于API模型）
    try:
        import tiktoken
        # 根据模型名称获取对应的编码器
        if "gpt-4o" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        elif "gpt-3.5" in model.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "gpt-4" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4-0613")
        elif "claude" in model.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # 默认使用gpt-4o编码器
            encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        
        # 将文本编码为tokens
        tokens = encoding.encode(text, disallowed_special=())
        
        if len(tokens) <= max_tokens:
            # 如果文本长度小于max_tokens，直接返回单个块
            return [f"[Session 1]\n{text}"]
        
        # 按max_tokens切分tokens
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            session_id = len(chunks) + 1
            chunks.append(f"[Session {session_id}]\n{chunk_text}")
        
        return chunks
        
    except Exception as e:
        print(f"Warning: tiktoken切分失败: {e}，使用字符切分作为fallback")
        # 如果tokenizer切分失败，使用字符切分作为fallback
        return _build_chunks_by_char(text, max_tokens * 4)  # 粗略估计：1 token ≈ 4 characters


def _build_chunks_by_char(text: str, max_chars: int) -> List[str]:
    """
    按字符数切分文本的fallback方法
    """
    if len(text) <= max_chars:
        return [f"[Session 1]\n{text}"]
    
    chunks = []
    for i in range(0, len(text), max_chars):
        chunk_text = text[i:i + max_chars]
        session_id = len(chunks) + 1
        chunks.append(f"[Session {session_id}]\n{chunk_text}")
    
    return chunks


def build_pages_from_sessions_and_abstracts(sessions: List[str], session_abstracts: List[str]) -> List[str]:
    """
    从sessions和abstracts构建pages
    
    Args:
        sessions: session列表
        session_abstracts: abstract列表
        
    Returns:
        List[str]: 构建的page列表
    """
    pages = []
    
    for i, session in enumerate(sessions):
        # 获取对应的abstract
        abstract = session_abstracts[i] if i < len(session_abstracts) else ""
        
        # 创建page：abstract + 原始内容
        page_content = f"ABSTRACT: {abstract}\n\nCONTENT: {session}"
        pages.append(page_content)
    
    return pages


# =============== 文本预处理 & 简易分词 ===============
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def normalize(s: str) -> str:
    """文本标准化"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s


def tokenize(s: str) -> List[str]:
    """简易分词"""
    s = normalize(s).lower()
    return _WORD_RE.findall(s)
