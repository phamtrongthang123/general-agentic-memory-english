# -*- coding: utf-8 -*-
"""
Text Chunking Utilities Module

Provides multiple text chunking strategies:
- Chunking by token count
- Chunking by sentences
- Smart chunking (combining paragraph and sentence boundaries)
"""

import re
from typing import List, Optional
import tiktoken


def chunk_text_by_tokens(
    text: str,
    max_tokens: int = 2000,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk text by token count

    Args:
        text: Text to be chunked
        max_tokens: Maximum number of tokens per chunk
        encoding_name: tiktoken encoding name

    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        # If encoding cannot be loaded, use simple character chunking
        return _chunk_by_chars(text, max_tokens * 4)  # Rough estimate
    
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks


def chunk_text_by_sentences(
    text: str,
    max_tokens: int = 2000,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk text by sentence boundaries, keeping sentences intact as much as possible

    Args:
        text: Text to be chunked
        max_tokens: Maximum number of tokens per chunk
        encoding_name: tiktoken encoding name

    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        return chunk_text_by_tokens(text, max_tokens, encoding_name)

    # Try to use NLTK for sentence splitting
    sentences = _split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))

        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Current chunk is full, save it and start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_text_smartly(
    text: str,
    max_tokens: int = 2000,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Smart chunking: combines paragraph and sentence boundaries

    Priority: paragraph > sentence > token

    Args:
        text: Text to be chunked
        max_tokens: Maximum number of tokens per chunk
        encoding_name: tiktoken encoding name

    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        return chunk_text_by_tokens(text, max_tokens, encoding_name)

    # First split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = len(encoding.encode(para))

        # If the paragraph itself exceeds the limit, split by sentences
        if para_tokens > max_tokens:
            # First save the current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split large paragraph
            sentences = _split_into_sentences(para)
            sent_chunk = []
            sent_tokens = 0

            for sent in sentences:
                sent_token_count = len(encoding.encode(sent))

                if sent_tokens + sent_token_count > max_tokens and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = [sent]
                    sent_tokens = sent_token_count
                else:
                    sent_chunk.append(sent)
                    sent_tokens += sent_token_count

            if sent_chunk:
                chunks.append(" ".join(sent_chunk))

        elif current_tokens + para_tokens > max_tokens and current_chunk:
            # Current chunk is full, save it and start a new chunk
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # Add the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences

    Prefer NLTK if available, otherwise use regular expressions
    """
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        return nltk.sent_tokenize(text)
    except Exception:
        # Fall back to simple regex splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def _chunk_by_chars(text: str, max_chars: int) -> List[str]:
    """Simple character-based chunking (fallback method)"""
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i + max_chars])
    return chunks

