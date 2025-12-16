"""
Token-aware chunking with header-aware splitting for privacy policies.
Optimized for legal documents with semantic boundaries.
"""
import re
from typing import List, Dict, Tuple
import tiktoken

class TokenAwareChunker:
    """
    Chunks text based on tokens with semantic boundary awareness.
    Recommended defaults: 512 tokens, 20-25% overlap
    """
    
    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 100,
        model_name: str = "gpt-3.5-turbo",
        preserve_headers: bool = True
    ):
        """
        Args:
            chunk_tokens: Target chunk size in tokens (default 512)
            overlap_tokens: Overlap between chunks in tokens (default 100, ~20%)
            model_name: Tokenizer model name
            preserve_headers: Try to keep headers with their content
        """
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_headers = preserve_headers
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def identify_headers(self, text: str) -> List[Tuple[int, int, str, int]]:
        """
        Identify section headers in the text.
        Returns list of (start_pos, end_pos, header_text, level)
        """
        headers = []
        
        # Common header patterns in privacy policies
        patterns = [
            # Numbered sections: "1.", "1.1", "A.", etc.
            (r'^(\d+\.|[A-Z]\.|\d+\.\d+\.?)\s+([A-Z][^\n]{10,100})', 1),
            # All caps headers
            (r'^([A-Z\s]{5,50})$', 2),
            # Title case headers
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,8}):?\s*$', 3),
        ]
        
        for pattern, level in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                start = match.start()
                end = match.end()
                header_text = match.group(0).strip()
                headers.append((start, end, header_text, level))
        
        # Sort by position and deduplicate
        headers = sorted(set(headers), key=lambda x: x[0])
        return headers
    
    def split_at_headers(self, text: str) -> List[Dict[str, any]]:
        """
        Split text at major section headers.
        Returns list of sections with metadata.
        """
        headers = self.identify_headers(text)
        
        if not headers:
            return [{"text": text, "header": None, "start": 0, "end": len(text)}]
        
        sections = []
        for i, (start, end, header_text, level) in enumerate(headers):
            # Get content from this header to next header (or end)
            next_start = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            content = text[start:next_start]
            
            sections.append({
                "text": content,
                "header": header_text,
                "level": level,
                "start": start,
                "end": next_start
            })
        
        return sections
    
    def chunk_text_tokens(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text using token-based sliding window.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text, tokens, metadata
        """
        # Encode text to tokens
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_tokens:
            return [{
                "text": text,
                "tokens": total_tokens,
                "start_char": 0,
                "end_char": len(text),
                "metadata": metadata or {}
            }]
        
        chunks = []
        start_idx = 0
        
        while start_idx < total_tokens:
            # Get chunk window
            end_idx = min(start_idx + self.chunk_tokens, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Try to break at sentence boundary if not at end
            if end_idx < total_tokens and len(chunk_text) > 100:
                # Find last sentence boundary
                for sep in ['. ', '.\n', '! ', '?\n']:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > len(chunk_text) * 0.5:  # At least halfway through
                        chunk_text = chunk_text[:last_sep + 1].strip()
                        # Recalculate actual end_idx
                        chunk_tokens = self.tokenizer.encode(chunk_text)
                        end_idx = start_idx + len(chunk_tokens)
                        break
            
            chunks.append({
                "text": chunk_text,
                "tokens": len(chunk_tokens),
                "start_token": start_idx,
                "end_token": end_idx,
                "metadata": metadata or {}
            })
            
            # Move window with overlap
            start_idx = end_idx - self.overlap_tokens
            
            # Prevent infinite loop
            if start_idx >= total_tokens - 10:
                break
        
        return chunks
    
    def chunk_document(self, text: str, url: str = None) -> List[Dict]:
        """
        Main chunking method - combines header-aware and token-based chunking.
        
        Args:
            text: Full document text
            url: Source URL for metadata
            
        Returns:
            List of chunks with full metadata
        """
        all_chunks = []
        chunk_id = 0
        
        if self.preserve_headers:
            # Split at headers first
            sections = self.split_at_headers(text)
            
            for section in sections:
                section_metadata = {
                    "url": url,
                    "header": section.get("header"),
                    "header_level": section.get("level"),
                    "section_start": section["start"]
                }
                
                # Chunk each section
                section_chunks = self.chunk_text_tokens(section["text"], section_metadata)
                
                # Add chunk IDs and position info
                for chunk in section_chunks:
                    chunk["chunk_id"] = chunk_id
                    chunk["start_char"] = section["start"] + chunk.get("start_char", 0)
                    all_chunks.append(chunk)
                    chunk_id += 1
        else:
            # Simple token-based chunking without header awareness
            chunks = self.chunk_text_tokens(text, {"url": url})
            for i, chunk in enumerate(chunks):
                chunk["chunk_id"] = i
                all_chunks.append(chunk)
        
        return all_chunks


# Convenience function for quick chunking
def chunk_privacy_policy(
    text: str,
    url: str = None,
    chunk_size: int = 512,
    overlap: int = 100
) -> List[Dict]:
    """
    Quick chunking function for privacy policies.
    
    Args:
        text: Policy text
        url: Source URL
        chunk_size: Tokens per chunk (default 512)
        overlap: Overlap tokens (default 100)
        
    Returns:
        List of chunks
    """
    chunker = TokenAwareChunker(
        chunk_tokens=chunk_size,
        overlap_tokens=overlap,
        preserve_headers=True
    )
    return chunker.chunk_document(text, url)
