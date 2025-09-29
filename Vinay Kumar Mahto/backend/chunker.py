from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        :param chunk_size: Number of characters per chunk
        :param overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += self.chunk_size - self.overlap
        return chunks
