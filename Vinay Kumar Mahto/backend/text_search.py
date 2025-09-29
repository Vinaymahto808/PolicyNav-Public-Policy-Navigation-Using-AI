
import re
from typing import List, Tuple
from difflib import SequenceMatcher

class TextSearcher:
    def __init__(self, text_chunks: List[str]):
        self.text_chunks = text_chunks
        self.full_text = " ".join(text_chunks)
    
    def search_relevant_chunks(self, query: str, max_chunks: int = 5) -> List[str]:
        """Find the most relevant text chunks based on the query."""
        if not query or not self.text_chunks:
            return self.text_chunks[:max_chunks]
        
        # Score each chunk based on keyword matches and similarity
        chunk_scores = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        for i, chunk in enumerate(self.text_chunks):
            chunk_lower = chunk.lower()
            
            # Keyword matching score
            chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
            keyword_matches = len(query_words.intersection(chunk_words))
            
            # Similarity score
            similarity = SequenceMatcher(None, query_lower, chunk_lower[:len(query_lower)]).ratio()
            
            # Combined score
            score = keyword_matches * 2 + similarity
            chunk_scores.append((score, i, chunk))
        
        # Sort by score and return top chunks
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, _, chunk in chunk_scores[:max_chunks]]
    
    def find_text_matches(self, search_term: str) -> List[Tuple[int, str]]:
        """Find exact matches of a search term in chunks."""
        matches = []
        for i, chunk in enumerate(self.text_chunks):
            if search_term.lower() in chunk.lower():
                matches.append((i, chunk))
        return matches
