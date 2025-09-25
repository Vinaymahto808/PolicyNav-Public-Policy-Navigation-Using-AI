"""
OCR and document parsing utilities for docx and pdf.
Includes chunking functions for text.
"""

from pathlib import Path
from typing import List, Dict, Any
from docx import Document
from PyPDF2 import PdfReader
import re
import json

# ---------------------------
# Node & Section Path Builders
# ---------------------------

def build_nodes(doc: Document) -> List[Dict[str, Any]]:
    """Convert docx paragraphs into nodes with heading levels and text."""
    nodes = []
    for para in doc.paragraphs:
        style = para.style.name.lower() if para.style and para.style.name else ""
        if "heading" in style:
            level_match = re.search(r"heading (\d+)", style)
            level = int(level_match.group(1)) if level_match else 1
            nodes.append({"level": level, "text": para.text.strip()})
        elif para.text.strip():
            nodes.append({"level": None, "text": para.text.strip()})
    return nodes


def assign_section_paths(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign hierarchical section paths to nodes based on heading levels."""
    section_stack: List[str] = []
    for node in nodes:
        if node["level"] is not None:
            # Adjust section stack based on heading level
            while len(section_stack) >= node["level"]:
                section_stack.pop()
            section_stack.append(node["text"])
        node["section_path"] = " > ".join(section_stack)
    return nodes

# ---------------------------
# Chunking
# ---------------------------

def chunk_by_words(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into word chunks of given size."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def chunk_by_headings(nodes: List[Dict[str, Any]], chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Chunk docx nodes by headings and word count."""
    chunks: List[Dict[str, Any]] = []
    buffer: List[str] = []
    current_section = ""

    def flush():
        nonlocal buffer, current_section
        if buffer:
            text = " ".join(buffer)
            chunks.append({"section": current_section, "text": text})
            buffer = []

    for node in nodes:
        if node["level"] is not None:
            # flush previous buffer when a heading comes
            flush()
            current_section = node["section_path"]
        else:
            words = node["text"].split()
            buffer.extend(words)
            if len(buffer) >= chunk_size:
                flush()

    flush()
    return chunks

# ---------------------------
# Processors
# ---------------------------

def process_docx(path: Path, mode: str = "words", chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Process a .docx file into text chunks."""
    doc = Document(path)
    nodes = build_nodes(doc)
    nodes = assign_section_paths(nodes)

    if mode == "headings":
        return chunk_by_headings(nodes, chunk_size=chunk_size)
    else:
        full_text = " ".join([n["text"] for n in nodes if n["text"]])
        return [{"section": "full", "text": chunk} for chunk in chunk_by_words(full_text, chunk_size)]


def process_pdf(path: Path, chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Process a PDF file into text chunks by words."""
    reader = PdfReader(str(path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    return [{"section": "pdf", "text": chunk} for chunk in chunk_by_words(full_text, chunk_size)]

# ---------------------------
# JSON Export
# ---------------------------

def save_chunks_to_json(chunks: List[Dict[str, Any]], output_path: Path) -> str:
    """Save extracted chunks into a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return str(output_path)

# ---------------------------
# Type Detection
# ---------------------------

def detect_kind(filename: str, mime: str | None = None) -> str:
    """Detect file type kind based on extension or mime."""
    lower = filename.lower()
    if lower.endswith(".docx"):
        return "docx"
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".txt") or lower.endswith(".md"):
        return "text"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".xlsx"):
        return "xlsx"
    if lower.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    if lower.endswith(".mp3"):
        return "audio"
    if lower.endswith(".mp4"):
        return "video"
    return "unknown"

# ---------------------------
# Exports
# ---------------------------

__all__ = [
    "build_nodes",
    "assign_section_paths",
    "chunk_by_words",
    "chunk_by_headings",
    "process_docx",
    "process_pdf",
    "detect_kind",
    "save_chunks_to_json",
]
