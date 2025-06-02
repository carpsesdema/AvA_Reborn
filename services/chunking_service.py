# services/chunking_service.py

import re
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Smart chunking service for breaking documents into optimal pieces for RAG
    Handles Python code, text files, and other document types
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Python-specific patterns
        self.python_patterns = {
            'class': re.compile(r'^class\s+\w+.*?:', re.MULTILINE),
            'function': re.compile(r'^def\s+\w+.*?:', re.MULTILINE),
            'import': re.compile(r'^(?:from\s+\S+\s+)?import\s+.+', re.MULTILINE),
            'comment_block': re.compile(r'"""[\s\S]*?"""', re.MULTILINE),
            'docstring': re.compile(r'^\s*"""[\s\S]*?"""', re.MULTILINE)
        }

    def chunk_document(self, content: str, file_path: str = None, metadata: Dict[str, Any] = None) -> List[
        Dict[str, Any]]:
        """
        Chunk a document into optimal pieces based on file type

        Args:
            content: Document content
            file_path: Path to the file (used to determine type)
            metadata: Additional metadata to include

        Returns:
            List of chunk dictionaries
        """

        if not content or not content.strip():
            return []

        file_path = Path(file_path) if file_path else None
        file_extension = file_path.suffix.lower() if file_path else ''

        # Determine chunking strategy based on file type
        if file_extension == '.py':
            chunks = self._chunk_python_code(content, file_path, metadata)
        elif file_extension in ['.md', '.txt', '.rst']:
            chunks = self._chunk_markdown_text(content, file_path, metadata)
        elif file_extension in ['.js', '.ts', '.jsx', '.tsx']:
            chunks = self._chunk_javascript_code(content, file_path, metadata)
        else:
            chunks = self._chunk_generic_text(content, file_path, metadata)

        logger.debug(f"Chunked {file_path or 'content'} into {len(chunks)} pieces")
        return chunks

    def _chunk_python_code(self, content: str, file_path: Path = None, metadata: Dict[str, Any] = None) -> List[
        Dict[str, Any]]:
        """Smart chunking for Python code files"""
        chunks = []
        lines = content.split('\n')

        # Extract imports first
        imports = []
        non_import_start = 0

        for i, line in enumerate(lines):
            if (line.strip().startswith('import ') or
                    line.strip().startswith('from ') or
                    line.strip() == '' or
                    line.strip().startswith('#')):
                imports.append(line)
                non_import_start = i + 1
            else:
                break

        import_text = '\n'.join(imports).strip()

        # Process the rest of the file
        remaining_content = '\n'.join(lines[non_import_start:])

        # Add imports as first chunk if they exist
        if import_text:
            chunks.append(self._create_chunk(
                import_text,
                chunk_id=f"{file_path.stem if file_path else 'content'}_imports",
                chunk_type="imports",
                file_path=file_path,
                metadata=metadata
            ))

        # Find class and function boundaries
        code_blocks = self._extract_python_blocks(remaining_content)

        current_chunk = ""
        current_chunk_id = 0

        for block in code_blocks:
            # If adding this block would exceed chunk size, save current chunk
            if (len(current_chunk) + len(block['content']) > self.chunk_size and
                    current_chunk.strip()):

                chunks.append(self._create_chunk(
                    current_chunk.strip(),
                    chunk_id=f"{file_path.stem if file_path else 'content'}_chunk_{current_chunk_id}",
                    chunk_type="code",
                    file_path=file_path,
                    metadata=metadata
                ))
                current_chunk_id += 1

                # Start new chunk with overlap
                current_chunk = self._get_overlap_content(current_chunk) + block['content']
            else:
                current_chunk += block['content'] + '\n'

        # Add final chunk if there's content
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(),
                chunk_id=f"{file_path.stem if file_path else 'content'}_chunk_{current_chunk_id}",
                chunk_type="code",
                file_path=file_path,
                metadata=metadata
            ))

        return chunks

    def _extract_python_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract logical Python code blocks (classes, functions, etc.)"""
        lines = content.split('\n')
        blocks = []
        current_block = []
        current_indent = 0
        block_type = "code"

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                current_block.append(line)
                i += 1
                continue

            # Check for class or function definition
            if (stripped.startswith('class ') or stripped.startswith('def ') or
                    stripped.startswith('async def ')):

                # Save previous block if it exists
                if current_block:
                    blocks.append({
                        'content': '\n'.join(current_block),
                        'type': block_type
                    })
                    current_block = []

                # Start new block
                block_type = 'class' if stripped.startswith('class ') else 'function'
                current_indent = len(line) - len(line.lstrip())
                current_block.append(line)

                # Add all indented lines that belong to this block
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()

                    # Empty lines are part of the block
                    if not next_stripped:
                        current_block.append(next_line)
                        i += 1
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip())

                    # If line is indented more than the definition, it's part of the block
                    if next_indent > current_indent:
                        current_block.append(next_line)
                        i += 1
                    else:
                        # End of block
                        break

                # Save the completed block
                blocks.append({
                    'content': '\n'.join(current_block),
                    'type': block_type
                })
                current_block = []
                block_type = "code"
                continue

            # Regular line
            current_block.append(line)
            i += 1

        # Add final block
        if current_block:
            blocks.append({
                'content': '\n'.join(current_block),
                'type': block_type
            })

        return blocks

    def _chunk_markdown_text(self, content: str, file_path: Path = None, metadata: Dict[str, Any] = None) -> List[
        Dict[str, Any]]:
        """Smart chunking for Markdown and text files"""
        chunks = []

        # Split by major headers first
        sections = re.split(r'\n(?=# )', content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # If section is small enough, use as single chunk
            if len(section) <= self.chunk_size:
                chunks.append(self._create_chunk(
                    section.strip(),
                    chunk_id=f"{file_path.stem if file_path else 'content'}_section_{i}",
                    chunk_type="section",
                    file_path=file_path,
                    metadata=metadata
                ))
            else:
                # Split large sections further
                sub_chunks = self._split_text_by_size(section, self.chunk_size, self.chunk_overlap)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append(self._create_chunk(
                        sub_chunk.strip(),
                        chunk_id=f"{file_path.stem if file_path else 'content'}_section_{i}_chunk_{j}",
                        chunk_type="text",
                        file_path=file_path,
                        metadata=metadata
                    ))

        return chunks

    def _chunk_javascript_code(self, content: str, file_path: Path = None, metadata: Dict[str, Any] = None) -> List[
        Dict[str, Any]]:
        """Basic chunking for JavaScript/TypeScript files"""
        return self._chunk_generic_text(content, file_path, metadata)

    def _chunk_generic_text(self, content: str, file_path: Path = None, metadata: Dict[str, Any] = None) -> List[
        Dict[str, Any]]:
        """Generic text chunking for unknown file types"""
        chunks = []
        text_chunks = self._split_text_by_size(content, self.chunk_size, self.chunk_overlap)

        for i, chunk_text in enumerate(text_chunks):
            chunks.append(self._create_chunk(
                chunk_text.strip(),
                chunk_id=f"{file_path.stem if file_path else 'content'}_chunk_{i}",
                chunk_type="text",
                file_path=file_path,
                metadata=metadata
            ))

        return chunks

    def _split_text_by_size(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks of specified size with overlap"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If this isn't the last chunk, try to break at a good boundary
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break > start:
                    end = paragraph_break
                else:
                    # Look for sentence break
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break > start:
                        end = sentence_break + 1
                    else:
                        # Look for line break
                        line_break = text.rfind('\n', start, end)
                        if line_break > start:
                            end = line_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - overlap)

        return chunks

    def _get_overlap_content(self, content: str) -> str:
        """Get the last part of content for overlap"""
        if len(content) <= self.chunk_overlap:
            return content

        # Try to break at a good boundary
        lines = content.split('\n')
        overlap_lines = []
        char_count = 0

        for line in reversed(lines):
            if char_count + len(line) > self.chunk_overlap:
                break
            overlap_lines.insert(0, line)
            char_count += len(line) + 1  # +1 for newline

        return '\n'.join(overlap_lines) + '\n'

    def _create_chunk(self, content: str, chunk_id: str, chunk_type: str,
                      file_path: Path = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a standardized chunk dictionary"""

        chunk_metadata = {
            'chunk_type': chunk_type,
            'chunk_size': len(content),
            'filename': file_path.name if file_path else 'unknown',
            'file_extension': file_path.suffix if file_path else '',
            'file_path': str(file_path) if file_path else ''
        }

        # Add any additional metadata
        if metadata:
            chunk_metadata.update(metadata)

        return {
            'id': chunk_id,
            'content': content,
            'metadata': chunk_metadata
        }

    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about a list of chunks"""
        if not chunks:
            return {'count': 0, 'total_size': 0, 'avg_size': 0, 'types': {}}

        total_size = sum(len(chunk['content']) for chunk in chunks)
        types = {}

        for chunk in chunks:
            chunk_type = chunk.get('metadata', {}).get('chunk_type', 'unknown')
            types[chunk_type] = types.get(chunk_type, 0) + 1

        return {
            'count': len(chunks),
            'total_size': total_size,
            'avg_size': total_size // len(chunks),
            'types': types
        }