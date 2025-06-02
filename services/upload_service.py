# services/upload_service.py

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


    # Define a dummy class if not available to prevent runtime errors if RAG_AVAILABLE check is missed upstream
    class SentenceTransformer:
        def __init__(self, model_name_or_path=None, device=None, cache_folder=None, use_auth_token=None):
            pass

        def encode(self, sentences, batch_size=32, show_progress_bar=None, output_value='sentence_embedding',
                   convert_to_numpy=True, convert_to_tensor=False, device=None, normalize_embeddings=False):
            logging.warning("SentenceTransformer dummy 'encode' called. Embeddings will not be generated.")
            # Return a list of zero vectors of a typical dimension if needed for compatibility
            # Assuming typical dimension like 384 for MiniLM
            if isinstance(sentences, str):
                return [[0.0] * 384]
            return [[0.0] * 384 for _ in sentences]

from .vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID
from .chunking_service import ChunkingService

logger = logging.getLogger(__name__)


class UploadService:
    """
    File upload and processing service for AvA RAG system
    Handles file reading, chunking, embedding generation, and vector storage
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedder: Optional[SentenceTransformer] = None
        self.embedder_ready = False
        # self._embedder_init_task = None # Removed, direct await now

        # Initialize services (these are synchronous)
        self.vector_db = VectorDBService()  # Assumes VectorDBService constructor is safe
        self.chunking_service = ChunkingService()  # Assumes ChunkingService constructor is safe

        # Supported file types - EXPANDED
        self.supported_extensions = {
            # Code
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.swift', '.kt', '.go', '.rs', '.rb', '.php', '.pl', '.sh',
            '.ipynb',  # Jupyter Notebooks (content extraction needed)
            # Text & Markup
            '.txt', '.md', '.rst', '.tex', '.log',
            # Config & Data
            '.json', '.yaml', '.yml', '.xml', '.csv', '.ini', '.toml',
            # Web
            '.html', '.css', '.scss', '.less',
            # Documents (basic support - content extraction from binary formats needs more work)
            '.pdf', '.docx',  # Added PDF and DOCX
            # SQL
            '.sql',
            # Shell & Batch
            '.bat', '.ps1'
        }
        logger.info(
            f"UploadService initialized. Embedder will be loaded asynchronously. RAG Available: {SENTENCE_TRANSFORMERS_AVAILABLE}")

    async def async_internal_init(self) -> bool:
        """
        Asynchronously initialize the embedder.
        This method should be called and awaited after the event loop is running.
        Returns True if initialization was successful, False otherwise.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("UploadService: SentenceTransformers not available - cannot initialize embedder.")
            self.embedder_ready = False
            return False

        if self.embedder_ready:
            logger.info("UploadService: Embedder already initialized.")
            return True

        logger.info("UploadService: Starting embedder initialization...")
        await self._initialize_embedder()  # Directly await the initialization
        return self.embedder_ready

    async def _initialize_embedder(self):
        """Initialize the sentence transformer model asynchronously"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:  # Should be caught by async_internal_init already
            logger.error("UploadService: _initialize_embedder called but SentenceTransformers not available.")
            self.embedder_ready = False
            return

        try:
            logger.info(f"UploadService: Loading embedding model: {self.embedding_model_name}...")
            loop = asyncio.get_event_loop()  # Get the current event loop
            # Run the blocking SentenceTransformer loading in an executor
            self.embedder = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                SentenceTransformer,  # The callable
                self.embedding_model_name  # Arguments to SentenceTransformer
            )
            self.embedder_ready = True
            logger.info("UploadService: Embedding model loaded successfully.")

        except Exception as e:
            logger.error(f"UploadService: Failed to load embedding model '{self.embedding_model_name}': {e}",
                         exc_info=True)
            self.embedder = None
            self.embedder_ready = False

    async def wait_for_embedder_ready(self, timeout_seconds: int = 60) -> bool:
        """
        Wait for embedder to be ready.
        Assumes async_internal_init has been called and is being awaited elsewhere,
        or this method is called after async_internal_init has completed.
        """
        if self.embedder_ready:
            return True

        # If async_internal_init wasn't called or completed, this check might be too simple.
        # However, with direct await in async_internal_init, this should mostly just confirm.
        logger.info("UploadService: Waiting for embedder to become ready...")
        start_time = time.time()
        while not self.embedder_ready:
            if (time.time() - start_time) > timeout_seconds:
                logger.error(
                    f"UploadService: Timeout waiting for embedder model '{self.embedding_model_name}' to load.")
                return False
            await asyncio.sleep(0.2)  # Poll status

        logger.info("UploadService: Embedder is now ready.")
        return self.embedder_ready

    def process_files_for_context(self, file_paths: List[Union[str, Path]],
                                  collection_id: str = GLOBAL_COLLECTION_ID) -> Optional['UploadResult']:
        """
        Process multiple files and add them to the vector database.
        Note: This method itself is synchronous and uses the embedder.
        Ensure embedder is ready before calling or handle async context appropriately if calling from async code.
        """
        if not self.embedder_ready:
            logger.warning(
                "UploadService: Embedder not ready - cannot process files. Call and await async_internal_init() and wait_for_embedder_ready() first.")
            return None

        results = {'successfully_processed': [], 'failed_files': [], 'total_chunks': 0, 'processing_time': 0.0}
        start_time = time.time()

        for file_path_item in file_paths:
            try:
                file_path = Path(file_path_item)
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    results['failed_files'].append(str(file_path))
                    continue
                if file_path.suffix.lower() not in self.supported_extensions:
                    logger.info(f"Skipping unsupported file type: {file_path}")  # Changed to INFO for less noise
                    # results['failed_files'].append(str(file_path)) # Don't treat as failure, just skip
                    continue

                file_result = self._process_single_file(file_path, collection_id)
                if file_result:
                    results['successfully_processed'].append(str(file_path))
                    results['total_chunks'] += file_result.get('chunk_count', 0)
                else:
                    results['failed_files'].append(str(file_path))
            except Exception as e:
                logger.error(f"Failed to process {file_path_item}: {e}", exc_info=True)
                results['failed_files'].append(str(file_path_item))

        results['processing_time'] = time.time() - start_time
        upload_result = UploadResult(
            successfully_added_files=len(results['successfully_processed']),
            failed_files=len(results['failed_files']),
            total_chunks=results['total_chunks']
        )
        upload_result.metadata = {'upload_summary_v5_final': results.copy()}  # Store detailed summary
        logger.info(f"UploadService: Processed {len(results['successfully_processed'])} files, "
                    f"{results['total_chunks']} chunks in {results['processing_time']:.2f}s. Failed: {len(results['failed_files'])}.")
        return upload_result

    def process_directory_for_context(self, directory_path: Union[str, Path],
                                      collection_id: str = GLOBAL_COLLECTION_ID,
                                      recursive: bool = True) -> Optional['UploadResult']:
        """Process all supported files in a directory."""
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return None

        all_found_files = []
        pattern_gen = directory_path.rglob if recursive else directory_path.glob

        # Log all files found before filtering
        raw_file_list = list(pattern_gen("*"))
        logger.info(
            f"UploadService: Scanned {directory_path}. Found {len(raw_file_list)} total items (files/dirs) before filtering.")

        # Filter for supported extensions
        file_paths_by_extension = []
        for ext in self.supported_extensions:
            # Reset generator for each extension or glob once and filter
            # For simplicity, re-globbing per extension or use a single glob and filter
            file_paths_by_extension.extend(
                directory_path.rglob(f"*{ext}") if recursive else directory_path.glob(f"*{ext}"))

        # Deduplicate (if rglobbing '*' and then filtering by suffix)
        # Since we glob per extension, it should be mostly unique file paths
        # unique_file_paths = list(set(file_paths_by_extension))
        # logger.info(f"UploadService: Found {len(unique_file_paths)} files matching supported extensions before ignoring directories.")

        ignore_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.env', 'build', 'dist', '.idea',
                       '.vscode'}
        # Filter out ignored directories and ensure they are files
        valid_file_paths = [
            f for f in file_paths_by_extension
            if f.is_file() and not any(part in ignore_dirs for part in f.parts)
        ]
        # Deduplicate final list as rglobbing per extension might overlap if a file somehow matches multiple (unlikely for distinct exts)
        valid_file_paths = list(set(valid_file_paths))

        logger.info(f"UploadService: Found {len(valid_file_paths)} supported and non-ignored files in {directory_path}")
        if not valid_file_paths:
            logger.info(f"UploadService: No processable files found in {directory_path}. Returning empty result.")
            return UploadResult(0, 0, 0)

        return self.process_files_for_context(valid_file_paths, collection_id)

    def _process_single_file(self, file_path: Path, collection_id: str) -> Optional[Dict[str, Any]]:
        """Process a single file (synchronous embedding)."""
        if not self.embedder_ready or not self.embedder:  # Added check for self.embedder instance
            logger.error(f"UploadService: Embedder not ready or not initialized, cannot process file {file_path.name}")
            return None
        try:
            # TODO: Implement specific content readers for .pdf, .docx using pypdf, python-docx
            # For now, it will only work well with text-based files in supported_extensions
            content = self._read_file_content(file_path)
            if not content:
                logger.warning(f"Could not read content or content is empty for {file_path}. Skipping.")
                return None

            file_metadata = {
                'filename': file_path.name, 'file_path': str(file_path),
                'file_extension': file_path.suffix, 'file_size': len(content),
                'file_hash': hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()
            }
            chunks = self.chunking_service.chunk_document(content, str(file_path), file_metadata)  # Pass str(file_path)
            if not chunks: logger.warning(f"No chunks generated for {file_path}"); return None

            documents_with_embeddings = []
            for chunk in chunks:
                try:
                    # Embedding is CPU/GPU bound, ensure it's handled correctly in async contexts if called from one
                    embedding = self.embedder.encode(chunk['content'], convert_to_tensor=False)
                    doc = {
                        'id': f"{file_metadata['file_hash']}_{chunk['id']}",
                        'content': chunk['content'], 'metadata': chunk['metadata'],
                        'embedding': embedding.tolist()  # Ensure it's a list for ChromaDB
                    }
                    documents_with_embeddings.append(doc)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {chunk['id']} from {file_path.name}: {e}",
                                 exc_info=True)

            if documents_with_embeddings:
                self.vector_db.add_documents(collection_id, documents_with_embeddings)
                logger.info(
                    f"UploadService: Added {len(documents_with_embeddings)} chunks from {file_path.name} to {collection_id}")
                return {'file_path': str(file_path), 'chunk_count': len(documents_with_embeddings),
                        'file_size': len(content)}
            logger.warning(f"No documents with embeddings were prepared for {file_path.name}")
            return None
        except Exception as e:
            logger.error(f"UploadService: Failed to process file {file_path}: {e}", exc_info=True)
            return None

    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding detection."""
        # Placeholder: Actual binary file content extraction (PDF, DOCX) needs specific libraries.
        if file_path.suffix.lower() in ['.pdf', '.docx']:
            logger.warning(
                f"Content extraction for {file_path.suffix} not fully implemented. Reading as plain text if possible or skipping.")
            # Attempt to read as text for now, or return empty to indicate it needs proper handling
            try:
                return file_path.read_text(encoding='utf-8', errors='ignore')  # very basic for now
            except:
                return ""

        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {file_path}. Trying latin-1.")
            try:
                return file_path.read_text(encoding='latin-1')
            except Exception as e_latin1:
                logger.error(f"Failed to read {file_path} with UTF-8 and latin-1: {e_latin1}", exc_info=True)
                return None
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}", exc_info=True)
            return None

    def query_vector_db(self, query: str, collection_ids: Optional[List[str]] = None,
                        n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the vector database for similar content."""
        if not self.embedder_ready:
            logger.warning("UploadService: Embedder not ready - cannot query vector DB.")
            return []
        if not self.vector_db:
            logger.error("UploadService: VectorDBService not initialized.")
            return []

        effective_collection_ids = collection_ids or [GLOBAL_COLLECTION_ID]
        all_results: List[Dict[str, Any]] = []

        for collection_id in effective_collection_ids:
            try:
                # Embedding the query text is synchronous here
                # query_embedding = self.embedder.encode(query, convert_to_tensor=False).tolist()
                # ChromaDB can take query_texts directly if an embedding function is configured for the collection,
                # or you can pass query_embeddings. Assuming VectorDBService handles this.
                results = self.vector_db.query_collection(collection_id, query, n_results)  # Pass text query
                for result in results: result['collection_id'] = collection_id
                all_results.extend(results)
            except Exception as e:
                logger.error(f"UploadService: Failed to query collection {collection_id}: {e}", exc_info=True)

        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        return all_results[:n_results]

    # ... (other methods like get_collection_stats, delete_file_chunks)

    def get_collection_stats(self, collection_id: str = GLOBAL_COLLECTION_ID) -> Dict[str, Any]:
        """Get statistics about a collection"""
        if not self.vector_db: return {}
        return self.vector_db.get_collection_stats(collection_id)

    def delete_file_chunks(self, file_path: Union[str, Path],
                           collection_id: str = GLOBAL_COLLECTION_ID):
        """Delete all chunks for a specific file (simplified)."""
        # This is a placeholder. Proper chunk deletion requires robust tracking of chunk IDs associated with a file.
        # A common way is to store file_hash in chunk metadata and delete by a 'where' filter.
        # Example: collection.delete(where={"file_hash": "hash_of_file_path"})
        logger.warning(
            f"Placeholder: Deleting chunks for {file_path} in {collection_id}. Implement proper chunk ID tracking for deletion.")
        # For now, this does nothing to avoid accidental data loss without proper implementation.
        # Example of what it might look like with metadata filtering (if supported and implemented in VectorDBService):
        # content = self._read_file_content(Path(file_path))
        # if content:
        #     file_hash = hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()
        #     self.vector_db.delete_documents_by_metadata(collection_id, {"file_hash": file_hash})


class UploadResult:
    """Result object for upload operations"""

    def __init__(self, successfully_added_files: int, failed_files: int, total_chunks: int):
        self.successfully_added_files = successfully_added_files
        self.failed_files = failed_files
        self.total_chunks = total_chunks
        self.metadata: Dict[str, Any] = {}

    def __str__(self):
        return (f"UploadResult(files_added: {self.successfully_added_files}, "
                f"files_failed: {self.failed_files}, total_chunks: {self.total_chunks})")