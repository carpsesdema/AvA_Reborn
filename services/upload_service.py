# services/upload_service.py

import logging
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import mimetypes
import time

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
        self.embedder = None
        self.embedder_ready = False
        self._embedder_init_task = None  # To hold the asyncio.Task

        # Initialize services (these are synchronous)
        self.vector_db = VectorDBService()
        self.chunking_service = ChunkingService()

        # Supported file types
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.txt', '.md', '.rst', '.html', '.css', '.json', '.yaml', '.yml',
            '.xml', '.sql', '.sh', '.bat', '.ps1', '.php', '.rb', '.go', '.rs'
        }

        # Note: Actual embedder initialization is deferred to async_internal_init

    async def async_internal_init(self) -> bool:
        """
        Asynchronously initialize the embedder.
        This method should be called and awaited after the event loop is running.
        Returns True if initialization task was started, False otherwise.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available - install with: pip install sentence-transformers")
            self.embedder_ready = False  # Ensure it's marked as not ready
            return False

        if self._embedder_init_task is None or self._embedder_init_task.done():
            logger.info("Creating task for embedder initialization...")
            self._embedder_init_task = asyncio.create_task(self._initialize_embedder())
            return True
        logger.info("Embedder initialization task already running or completed.")
        return True

    async def _initialize_embedder(self):
        """Initialize the sentence transformer model asynchronously"""
        # This check is now redundant if async_internal_init does it, but good for safety.
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Attempted to initialize embedder, but SentenceTransformers not available.")
            return

        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")

            loop = asyncio.get_event_loop()
            self.embedder = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.embedding_model_name)
            )

            self.embedder_ready = True
            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            self.embedder_ready = False

    async def wait_for_embedder_ready(self, timeout_seconds: int = 60) -> bool:
        """Wait for embedder to be ready"""
        if self.embedder_ready:
            return True

        if self._embedder_init_task is None:
            logger.warning("wait_for_embedder_ready called before embedder initialization was started.")
            # Optionally, try to start it now, though this might indicate a logic error elsewhere
            # await self.async_internal_init()
            # if self._embedder_init_task is None: # if still None, then something is wrong
            #     return False
            return False

        start_time = time.time()
        while not self.embedder_ready:
            if self._embedder_init_task.done():
                # If the task is done but embedder_ready is false, it means initialization failed.
                # _initialize_embedder should set embedder_ready to true on success.
                if not self.embedder_ready:  # Double check
                    logger.error(
                        "Embedder initialization task completed, but embedder is not ready. Check logs for errors.")
                    return False
                break  # Task is done, and presumably embedder_ready is true if it succeeded

            if (time.time() - start_time) > timeout_seconds:
                logger.error(f"Timeout waiting for embedder model '{self.embedding_model_name}' to load.")
                return False
            await asyncio.sleep(0.2)
        return self.embedder_ready

    def process_files_for_context(self, file_paths: List[Union[str, Path]],
                                  collection_id: str = GLOBAL_COLLECTION_ID) -> Optional['UploadResult']:
        """
        Process multiple files and add them to the vector database

        Args:
            file_paths: List of file paths to process
            collection_id: Collection to add documents to

        Returns:
            UploadResult with processing summary
        """

        if not self.embedder_ready:
            logger.warning(
                "Embedder not ready - cannot process files. Call and await async_internal_init() and wait_for_embedder_ready().")
            return None

        results = {
            'successfully_processed': [],
            'failed_files': [],
            'total_chunks': 0,
            'processing_time': 0
        }

        start_time = time.time()

        for file_path in file_paths:
            try:
                file_path = Path(file_path)

                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    results['failed_files'].append(str(file_path))
                    continue

                if file_path.suffix.lower() not in self.supported_extensions:
                    logger.warning(f"Unsupported file type: {file_path}")
                    results['failed_files'].append(str(file_path))
                    continue

                # Process single file
                file_result = self._process_single_file(file_path, collection_id)

                if file_result:
                    results['successfully_processed'].append(str(file_path))
                    results['total_chunks'] += file_result['chunk_count']
                else:
                    results['failed_files'].append(str(file_path))

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results['failed_files'].append(str(file_path))

        results['processing_time'] = time.time() - start_time

        # Create result object
        upload_result = UploadResult(
            successfully_added_files=len(results['successfully_processed']),
            failed_files=len(results['failed_files']),
            total_chunks=results['total_chunks']
        )

        upload_result.metadata = {
            'upload_summary_v5_final': {
                'successfully_added_files': len(results['successfully_processed']),
                'failed_files': len(results['failed_files']),
                'total_chunks': results['total_chunks'],
                'processing_time': results['processing_time']
            }
        }

        logger.info(f"Processed {len(results['successfully_processed'])} files, "
                    f"{results['total_chunks']} chunks in {results['processing_time']:.2f}s")

        return upload_result

    def process_directory_for_context(self, directory_path: Union[str, Path],
                                      collection_id: str = GLOBAL_COLLECTION_ID,
                                      recursive: bool = True) -> Optional['UploadResult']:
        """
        Process all supported files in a directory

        Args:
            directory_path: Directory to scan
            collection_id: Collection to add documents to
            recursive: Whether to scan subdirectories

        Returns:
            UploadResult with processing summary
        """

        directory_path = Path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return None

        # Find all supported files
        file_paths = []

        if recursive:
            for ext in self.supported_extensions:
                file_paths.extend(directory_path.rglob(f"*{ext}"))
        else:
            for ext in self.supported_extensions:
                file_paths.extend(directory_path.glob(f"*{ext}"))

        # Filter out common directories to ignore
        ignore_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.env'}
        file_paths = [f for f in file_paths if not any(part in ignore_dirs for part in f.parts)]

        logger.info(f"Found {len(file_paths)} supported files in {directory_path}")

        if not file_paths:
            return UploadResult(0, 0, 0)

        return self.process_files_for_context(file_paths, collection_id)

    def _process_single_file(self, file_path: Path, collection_id: str) -> Optional[Dict[str, Any]]:
        """Process a single file"""
        try:
            # Read file content
            content = self._read_file_content(file_path)
            if not content:
                return None

            # Create file metadata
            file_metadata = {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_extension': file_path.suffix,
                'file_size': len(content),
                'file_hash': hashlib.md5(content.encode()).hexdigest()
            }

            # Chunk the content
            chunks = self.chunking_service.chunk_document(content, file_path, file_metadata)

            if not chunks:
                logger.warning(f"No chunks generated for {file_path}")
                return None

            # Generate embeddings for chunks
            documents_with_embeddings = []

            for chunk in chunks:
                try:
                    # Generate embedding
                    embedding = self.embedder.encode(chunk['content'], convert_to_tensor=False)

                    # Create document for vector DB
                    doc = {
                        'id': f"{file_metadata['file_hash']}_{chunk['id']}",
                        'content': chunk['content'],
                        'metadata': chunk['metadata'],
                        'embedding': embedding.tolist()  # Convert numpy array to list
                    }

                    documents_with_embeddings.append(doc)

                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {chunk['id']}: {e}")
                    continue

            # Add to vector database
            if documents_with_embeddings:
                self.vector_db.add_documents(collection_id, documents_with_embeddings)

                logger.info(f"Added {len(documents_with_embeddings)} chunks from {file_path.name}")

                return {
                    'file_path': str(file_path),
                    'chunk_count': len(documents_with_embeddings),
                    'file_size': len(content)
                }

            return None

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return None

    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding detection"""
        try:
            # Try UTF-8 first
            try:
                return file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Try other common encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        return file_path.read_text(encoding=encoding)
                    except UnicodeDecodeError:
                        continue

                logger.warning(f"Could not decode {file_path} - skipping")
                return None

        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return None

    def query_vector_db(self, query: str, collection_ids: List[str] = None,
                        n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar content

        Args:
            query: Search query
            collection_ids: Collections to search (default: global collection)
            n_results: Number of results to return

        Returns:
            List of similar documents
        """

        if not self.embedder_ready:
            logger.warning("Embedder not ready - cannot query")
            return []

        if not collection_ids:
            collection_ids = [GLOBAL_COLLECTION_ID]

        all_results = []

        for collection_id in collection_ids:
            try:
                results = self.vector_db.query_collection(collection_id, query, n_results)

                # Add collection info to results
                for result in results:
                    result['collection_id'] = collection_id

                all_results.extend(results)

            except Exception as e:
                logger.error(f"Failed to query collection {collection_id}: {e}")

        # Sort by distance/similarity and limit results
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        return all_results[:n_results]

    def get_collection_stats(self, collection_id: str = GLOBAL_COLLECTION_ID) -> Dict[str, Any]:
        """Get statistics about a collection"""
        return self.vector_db.get_collection_stats(collection_id)

    def delete_file_chunks(self, file_path: Union[str, Path],
                           collection_id: str = GLOBAL_COLLECTION_ID):
        """Delete all chunks for a specific file"""
        file_path = Path(file_path)

        # Calculate file hash to find chunks
        try:
            content = self._read_file_content(file_path)
            if content:
                file_hash = hashlib.md5(content.encode()).hexdigest()

                # Query for chunks with this file hash
                # This is a simplified approach - in practice you might need
                # to implement a more sophisticated chunk tracking system
                logger.info(f"Deleting chunks for {file_path.name} (hash: {file_hash})")

        except Exception as e:
            logger.error(f"Failed to delete chunks for {file_path}: {e}")


class UploadResult:
    """Result object for upload operations"""

    def __init__(self, successfully_added_files: int, failed_files: int, total_chunks: int):
        self.successfully_added_files = successfully_added_files
        self.failed_files = failed_files
        self.total_chunks = total_chunks
        self.metadata = {}

    def __str__(self):
        return (f"UploadResult(files: {self.successfully_added_files} success, "
                f"{self.failed_files} failed, chunks: {self.total_chunks})")