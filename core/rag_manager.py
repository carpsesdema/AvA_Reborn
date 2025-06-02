# core/rag_manager.py - RAG Integration for New AvA

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

# Import your existing RAG services
try:
    from services.vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID
    from services.upload_service import UploadService
    from services.chunking_service import ChunkingService
    from services.rag_sync_service import RagSyncService

    RAG_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"RAG services not available: {e}")
    RAG_AVAILABLE = False


class RAGManager(QObject):
    """
    RAG Manager for the new AvA system - integrates your existing RAG services
    """

    # Signals for UI updates
    status_changed = Signal(str, str)  # (status_text, color)
    upload_completed = Signal(str, int)  # (collection_id, files_processed)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # RAG Services
        self.vector_db_service: Optional[VectorDBService] = None
        self.upload_service: Optional[UploadService] = None
        self.chunking_service: Optional[ChunkingService] = None
        self.rag_sync_service: Optional[RagSyncService] = None

        # State
        self.is_ready = False
        self.current_status = "RAG Not Initialized" # Default status

        if not RAG_AVAILABLE:
            self.logger.error("RAG services not available - RAG functionality disabled")
            # We can't emit here directly if the event loop isn't running yet for QObject
            # The status will be updated after async_initialize is called.

    async def async_initialize(self):
        """Asynchronously initialize RAG services. Called after event loop is running."""
        if not RAG_AVAILABLE:
            self.status_changed.emit("RAG: Dependencies Missing", "#ef4444")
            self.current_status = "RAG: Dependencies Missing"
            return

        self.current_status = "Initializing RAG..."
        self.status_changed.emit(self.current_status, "#ffb900") # Amber
        try:
            self.logger.info("Initializing RAG services...")

            # Initialize vector DB service (synchronous)
            self.vector_db_service = VectorDBService(index_dimension=384)  # MiniLM default

            # Initialize upload service (partially synchronous, with async internal init)
            self.upload_service = UploadService()
            # Now explicitly await its internal async initialization
            if not await self.upload_service.async_internal_init():
                 self.logger.error("UploadService embedder failed to initialize.")
                 self.status_changed.emit("RAG: Embedder Load Fail", "#ef4444") # Red
                 self.current_status = "RAG: Embedder Load Fail"
                 return

            # Initialize chunking service (synchronous)
            self.chunking_service = ChunkingService(chunk_size=1000, chunk_overlap=150)

            # Now that UploadService's embedder is loading/loaded, wait for it to be fully ready
            # This task is now created when the event loop is definitely running.
            asyncio.create_task(self._wait_for_embedder_ready())
            self.current_status = "RAG: Initializing embedder..."
            self.status_changed.emit(self.current_status, "#ffb900") # Amber


        except Exception as e:
            self.logger.error(f"Failed to initialize RAG services: {e}", exc_info=True)
            self.current_status = "RAG: Init Failed"
            self.status_changed.emit(self.current_status, "#ef4444") # Red

    async def _wait_for_embedder_ready(self):
        """Wait for embedder to be ready and update status"""
        try:
            if self.upload_service:
                # The wait_for_embedder_ready should ideally be on UploadService itself
                ready = await self.upload_service.wait_for_embedder_ready(timeout_seconds=60)
                if ready:
                    self.is_ready = True
                    self.current_status = "RAG: Ready"
                    self.status_changed.emit(self.current_status, "#4ade80") # Green
                    self.logger.info("RAG system fully initialized and ready")
                else:
                    self.current_status = "RAG: Embedder Timeout"
                    self.status_changed.emit(self.current_status, "#ef4444") # Red
                    self.logger.error("RAG embedder timed out.")
            else:
                self.current_status = "RAG: Service Error"
                self.status_changed.emit(self.current_status, "#ef4444") # Red
                self.logger.error("RAG Upload Service not available for waiting.")
        except Exception as e:
            self.logger.error(f"RAG _wait_for_embedder_ready error: {e}", exc_info=True)
            self.current_status = "RAG: Error"
            self.status_changed.emit(self.current_status, "#ef4444") # Red

    def scan_directory_dialog(self, parent_widget=None) -> bool:
        """Open directory selection dialog and scan for RAG"""
        if not self.is_ready:
            QMessageBox.warning(parent_widget, "RAG Not Ready",
                                "RAG system is not ready yet. Please wait for initialization.")
            return False

        directory = QFileDialog.getExistingDirectory(
            parent_widget,
            "Select Directory to Scan for RAG",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            asyncio.create_task(self._scan_directory_async(directory))
            return True
        return False

    def add_files_dialog(self, parent_widget=None) -> bool:
        """Open file selection dialog and add files to RAG"""
        if not self.is_ready:
            QMessageBox.warning(parent_widget, "RAG Not Ready",
                                "RAG system is not ready yet. Please wait for initialization.")
            return False

        files, _ = QFileDialog.getOpenFileNames(
            parent_widget,
            "Select Files to Add to RAG",
            str(Path.home()),
            "Python Files (*.py);;Text Files (*.txt *.md);;All Files (*)"
        )

        if files:
            asyncio.create_task(self._add_files_async(files))
            return True
        return False

    async def _scan_directory_async(self, directory_path: str):
        """Scan directory and add to global RAG collection"""
        try:
            self.status_changed.emit("RAG: Scanning directory...", "#61dafb")

            if self.upload_service:
                result = self.upload_service.process_directory_for_context(
                    directory_path, GLOBAL_COLLECTION_ID
                )

                if result and hasattr(result, 'metadata'):
                    metadata = getattr(result, 'metadata', {})
                    upload_summary = metadata.get('upload_summary_v5_final', {})
                    files_added = upload_summary.get('successfully_added_files', 0)

                    if files_added > 0:
                        self.status_changed.emit(f"RAG: Added {files_added} files from directory", "#4ade80")
                        self.upload_completed.emit(GLOBAL_COLLECTION_ID, files_added)
                    else:
                        self.status_changed.emit("RAG: No files added from directory", "#ffb900")
                else:
                    self.status_changed.emit("RAG: Directory scan failed", "#ef4444")

        except Exception as e:
            self.logger.error(f"Directory scan error: {e}")
            self.status_changed.emit("RAG: Scan error", "#ef4444")

    async def _add_files_async(self, file_paths: List[str]):
        """Add selected files to global RAG collection"""
        try:
            self.status_changed.emit(f"RAG: Adding {len(file_paths)} files...", "#61dafb")

            if self.upload_service:
                result = self.upload_service.process_files_for_context(
                    file_paths, GLOBAL_COLLECTION_ID
                )

                if result and hasattr(result, 'metadata'):
                    metadata = getattr(result, 'metadata', {})
                    upload_summary = metadata.get('upload_summary_v5_final', {})
                    files_added = upload_summary.get('successfully_added_files', 0)

                    if files_added > 0:
                        self.status_changed.emit(f"RAG: Added {files_added} files", "#4ade80")
                        self.upload_completed.emit(GLOBAL_COLLECTION_ID, files_added)
                    else:
                        self.status_changed.emit("RAG: No files were added", "#ffb900")
                else:
                    self.status_changed.emit("RAG: File upload failed", "#ef4444")

        except Exception as e:
            self.logger.error(f"File upload error: {e}")
            self.status_changed.emit("RAG: Upload error", "#ef4444")

    def query_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query RAG for context - used by workflow engine"""
        if not self.is_ready or not self.upload_service:
            return []

        try:
            results = self.upload_service.query_vector_db(
                query, [GLOBAL_COLLECTION_ID], n_results=k
            )

            self.logger.debug(f"RAG query '{query[:50]}...' returned {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"RAG query error: {e}")
            return []

    def get_context_for_code_generation(self, task_description: str, file_type: str = "python") -> str:
        """Get RAG context specifically for code generation"""
        if not self.is_ready:
            return ""

        # Enhanced query for code generation
        enhanced_query = f"{task_description} {file_type} code examples best practices"

        results = self.query_context(enhanced_query, k=3)

        if not results:
            return ""

        # Format context for LLM
        context_parts = []
        for i, result in enumerate(results):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            source = metadata.get('filename', 'Unknown')

            context_parts.append(f"# Example {i + 1} (from {source}):\n{content}")

        context = "\n\n".join(context_parts)

        # Truncate if too long (keep under 2000 chars for context)
        if len(context) > 2000:
            context = context[:1900] + "\n... (truncated)"

        return context

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about RAG collections"""
        if not self.vector_db_service:
            return {}

        try:
            collections = self.vector_db_service.get_available_collections()
            collection_info = {}

            for collection_id in collections:
                size = self.vector_db_service.get_collection_size(collection_id)
                collection_info[collection_id] = {
                    'size': size,
                    'ready': self.vector_db_service.is_ready(collection_id)
                }

            return collection_info

        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {}

    def clear_global_collection(self) -> bool:
        """Clear the global RAG collection"""
        if not self.vector_db_service:
            return False

        try:
            success = self.vector_db_service.clear_collection(GLOBAL_COLLECTION_ID)
            if success:
                self.status_changed.emit("RAG: Global collection cleared", "#4ade80")
            else:
                self.status_changed.emit("RAG: Failed to clear collection", "#ef4444")
            return success
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            self.status_changed.emit("RAG: Clear error", "#ef4444")
            return False