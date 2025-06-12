# core/rag_manager.py - Enhanced with status signals

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

try:
    from services.vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID
    from services.upload_service import UploadService

    RAG_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"RAG services not available: {e}")
    RAG_AVAILABLE = False


    class VectorDBService:
        pass


    class UploadService:
        pass


    GLOBAL_COLLECTION_ID = "dummy_collection"


class RAGManager(QObject):
    """
    RAG Manager for the AvA system with status update signals.
    """
    status_changed = Signal(str, str)
    upload_completed = Signal(str, int)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.vector_db_service: Optional[VectorDBService] = None
        self.upload_service: Optional[UploadService] = None

        self.is_ready = False
        self.current_status = "RAG: Not Initialized"

        if not RAG_AVAILABLE:
            self.logger.error("RAG services dependencies missing - RAG functionality disabled.")

    async def async_initialize(self):
        self.logger.info("RAGManager: Starting async initialization.")
        if not RAG_AVAILABLE:
            self.current_status = "RAG: Dependencies Missing"
            self.status_changed.emit(self.current_status, "error")
            return

        try:
            self.current_status = "RAG: Initializing..."
            self.status_changed.emit(self.current_status, "working")
            self.vector_db_service = VectorDBService()
            self.upload_service = UploadService()

            self.current_status = "RAG: Loading Embedder..."
            self.status_changed.emit(self.current_status, "working")

            if await self.upload_service.async_internal_init():
                if await self.upload_service.wait_for_embedder_ready():
                    self.is_ready = True
                    self.current_status = "RAG: Ready"
                    self.status_changed.emit(self.current_status, "success")
                    self.logger.info("RAGManager: RAG system fully initialized.")
                else:
                    self.is_ready = False
                    self.current_status = "RAG: Embedder Timeout"
                    self.status_changed.emit(self.current_status, "error")
            else:
                self.is_ready = False
                self.current_status = "RAG: Embedder Init Fail"
                self.status_changed.emit(self.current_status, "error")

        except Exception as e:
            self.logger.error(f"RAGManager: Initialization failed: {e}", exc_info=True)
            self.is_ready = False
            self.current_status = "RAG: Init Exception"
            self.status_changed.emit(self.current_status, "error")

    def scan_directory_dialog(self, parent_widget=None):
        if not self.is_ready:
            QMessageBox.warning(parent_widget, "RAG Not Ready", "RAG system is not ready. Please wait.")
            return

        directory = QFileDialog.getExistingDirectory(parent_widget, "Select Directory to Scan", str(Path.home()))
        if directory:
            self.status_changed.emit(f"RAG: Scanning...", "working")
            # Run the synchronous method in an async task to avoid blocking
            asyncio.create_task(self.manual_sync_directory_async(directory))

    async def manual_sync_directory_async(self, directory: str):
        """Asynchronously syncs a directory."""
        try:
            result = self.upload_service.process_directory_for_context(directory)
            if result and result.successfully_added_files > 0:
                self.upload_completed.emit(GLOBAL_COLLECTION_ID, result.successfully_added_files)
                self.status_changed.emit(f"RAG: Added {result.successfully_added_files} files", "success")
            else:
                self.status_changed.emit("RAG: No new files added", "ready")
        except Exception as e:
            self.logger.error(f"Directory scan error: {e}", exc_info=True)
            self.status_changed.emit("RAG: Scan Error", "error")

    async def query_context_async(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Asynchronous version of query_context to be called from the event loop."""
        if not self.is_ready:
            self.logger.warning("RAGManager.query_context_async called but RAG is not ready.")
            return []

        loop = asyncio.get_event_loop()
        try:
            # Run the synchronous, potentially blocking DB query in a thread pool
            return await loop.run_in_executor(
                None,  # Default executor
                self.query_context,  # The synchronous function to run
                query, k  # Arguments for the function
            )
        except Exception as e:
            self.logger.error(f"RAG async query failed: {e}", exc_info=True)
            return []

    def query_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Synchronous version for internal or non-async use."""
        if not self.is_ready:
            self.logger.warning("RAGManager.query_context called but RAG is not ready.")
            return []
        try:
            return self.upload_service.query_vector_db(query, [GLOBAL_COLLECTION_ID], n_results=k)
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}", exc_info=True)
            return []