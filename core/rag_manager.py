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
    from services.rag_sync_service import RagSyncService # Assuming this exists

    RAG_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"RAG services or dependencies not available: {e}", exc_info=True)
    RAG_AVAILABLE = False
    # Define dummy classes if RAG is not available to prevent further import errors
    class VectorDBService: pass
    class UploadService: pass
    class ChunkingService: pass
    class RagSyncService: pass
    GLOBAL_COLLECTION_ID = "dummy_collection"


class RAGManager(QObject):
    """
    RAG Manager for the new AvA system - integrates your existing RAG services
    """
    status_changed = Signal(str, str)  # (status_text, color_or_key)
    upload_completed = Signal(str, int)  # (collection_id, files_processed)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.vector_db_service: Optional[VectorDBService] = None
        self.upload_service: Optional[UploadService] = None
        self.chunking_service: Optional[ChunkingService] = None
        # self.rag_sync_service: Optional[RagSyncService] = None # If you use it

        self.is_ready = False
        self.current_status = "RAG: Not Initialized"

        if not RAG_AVAILABLE:
            self.logger.error("RAG services dependencies missing - RAG functionality disabled.")
            # Status will be updated in async_initialize

    async def async_initialize(self):
        """Asynchronously initialize RAG services."""
        self.logger.info("RAGManager: Starting async_initialize.")
        if not RAG_AVAILABLE:
            self.current_status = "RAG: Dependencies Missing"
            self.status_changed.emit(self.current_status, "error") # Use key for StatusIndicator
            self.logger.warning(f"RAGManager: {self.current_status}")
            self.is_ready = False
            return

        self.current_status = "RAG: Initializing Core..."
        self.status_changed.emit(self.current_status, "working") # Use key
        self.logger.info(f"RAGManager: {self.current_status}")

        try:
            # Initialize vector DB service (synchronous part)
            self.vector_db_service = VectorDBService(index_dimension=384) # Assuming MiniLM default
            self.logger.info("RAGManager: VectorDBService initialized.")

            # Initialize upload service (partially synchronous, with async internal init)
            self.upload_service = UploadService() # Synchronous instantiation
            self.logger.info("RAGManager: UploadService instantiated.")

            # Initialize chunking service (synchronous)
            self.chunking_service = ChunkingService(chunk_size=1000, chunk_overlap=150)
            self.logger.info("RAGManager: ChunkingService initialized.")

            # Now, explicitly initialize UploadService's async parts (like embedder loading)
            self.current_status = "RAG: Loading Embedder..."
            self.status_changed.emit(self.current_status, "working")
            self.logger.info(f"RAGManager: {self.current_status}")

            if not await self.upload_service.async_internal_init(): # This starts embedder loading
                 self.logger.error("RAGManager: UploadService embedder failed to START initialization.")
                 self.current_status = "RAG: Embedder Start Fail"
                 self.status_changed.emit(self.current_status, "error")
                 self.is_ready = False
                 return

            # Wait for the embedder to be fully ready
            embedder_ready_status = await self.upload_service.wait_for_embedder_ready(timeout_seconds=60)

            if embedder_ready_status:
                self.is_ready = True
                self.current_status = "RAG: Ready"
                self.status_changed.emit(self.current_status, "ready") # Use key
                self.logger.info("RAGManager: RAG system fully initialized and ready (embedder confirmed).")
            else:
                self.is_ready = False
                self.current_status = "RAG: Embedder Load Timeout/Fail"
                self.status_changed.emit(self.current_status, "error")
                self.logger.error("RAGManager: Embedder did not become ready.")

        except Exception as e:
            self.logger.error(f"RAGManager: Failed to initialize RAG services: {e}", exc_info=True)
            self.current_status = "RAG: Initialization Exception"
            self.status_changed.emit(self.current_status, "error")
            self.is_ready = False


    def scan_directory_dialog(self, parent_widget=None) -> bool:
        if not self.is_ready:
            QMessageBox.warning(parent_widget, "RAG Not Ready", "RAG system is not ready. Please wait or check logs.")
            return False
        # ... (rest of the method as before)
        directory = QFileDialog.getExistingDirectory(parent_widget, "Select Directory to Scan", str(Path.home()))
        if directory:
            self.status_changed.emit(f"RAG: Scanning {Path(directory).name}...", "working")
            # Run the blocking operation in a separate thread for GUI responsiveness
            asyncio.create_task(self._scan_directory_async_wrapper(directory))
            return True
        return False

    async def _scan_directory_async_wrapper(self, directory_path: str):
        """Wraps the synchronous scan in an async task for non-blocking GUI."""
        loop = asyncio.get_event_loop()
        try:
            # process_directory_for_context is synchronous and might block
            # Run it in an executor to avoid blocking the main asyncio loop
            result = await loop.run_in_executor(
                None, # Uses default ThreadPoolExecutor
                self.upload_service.process_directory_for_context,
                directory_path,
                GLOBAL_COLLECTION_ID
            )
            # Process result (this part is back in the asyncio thread)
            if result and hasattr(result, 'metadata'):
                metadata = getattr(result, 'metadata', {})
                upload_summary = metadata.get('upload_summary_v5_final', {})
                files_added = upload_summary.get('successfully_added_files', 0)
                if files_added > 0:
                    self.status_changed.emit(f"RAG: Added {files_added} files", "success")
                    self.upload_completed.emit(GLOBAL_COLLECTION_ID, files_added)
                else:
                    self.status_changed.emit("RAG: No new files added", "ready")
            else:
                self.status_changed.emit("RAG: Directory scan failed or no files", "warning")
        except Exception as e:
            self.logger.error(f"Async directory scan error: {e}", exc_info=True)
            self.status_changed.emit("RAG: Scan Error", "error")


    def add_files_dialog(self, parent_widget=None) -> bool:
        if not self.is_ready:
            QMessageBox.warning(parent_widget, "RAG Not Ready", "RAG system is not ready. Please wait or check logs.")
            return False
        # ... (rest of the method as before)
        files, _ = QFileDialog.getOpenFileNames(parent_widget, "Select Files to Add", str(Path.home()), "All Files (*.*)")
        if files:
            self.status_changed.emit(f"RAG: Adding {len(files)} files...", "working")
            asyncio.create_task(self._add_files_async_wrapper(files))
            return True
        return False

    async def _add_files_async_wrapper(self, file_paths: List[str]):
        """Wraps the synchronous file processing in an async task."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self.upload_service.process_files_for_context,
                file_paths,
                GLOBAL_COLLECTION_ID
            )
            if result and hasattr(result, 'metadata'):
                metadata = getattr(result, 'metadata', {})
                upload_summary = metadata.get('upload_summary_v5_final', {})
                files_added = upload_summary.get('successfully_added_files', 0)
                if files_added > 0:
                    self.status_changed.emit(f"RAG: Added {files_added} files", "success")
                    self.upload_completed.emit(GLOBAL_COLLECTION_ID, files_added)
                else:
                    self.status_changed.emit("RAG: No new files added", "ready")
            else:
                self.status_changed.emit("RAG: File add failed", "warning")
        except Exception as e:
            self.logger.error(f"Async file add error: {e}", exc_info=True)
            self.status_changed.emit("RAG: Add Files Error", "error")


    def query_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_ready or not self.upload_service:
            self.logger.warning("RAGManager.query_context: RAG not ready or upload_service missing.")
            return []
        try:
            # query_vector_db itself might be blocking if embedder.encode is blocking
            # For simplicity, assuming it's fast enough or UploadService handles threading
            results = self.upload_service.query_vector_db(query, [GLOBAL_COLLECTION_ID], n_results=k)
            self.logger.debug(f"RAG query '{query[:50]}...' returned {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"RAG query error: {e}", exc_info=True)
            return []

    def get_context_for_code_generation(self, task_description: str, file_type: str = "python") -> str:
        if not self.is_ready:
            self.logger.warning("RAGManager.get_context_for_code_generation: RAG not ready.")
            return ""
        enhanced_query = f"{task_description} {file_type} code examples best practices"
        results = self.query_context(enhanced_query, k=3)
        if not results: return ""
        context_parts = [f"# Example {i+1} (from {r.get('metadata', {}).get('filename', 'Unknown')}):\n{r.get('content', '')}" for i, r in enumerate(results)]
        context = "\n\n".join(context_parts)
        return context[:1900] + "\n... (truncated)" if len(context) > 2000 else context

    def get_collection_info(self) -> Dict[str, Any]:
        if not self.vector_db_service:
            self.logger.warning("RAGManager.get_collection_info: VectorDBService not available.")
            return {}
        try:
            collections = self.vector_db_service.get_available_collections()
            collection_info = {}
            for col_id in collections:
                size = self.vector_db_service.get_collection_size(col_id)
                # Assuming is_ready for a collection means it exists and is queryable
                collection_info[col_id] = {'size': size, 'ready': True} # Simplified
            return collection_info
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}", exc_info=True)
            return {}

    def clear_global_collection(self) -> bool:
        if not self.vector_db_service:
            self.logger.warning("RAGManager.clear_global_collection: VectorDBService not available.")
            return False
        try:
            # This operation can be slow, consider running in executor if it blocks UI
            success = self.vector_db_service.clear_collection(GLOBAL_COLLECTION_ID)
            if success:
                self.status_changed.emit("RAG: Global collection cleared", "success")
                self.is_ready = True # Re-check or assume ready after clear and re-init of collection
                self.current_status = "RAG: Ready (Collection Cleared)"
                self.status_changed.emit(self.current_status, "ready")
            else:
                self.status_changed.emit("RAG: Failed to clear collection", "warning")
            return success
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}", exc_info=True)
            self.status_changed.emit("RAG: Clear Error", "error")
            return False

