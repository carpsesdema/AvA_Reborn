# services/rag_sync_service.py

import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Callable
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

from .upload_service import UploadService
from .vector_db_service import GLOBAL_COLLECTION_ID

logger = logging.getLogger(__name__)


class RagFileWatcher(FileSystemEventHandler):
    """File system event handler for RAG sync"""

    def __init__(self, sync_service: 'RagSyncService'):
        super().__init__()
        self.sync_service = sync_service
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.txt', '.md', '.rst', '.html', '.css', '.json', '.yaml', '.yml'
        }

    def on_modified(self, event):
        if not event.is_directory:
            self.sync_service.queue_file_update(event.src_path, 'modified')

    def on_created(self, event):
        if not event.is_directory:
            self.sync_service.queue_file_update(event.src_path, 'created')

    def on_deleted(self, event):
        if not event.is_directory:
            self.sync_service.queue_file_update(event.src_path, 'deleted')

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported"""
        return Path(file_path).suffix.lower() in self.supported_extensions


class RagSyncService:
    """
    Real-time file synchronization service for AvA RAG system
    Watches directories for changes and updates the vector database automatically
    """

    def __init__(self, upload_service: UploadService = None):
        self.upload_service = upload_service or UploadService()

        # File watching
        self.observer = None
        self.watched_directories: Dict[str, str] = {}  # path -> collection_id
        self.update_queue = asyncio.Queue()
        self.is_running = False

        # Debouncing to avoid too frequent updates
        self.debounce_delay = 2.0  # seconds
        self.pending_updates: Dict[str, Dict] = {}  # file_path -> {timestamp, action}

        # Callbacks for status updates
        self.status_callbacks: List[Callable] = []

        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog not available - install with: pip install watchdog")

    def add_status_callback(self, callback: Callable[[str, str], None]):
        """Add callback for status updates"""
        self.status_callbacks.append(callback)

    def _notify_status(self, message: str, level: str = "info"):
        """Notify status callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(message, level)
            except Exception as e:
                logger.error(f"Status callback failed: {e}")

    async def start_watching(self, directory_paths: List[str],
                             collection_id: str = GLOBAL_COLLECTION_ID):
        """
        Start watching directories for file changes

        Args:
            directory_paths: List of directories to watch
            collection_id: Collection to sync files to
        """

        if not WATCHDOG_AVAILABLE:
            logger.error("Cannot start watching - watchdog not available")
            return False

        try:
            self.observer = Observer()
            event_handler = RagFileWatcher(self)

            for directory_path in directory_paths:
                path = Path(directory_path)

                if not path.exists() or not path.is_dir():
                    logger.warning(f"Directory not found: {directory_path}")
                    continue

                self.observer.schedule(event_handler, str(path), recursive=True)
                self.watched_directories[str(path)] = collection_id

                logger.info(f"Started watching: {directory_path}")
                self._notify_status(f"Watching: {path.name}", "info")

            self.observer.start()
            self.is_running = True

            # Start update processing task
            asyncio.create_task(self._process_updates())

            return True

        except Exception as e:
            logger.error(f"Failed to start watching: {e}")
            self._notify_status(f"Watch failed: {e}", "error")
            return False

    def stop_watching(self):
        """Stop watching all directories"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None

        self.is_running = False
        self.watched_directories.clear()

        logger.info("Stopped watching directories")
        self._notify_status("Stopped watching", "info")

    def queue_file_update(self, file_path: str, action: str):
        """Queue a file update with debouncing"""
        if not self._is_supported_file(file_path):
            return

        current_time = time.time()

        # Update pending updates with debouncing
        self.pending_updates[file_path] = {
            'timestamp': current_time,
            'action': action,
            'file_path': file_path
        }

        logger.debug(f"Queued {action} for {Path(file_path).name}")

    async def _process_updates(self):
        """Process queued file updates with debouncing"""
        while self.is_running:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms

                current_time = time.time()
                ready_updates = []

                # Find updates that are ready (past debounce delay)
                for file_path, update_info in list(self.pending_updates.items()):
                    if current_time - update_info['timestamp'] >= self.debounce_delay:
                        ready_updates.append(update_info)
                        del self.pending_updates[file_path]

                # Process ready updates
                for update_info in ready_updates:
                    await self._process_single_update(update_info)

            except Exception as e:
                logger.error(f"Error processing updates: {e}")

    async def _process_single_update(self, update_info: Dict):
        """Process a single file update"""
        file_path = update_info['file_path']
        action = update_info['action']

        try:
            # Find which collection this file belongs to
            collection_id = self._find_collection_for_file(file_path)
            if not collection_id:
                return

            path = Path(file_path)

            if action in ['created', 'modified']:
                if path.exists():
                    # Wait for embedder to be ready
                    if not await self.upload_service.wait_for_embedder_ready(timeout_seconds=5):
                        logger.warning("Embedder not ready - skipping update")
                        return

                    # Process the file
                    result = self.upload_service.process_files_for_context([file_path], collection_id)

                    if result and result.successfully_added_files > 0:
                        logger.info(f"Synced {action}: {path.name}")
                        self._notify_status(f"Synced: {path.name}", "success")
                    else:
                        logger.warning(f"Failed to sync: {path.name}")
                        self._notify_status(f"Sync failed: {path.name}", "warning")

            elif action == 'deleted':
                # Handle file deletion
                self.upload_service.delete_file_chunks(file_path, collection_id)
                logger.info(f"Removed from RAG: {path.name}")
                self._notify_status(f"Removed: {path.name}", "info")

        except Exception as e:
            logger.error(f"Failed to process {action} for {file_path}: {e}")
            self._notify_status(f"Error: {Path(file_path).name}", "error")

    def _find_collection_for_file(self, file_path: str) -> Optional[str]:
        """Find which collection a file belongs to based on watched directories"""
        file_path = Path(file_path)

        # Find the most specific watched directory
        best_match = None
        best_match_parts = 0

        for watched_dir, collection_id in self.watched_directories.items():
            watched_path = Path(watched_dir)

            try:
                # Check if file is within this watched directory
                file_path.relative_to(watched_path)

                # Use the most specific match (most path parts)
                parts = len(watched_path.parts)
                if parts > best_match_parts:
                    best_match = collection_id
                    best_match_parts = parts

            except ValueError:
                # File is not within this directory
                continue

        return best_match

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported"""
        supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.txt', '.md', '.rst', '.html', '.css', '.json', '.yaml', '.yml'
        }

        return Path(file_path).suffix.lower() in supported_extensions

    def get_watched_directories(self) -> List[str]:
        """Get list of currently watched directories"""
        return list(self.watched_directories.keys())

    def get_sync_stats(self) -> Dict[str, any]:
        """Get synchronization statistics"""
        return {
            'is_running': self.is_running,
            'watched_directories': len(self.watched_directories),
            'pending_updates': len(self.pending_updates),
            'directories': list(self.watched_directories.keys())
        }

    async def manual_sync_directory(self, directory_path: str,
                                    collection_id: str = GLOBAL_COLLECTION_ID) -> bool:
        """Manually trigger a full sync of a directory"""
        try:
            self._notify_status(f"Manual sync started: {Path(directory_path).name}", "info")

            # Wait for embedder
            if not await self.upload_service.wait_for_embedder_ready(timeout_seconds=10):
                self._notify_status("Embedder not ready", "error")
                return False

            # Process directory
            result = self.upload_service.process_directory_for_context(directory_path, collection_id)

            if result and result.successfully_added_files > 0:
                self._notify_status(
                    f"Synced {result.successfully_added_files} files",
                    "success"
                )
                return True
            else:
                self._notify_status("No files synced", "warning")
                return False

        except Exception as e:
            logger.error(f"Manual sync failed: {e}")
            self._notify_status(f"Sync error: {e}", "error")
            return False

    def __del__(self):
        """Cleanup when service is destroyed"""
        if self.is_running:
            self.stop_watching()