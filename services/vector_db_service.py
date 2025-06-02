# services/vector_db_service.py

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Global collection ID for general code knowledge
GLOBAL_COLLECTION_ID = "ava_global_knowledge"

logger = logging.getLogger(__name__)


class VectorDBService:
    """
    ChromaDB-based vector database service for AvA RAG system
    """

    def __init__(self, index_dimension: int = 384, persist_dir: str = "./rag_data"):
        self.index_dimension = index_dimension
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        self.collections = {}

        if CHROMADB_AVAILABLE:
            self._initialize_chromadb()
        else:
            logger.error("ChromaDB not available - install with: pip install chromadb")
            raise ImportError("ChromaDB not available")

    def _initialize_chromadb(self):
        """Initialize ChromaDB client"""
        try:
            # Create persistent ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create or get the global collection
            self._ensure_collection_exists(GLOBAL_COLLECTION_ID)

            logger.info(f"ChromaDB initialized at {self.persist_dir}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _ensure_collection_exists(self, collection_id: str):
        """Ensure a collection exists"""
        try:
            if collection_id not in self.collections:
                # Try to get existing collection first
                try:
                    collection = self.client.get_collection(name=collection_id)
                    logger.info(f"Retrieved existing collection: {collection_id}")
                except:
                    # Create new collection if it doesn't exist
                    collection = self.client.create_collection(
                        name=collection_id,
                        metadata={"description": f"AvA knowledge collection: {collection_id}"}
                    )
                    logger.info(f"Created new collection: {collection_id}")

                self.collections[collection_id] = collection

        except Exception as e:
            logger.error(f"Failed to ensure collection {collection_id}: {e}")
            raise

    def add_documents(self, collection_id: str, documents: List[Dict[str, Any]]):
        """Add documents to a collection"""
        try:
            self._ensure_collection_exists(collection_id)
            collection = self.collections[collection_id]

            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            embeddings = []

            for doc in documents:
                ids.append(doc.get('id', str(len(ids))))
                texts.append(doc.get('content', ''))
                metadatas.append(doc.get('metadata', {}))

                # If embeddings are provided, use them
                if 'embedding' in doc:
                    embeddings.append(doc['embedding'])

            # Add to collection
            if embeddings:
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Let ChromaDB generate embeddings
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )

            logger.info(f"Added {len(documents)} documents to {collection_id}")

        except Exception as e:
            logger.error(f"Failed to add documents to {collection_id}: {e}")
            raise

    def query_collection(self, collection_id: str, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query a collection for similar documents"""
        try:
            self._ensure_collection_exists(collection_id)
            collection = self.collections[collection_id]

            # Query the collection
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })

            logger.debug(f"Query '{query_text[:50]}...' returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to query {collection_id}: {e}")
            return []

    def get_collection_size(self, collection_id: str) -> int:
        """Get the number of documents in a collection"""
        try:
            self._ensure_collection_exists(collection_id)
            collection = self.collections[collection_id]
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get size of {collection_id}: {e}")
            return 0

    def clear_collection(self, collection_id: str) -> bool:
        """Clear all documents from a collection"""
        try:
            if collection_id in self.collections:
                self.client.delete_collection(name=collection_id)
                del self.collections[collection_id]

                # Recreate empty collection
                self._ensure_collection_exists(collection_id)

                logger.info(f"Cleared collection: {collection_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to clear {collection_id}: {e}")
            return False

    def get_available_collections(self) -> List[str]:
        """Get list of available collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def is_ready(self, collection_id: str = None) -> bool:
        """Check if the service is ready"""
        if not self.client:
            return False

        if collection_id:
            return collection_id in self.collections

        return True

    def delete_documents(self, collection_id: str, document_ids: List[str]):
        """Delete specific documents from a collection"""
        try:
            self._ensure_collection_exists(collection_id)
            collection = self.collections[collection_id]

            collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from {collection_id}")

        except Exception as e:
            logger.error(f"Failed to delete documents from {collection_id}: {e}")

    def update_documents(self, collection_id: str, documents: List[Dict[str, Any]]):
        """Update existing documents in a collection"""
        try:
            self._ensure_collection_exists(collection_id)
            collection = self.collections[collection_id]

            # Prepare update data
            ids = []
            texts = []
            metadatas = []
            embeddings = []

            for doc in documents:
                ids.append(doc.get('id'))
                texts.append(doc.get('content', ''))
                metadatas.append(doc.get('metadata', {}))

                if 'embedding' in doc:
                    embeddings.append(doc['embedding'])

            # Update in collection
            if embeddings:
                collection.update(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                collection.update(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )

            logger.info(f"Updated {len(documents)} documents in {collection_id}")

        except Exception as e:
            logger.error(f"Failed to update documents in {collection_id}: {e}")

    def get_collection_stats(self, collection_id: str) -> Dict[str, Any]:
        """Get statistics about a collection"""
        try:
            self._ensure_collection_exists(collection_id)
            collection = self.collections[collection_id]

            return {
                'name': collection_id,
                'count': collection.count(),
                'metadata': collection.metadata if hasattr(collection, 'metadata') else {}
            }
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_id}: {e}")
            return {'name': collection_id, 'count': 0, 'metadata': {}}