# core/domain_context_manager.py - Refactored Facade
import asyncio
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Local, refactored imports
from core.domain.models import ContextType, ContextItem, DatabaseSchema, APIDefinition
from .domain.database_analyzer import DatabaseSchemaIntrospector
from .domain.api_analyzer import APIDefinitionParser
from core.domain.structure_analyzer import ProjectStructureAnalyzer


class DomainContextManager:
    """
    🎯 Central Domain Context Facade

    Orchestrates various analyzers to discover and cache domain-specific
    context for the project, such as database schemas, API definitions,
    and architectural patterns.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cache_db_path = self.project_root / ".ava_context_cache.db"

        # Initialize individual analyzer components
        self.db_introspector = DatabaseSchemaIntrospector()
        self.api_parser = APIDefinitionParser()
        self.structure_analyzer = ProjectStructureAnalyzer()

        # In-memory and on-disk cache
        self.context_cache: Dict[str, ContextItem] = {}
        self.cache_db: Optional[sqlite3.Connection] = None
        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize SQLite cache database for persistence."""
        try:
            self.cache_db = sqlite3.connect(str(self.cache_db_path))
            self.cache_db.execute("""
                CREATE TABLE IF NOT EXISTS context_cache (
                    cache_key TEXT PRIMARY KEY,
                    context_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    confidence REAL,
                    source_hash TEXT,
                    last_updated TEXT
                )
            """)
            self.cache_db.commit()
        except Exception as e:
            print(f"Error initializing context cache: {e}")
            self.cache_db = None

    async def get_comprehensive_context(self) -> Dict[str, Any]:
        """
        Get a comprehensive domain context for the entire project by
        orchestrating all available analyzers.
        """
        context = {
            'database_schema': None,
            'api_definition': None,
            'project_structure': None,
            'frameworks': [],
            'patterns': [],
        }

        # Run analyzers concurrently for efficiency
        tasks = {
            'db': asyncio.create_task(self.db_introspector.discover_schema(self.project_root)),
            'api': asyncio.create_task(self.api_parser.discover_api_definition(self.project_root)),
            'structure': asyncio.create_task(self.structure_analyzer.analyze_project_structure(self.project_root)),
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results_map = dict(zip(tasks.keys(), results))

        # Process database schema results
        schema = results_map.get('db')
        if isinstance(schema, DatabaseSchema):
            context['database_schema'] = {
                'tables': schema.tables, 'relationships': schema.relationships, 'orm_models': schema.orm_models
            }
            self._cache_context('database_schema', schema, 0.9)
        elif isinstance(schema, Exception):
            print(f"Error discovering database schema: {schema}")

        # Process API definition results
        api_def = results_map.get('api')
        if isinstance(api_def, APIDefinition):
            context['api_definition'] = {
                'endpoints': api_def.endpoints, 'models': api_def.models, 'authentication': api_def.authentication
            }
            self._cache_context('api_definition', api_def, 0.9)
        elif isinstance(api_def, Exception):
            print(f"Error discovering API definition: {api_def}")

        # Process project structure results
        structure = results_map.get('structure')
        if isinstance(structure, dict):
            context['project_structure'] = structure
            context['frameworks'] = structure.get('frameworks', [])
            context['patterns'] = structure.get('patterns', [])
            self._cache_context('project_structure', structure, 0.8)
        elif isinstance(structure, Exception):
            print(f"Error analyzing project structure: {structure}")

        return context

    def _cache_context(self, cache_key: str, content_obj: Any, confidence: float):
        """Caches the analysis result to avoid re-computation."""
        if not self.cache_db: return
        try:
            # Convert dataclass to dict for JSON serialization
            if hasattr(content_obj, '__dict__'):
                content_dict = content_obj.__dict__
            elif isinstance(content_obj, dict):
                content_dict = content_obj
            else:
                return  # Cannot serialize

            content_json = json.dumps(content_dict, default=str)
            context_item = ContextItem(
                context_type=ContextType.PROJECT_STRUCTURE, name=cache_key, content=content_dict,
                metadata={}, confidence=confidence
            )
            self.context_cache[cache_key] = context_item

            self.cache_db.execute(
                "INSERT OR REPLACE INTO context_cache (cache_key, content, confidence, source_hash, last_updated) VALUES (?, ?, ?, ?, ?)",
                (cache_key, content_json, confidence, hashlib.md5(content_json.encode()).hexdigest(),
                 datetime.now().isoformat())
            )
            self.cache_db.commit()
        except Exception as e:
            print(f"Error caching context for '{cache_key}': {e}")

    def clear_cache(self):
        """Clear all cached context."""
        self.context_cache.clear()
        if self.cache_db:
            try:
                self.cache_db.execute("DELETE FROM context_cache")
                self.cache_db.commit()
            except Exception as e:
                print(f"Error clearing cache: {e}")