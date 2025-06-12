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
        self.context_cache: Dict[str, Any] = {}
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
        # Return from in-memory cache if available
        if "comprehensive" in self.context_cache:
            return self.context_cache["comprehensive"]

        context = {
            'database_schema': None, 'api_definition': None,
            'project_structure': None, 'frameworks': [], 'patterns': [],
        }

        tasks = {
            'db': asyncio.create_task(self.db_introspector.discover_schema(self.project_root)),
            'api': asyncio.create_task(self.api_parser.discover_api_definition(self.project_root)),
            'structure': asyncio.create_task(self.structure_analyzer.analyze_project_structure(self.project_root)),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results_map = dict(zip(tasks.keys(), results))

        schema = results_map.get('db')
        if isinstance(schema, DatabaseSchema):
            context['database_schema'] = schema.__dict__
        api_def = results_map.get('api')
        if isinstance(api_def, APIDefinition):
            context['api_definition'] = api_def.__dict__
        structure = results_map.get('structure')
        if isinstance(structure, dict):
            context['project_structure'] = structure
            context['frameworks'] = structure.get('frameworks', [])
            context['patterns'] = structure.get('patterns', [])

        self.context_cache["comprehensive"] = context
        return context

    async def get_context_for_file(self, file_path_str: str) -> Optional[Dict[str, Any]]:
        """
        Get relevant domain context for a single file. Assumes comprehensive
        context has already been generated and cached.
        """
        comprehensive_context = self.context_cache.get("comprehensive")
        if not comprehensive_context:
            # Fallback if comprehensive context wasn't generated first
            comprehensive_context = await self.get_comprehensive_context()

        file_path = Path(file_path_str)
        relevant_context = {}

        # Database context: Check if the file is an ORM model
        if db_schema := comprehensive_context.get('database_schema'):
            for orm_model in db_schema.get('orm_models', []):
                if Path(orm_model.get('file_path')) == file_path:
                    relevant_context['related_db_model'] = orm_model
                    break

        # API context: Check if the file defines API endpoints
        if api_def := comprehensive_context.get('api_definition'):
            related_endpoints = [
                ep for ep in api_def.get('endpoints', [])
                if Path(ep.get('file_path')) == file_path
            ]
            if related_endpoints:
                relevant_context['api_endpoints_defined'] = related_endpoints

        # Structure context: Provide general framework/pattern info
        relevant_context['project_frameworks'] = [fw.get('name') for fw in comprehensive_context.get('frameworks', [])]
        relevant_context['project_patterns'] = comprehensive_context.get('patterns', [])

        return relevant_context if relevant_context else None

    def _cache_context(self, cache_key: str, content_obj: Any, confidence: float):
        """Caches the analysis result to avoid re-computation."""
        # Simplified to use in-memory cache primarily for speed during a single run.
        if hasattr(content_obj, '__dict__'):
            self.context_cache[cache_key] = content_obj.__dict__
        elif isinstance(content_obj, dict):
            self.context_cache[cache_key] = content_obj
        else:
            return

    def clear_cache(self):
        """Clear all cached context."""
        self.context_cache.clear()