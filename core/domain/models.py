# core/domain/models.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class ContextType(Enum):
    """Enumeration for different types of discovered domain context."""
    DATABASE_SCHEMA = "database_schema"
    API_DEFINITION = "api_definition"
    PROJECT_STRUCTURE = "project_structure"
    CODE_PATTERNS = "code_patterns"
    CONFIGURATION = "configuration"
    DEPENDENCIES = "dependencies"
    TESTS = "tests"


@dataclass
class ContextItem:
    """Represents a single piece of discovered context."""
    context_type: ContextType
    name: str
    content: Any
    metadata: Dict[str, Any]
    confidence: float
    source_file: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DatabaseSchema:
    """Represents a discovered database schema, including ORM models."""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    orm_models: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class APIDefinition:
    """Represents a discovered API structure from code or spec files."""
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    models: List[Dict[str, Any]] = field(default_factory=list)
    authentication: Dict[str, Any] = field(default_factory=dict)
    middleware: List[str] = field(default_factory=list)
    base_url: Optional[str] = None