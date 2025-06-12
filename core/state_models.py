# core/state_models.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

@dataclass
class FileState:
    """Represents the state and content of a single file."""
    path: str
    content: str
    last_modified: float
    is_new: bool = False
    is_modified: bool = False
    patterns: List[str] = field(default_factory=list)

@dataclass
class ProjectPattern:
    """Represents a detected pattern or convention in the project."""
    pattern_id: str
    description: str
    files: List[str] = field(default_factory=list)
    type: str = "unknown"  # e.g., 'testing', 'utility', 'model'

@dataclass
class AIDecision:
    """Represents a decision made by an AI agent."""
    agent: str
    decision: str
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: Optional[str] = None
    related_files: List[str] = field(default_factory=list)

@dataclass
class TeamInsight:
    """
    A structured piece of knowledge or learning contributed by an AI agent.
    This forms the collective 'memory' and 'experience' of the AI team.
    """
    insight_id: str
    timestamp: datetime
    insight_type: str  # e.g., 'architectural', 'implementation', 'testing', 'review'
    source_agent: str
    content: str
    impact_level: str  # 'high', 'medium', 'low'
    related_files: List[str] = field(default_factory=list)
    confidence: float = 1.0