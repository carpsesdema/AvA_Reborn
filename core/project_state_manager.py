# core/project_state_manager.py

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re
import uuid

from .state_models import FileState, ProjectPattern, AIDecision, TeamInsight


class ProjectStateManager:
    """
    Manages the state of the project, including file contents,
    detected patterns, and team insights. V2 with enhanced context management.
    """

    def __init__(self, project_path: str):
        self.project_root = Path(project_path).resolve()
        self.files: Dict[str, FileState] = {}
        self.patterns: Dict[str, ProjectPattern] = {}
        self.team_insights: Dict[str, TeamInsight] = {}
        self.decisions: List[AIDecision] = []
        self.project_metadata: Dict[str, Any] = {"name": self.project_root.name}
        self.logger = logging.getLogger(__name__)
        self.scan_project()

    def scan_project(self, ignore_patterns: List[str] = None):
        """Scan project files, ignoring specified patterns."""
        if ignore_patterns is None:
            ignore_patterns = ['.git', '__pycache__', '.DS_Store', 'venv', '*.log']

        self.logger.info(f"Scanning project at {self.project_root}...")
        for path in self.project_root.rglob('*'):
            if path.is_file():
                relative_path = str(path.relative_to(self.project_root))

                if any(path.match(p) for p in ignore_patterns):
                    continue

                try:
                    content = path.read_text(encoding='utf-8')
                    last_mod = path.stat().st_mtime
                    self.files[relative_path] = FileState(
                        path=relative_path,
                        content=content,
                        last_modified=last_mod
                    )
                except (IOError, UnicodeDecodeError) as e:
                    self.logger.warning(f"Could not read file {relative_path}: {e}")

        self._detect_initial_patterns()
        self.logger.info(f"Scan complete. Found {len(self.files)} files.")

    def _detect_initial_patterns(self):
        """Detect initial coding patterns and conventions from scanned files."""
        patterns = {
            "uses_type_hints": False,
            "has_docstrings": False,
            "main_files": [],
            "test_framework": "unknown",
            "dependencies": set()
        }

        for file_state in self.files.values():
            if file_state.path.endswith('.py'):
                self._analyze_python_file(file_state, patterns)

        self.add_project_pattern(
            "coding_style",
            f"Type Hinting: {'Yes' if patterns['uses_type_hints'] else 'No'}, Docstrings: {'Yes' if patterns['has_docstrings'] else 'No'}",
            type="convention"
        )
        self.add_project_pattern(
            "testing",
            f"Uses {patterns['test_framework']}",
            files=self._find_test_files(patterns['test_framework']),
            type="testing"
        )
        self.add_project_pattern(
            "dependencies",
            f"Project has {len(patterns['dependencies'])} dependencies.",
            type="dependency"
        )

        self.project_metadata['established_patterns'] = patterns

    def _analyze_python_file(self, file_state: FileState, patterns: Dict):
        """Helper to analyze a single python file for patterns."""
        content = file_state.content
        if not patterns["uses_type_hints"] and re.search(r'def \w+\(.*?\)\s*->\s*\w+:', content):
            patterns["uses_type_hints"] = True
        if not patterns["has_docstrings"] and (
                re.search(r'def \w+\(.*\):\s*"""', content) or re.search(r'class \w+:\s*"""', content)):
            patterns["has_docstrings"] = True
        if "if __name__ == '__main__':" in content:
            patterns["main_files"].append(file_state.path)

        # Dependency and framework detection
        imports = re.findall(r'^(?:from|import)\s+([a-zA-Z0-9_.]+)', content, re.MULTILINE)
        for imp in imports:
            base_module = imp.split('.')[0]
            patterns["dependencies"].add(base_module)
            if base_module == "pytest":
                patterns["test_framework"] = "pytest"
            elif base_module == "unittest":
                patterns["test_framework"] = "unittest"

    def _find_test_files(self, framework: str) -> List[str]:
        """Find test files based on the detected framework."""
        if framework == "pytest":
            return [f.path for f in self.files.values() if f.path.startswith('tests/') or f.path.startswith('test_')]
        elif framework == "unittest":
            return [f.path for f in self.files.values() if f.path.startswith('test_')]
        return []

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get the content of a specific file."""
        return self.files[file_path].content if file_path in self.files else None

    def update_file(self, file_path: str, content: str):
        """Update the content of a file in the state."""
        if file_path not in self.files:
            self.files[file_path] = FileState(path=file_path, content=content, last_modified=time.time(), is_new=True)
            self.logger.info(f"File added to state: {file_path}")
        else:
            self.files[file_path].content = content
            self.files[file_path].is_modified = True
            self.files[file_path].last_modified = time.time()
            self.logger.info(f"File updated in state: {file_path}")

    def add_project_pattern(self, pattern_id: str, description: str, files: List[str] = None, type: str = "unknown"):
        """Add or update a detected project pattern."""
        self.patterns[pattern_id] = ProjectPattern(pattern_id, description, files or [], type)

    def add_team_insight(self, insight_type: str, source_agent: str, content: str,
                         impact_level: str = "medium", related_files: List[str] = None, confidence: float = 1.0):
        """Add a new insight to the team's knowledge base."""
        insight_id = str(uuid.uuid4())
        insight = TeamInsight(
            insight_id=insight_id,
            timestamp=datetime.now(),
            insight_type=insight_type,
            source_agent=source_agent,
            content=content,
            impact_level=impact_level,
            related_files=related_files or [],
            confidence=confidence
        )
        self.team_insights[insight_id] = insight
        self.logger.info(f"New insight from {source_agent}: {content[:50]}...")

    def get_full_project_context(self) -> str:
        """Get a summary of the entire project state for high-level context."""
        file_list = "\n".join([f"- {path}" for path in self.files.keys()])
        pattern_list = "\n".join([f"- {p.description}" for p in self.patterns.values()])
        insight_list = "\n".join([f"- [{i.source_agent}] {i.content}" for i in self.team_insights.values()])

        return f"""
        Project: {self.project_metadata.get('name', 'N/A')}

        Files ({len(self.files)}):
        {file_list}

        Established Patterns:
        {pattern_list}

        Recent Team Insights:
        {insight_list}
        """

    def get_enhanced_project_context(self, for_file: str = None, ai_role: str = None) -> Dict[str, Any]:
        """
        V2 CONTEXT: Get intelligent, filtered context for a specific task.
        - Filters insights based on relevance to the file and AI role.
        - Provides established project patterns.
        """
        context = {
            "project_name": self.project_metadata.get("name"),
            "established_patterns": self.project_metadata.get("established_patterns", {}),
            "team_insights": []
        }

        # Filter insights
        for insight in sorted(self.team_insights.values(), key=lambda i: i.timestamp, reverse=True):
            is_relevant = False
            # Relevance by file association
            if for_file and for_file in insight.related_files:
                is_relevant = True
            # Relevance by insight type and AI role
            if ai_role:
                if (ai_role == 'ArchitectService' and insight.insight_type == 'architectural') or \
                        (ai_role == 'CoderService' and insight.insight_type in ['implementation', 'review']) or \
                        (ai_role == 'ReviewerService' and insight.insight_type in ['implementation', 'review',
                                                                                   'testing']):
                    is_relevant = True

            # If no specific filter is met, include high-impact insights
            if insight.impact_level == 'high':
                is_relevant = True

            if is_relevant:
                context["team_insights"].append(insight.__dict__)

        # Limit to the most recent 5 relevant insights
        context["team_insights"] = context["team_insights"][:5]

        return context