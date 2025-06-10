# core/project_state_manager.py - Enhanced with Team Communication

import hashlib
import json
import re
import ast
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FileState:
    path: str
    content: str
    hash: str
    dependencies: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    last_modified: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.8
    ai_decisions: List[Dict[str, Any]] = field(default_factory=list)
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_modified'] = self.last_modified.isoformat() if isinstance(self.last_modified, datetime) else str(
            self.last_modified)
        return data


@dataclass
class ProjectPattern:
    pattern_id: str
    description: str
    examples: List[str]
    confidence: float
    usage_count: int
    pattern_type: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIDecision:
    """Represents a decision made by an AI specialist"""
    ai_role: str  # planner, coder, assembler, reviewer
    decision_type: str  # architecture, implementation, naming, etc.
    context: str
    reasoning: str
    alternatives_considered: List[str] = field(default_factory=list)
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=datetime.now)
    file_affected: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp)
        return data


@dataclass
class TeamInsight:
    """Represents insights and knowledge accumulated by the AI team"""
    insight_type: str  # architectural, implementation, quality, integration
    source_agent: str  # planner, coder, assembler, reviewer
    content: str
    impact_level: str  # critical, high, medium, low
    related_files: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    applies_to_future: bool = True  # Should this guide future work?

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp)
        return data


class ProjectStateManager:
    """
    🧠 Enhanced Central Intelligence for Project-Aware AI Collaboration

    This manager maintains complete awareness of:
    - Project structure and file relationships
    - Coding patterns and conventions established
    - AI decisions and reasoning chains
    - User feedback and preferences
    - Quality metrics and improvement opportunities
    - **NEW: Team insights and collaborative knowledge**
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.files: Dict[str, FileState] = {}
        self.patterns: Dict[str, ProjectPattern] = {}
        self.ai_decisions: List[AIDecision] = []
        self.user_preferences: Dict[str, Any] = {}

        # NEW: Team insight storage
        self.team_insights: List[TeamInsight] = []
        self.team_context_cache: Dict[str, Any] = {}
        self.last_context_update: datetime = datetime.now()

        self.project_metadata = {
            "name": self.project_root.name,
            "created": datetime.now(),
            "last_updated": datetime.now(),
            "total_files": 0,
            "ai_iterations": 0
        }

        self._scan_existing_project()
        self._detect_initial_patterns()
        self.load_state()

    # NEW: Team insight methods
    def add_team_insight(self, insight_type: str, source_agent: str, content: str,
                         impact_level: str = "medium", related_files: List[str] = None,
                         applies_to_future: bool = True):
        """Add insight from an AI agent to the team knowledge base."""
        insight = TeamInsight(
            insight_type=insight_type,
            source_agent=source_agent,
            content=content,
            impact_level=impact_level,
            related_files=related_files or [],
            applies_to_future=applies_to_future
        )
        self.team_insights.append(insight)
        self._invalidate_context_cache()
        self.save_state()

    def get_team_insights_by_type(self, insight_type: str) -> List[TeamInsight]:
        """Get all team insights of a specific type."""
        return [i for i in self.team_insights if i.insight_type == insight_type]

    def get_team_insights_by_agent(self, agent: str) -> List[TeamInsight]:
        """Get all insights contributed by a specific agent."""
        return [i for i in self.team_insights if i.source_agent == agent]

    def get_applicable_insights(self, file_path: str = None) -> List[TeamInsight]:
        """Get insights that apply to current work, optionally filtered by file."""
        applicable = [i for i in self.team_insights if i.applies_to_future]
        if file_path:
            rel_path = self._get_relative_path(file_path)
            applicable = [i for i in applicable if not i.related_files or rel_path in i.related_files]
        return sorted(applicable, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x.impact_level])

    def _invalidate_context_cache(self):
        """Invalidate the context cache when insights change."""
        self.team_context_cache = {}
        self.last_context_update = datetime.now()

    def get_enhanced_project_context(self, for_file: str = None, ai_role: str = None) -> Dict[str, Any]:
        """Enhanced version of get_project_context with team insights."""
        # Check if we have a cached version that's still fresh
        cache_key = f"{for_file}_{ai_role}"
        if (cache_key in self.team_context_cache and
                (datetime.now() - self.last_context_update).seconds < 300):  # 5 minute cache
            return self.team_context_cache[cache_key]

        # Get base context (existing functionality)
        context = self.get_project_context(for_file, ai_role)

        # Add team insights
        context["team_insights"] = {
            "architectural_insights": [i.to_dict() for i in self.get_team_insights_by_type("architectural")],
            "implementation_patterns": [i.to_dict() for i in self.get_team_insights_by_type("implementation")],
            "quality_standards": [i.to_dict() for i in self.get_team_insights_by_type("quality")],
            "integration_learnings": [i.to_dict() for i in self.get_team_insights_by_type("integration")],
            "applicable_to_current": [i.to_dict() for i in self.get_applicable_insights(for_file)]
        }

        # Add team communication summary
        context["team_communication"] = self._build_team_communication_summary()

        # Cache the result
        self.team_context_cache[cache_key] = context
        return context

    def _build_team_communication_summary(self) -> Dict[str, Any]:
        """Build a summary of team communication patterns and priorities."""
        recent_insights = [i for i in self.team_insights if
                           (datetime.now() - i.timestamp).days < 7]  # Last week

        return {
            "recent_priorities": [i.content for i in recent_insights if i.impact_level in ["critical", "high"]],
            "established_patterns": [i.content for i in self.team_insights if i.insight_type == "implementation"],
            "quality_focus_areas": [i.content for i in recent_insights if i.insight_type == "quality"],
            "architectural_decisions": [i.content for i in self.team_insights if i.insight_type == "architectural"],
            "lessons_learned": [i.content for i in recent_insights if "lesson" in i.content.lower()]
        }

    # All existing methods remain unchanged - just adding new functionality
    def add_file(self, file_path: str, content: str, ai_role: str = None,
                 reasoning: str = "") -> FileState:
        rel_path = self._get_relative_path(file_path)
        file_hash = hashlib.md5(content.encode('utf-8', 'ignore')).hexdigest()
        dependencies = self._extract_dependencies(content)
        exports = self._extract_exports(content)
        imports = self._extract_imports(content)
        current_time = datetime.now()

        file_state = FileState(
            path=rel_path, content=content, hash=file_hash,
            dependencies=dependencies, exports=exports, imports=imports,
            last_modified=current_time, ai_decisions=[], user_feedback=[]
        )

        if ai_role and reasoning:
            decision = AIDecision(
                ai_role=ai_role, decision_type="file_creation_or_update",
                context=f"File {rel_path} created/updated.",
                reasoning=reasoning,
                timestamp=current_time, file_affected=rel_path
            )
            file_state.ai_decisions.append(decision.to_dict())
            self.ai_decisions.append(decision)

        self.files[rel_path] = file_state
        self._update_patterns()
        self._update_metadata()
        self._invalidate_context_cache()  # NEW: Invalidate cache when files change
        return file_state

    def get_project_context(self, for_file: str = None, ai_role: str = None) -> Dict[str, Any]:
        project_overview_serializable = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in self.project_metadata.items()
        }
        project_overview_serializable.setdefault("main_files", self._identify_main_files())
        project_overview_serializable.setdefault("architecture_type", self._detect_architecture_type())

        context = {
            "project_overview": project_overview_serializable,
            "established_patterns": {
                "naming_conventions": self._get_naming_patterns(),
                "code_structure": self._get_structure_patterns(),
                "import_patterns": self._get_import_patterns(),
                "architectural_decisions": self._get_architectural_decisions(),
            },
            "file_relationships": self._build_dependency_graph(),
            "coding_standards": self._extract_coding_standards(),
            "recent_decisions": self._get_recent_ai_decisions(limit=5),
            "user_preferences": self.user_preferences.copy()
        }

        if for_file:
            context["target_file_context"] = self._get_file_specific_context(for_file)
        if ai_role:
            context["role_specific_guidance"] = self._get_role_guidance(ai_role)
        return context

    def record_ai_decision(self, ai_role: str, decision_type: str, context: str,
                           reasoning: str, confidence: float = 0.8,
                           alternatives: List[str] = None, file_affected: str = None):
        decision = AIDecision(
            ai_role=ai_role, decision_type=decision_type, context=context,
            reasoning=reasoning, alternatives_considered=alternatives or [],
            confidence=confidence, file_affected=file_affected
        )
        self.ai_decisions.append(decision)
        if file_affected and file_affected in self.files:
            self.files[file_affected].ai_decisions.append(decision.to_dict())
        self._invalidate_context_cache()  # NEW: Invalidate cache when decisions change

    def save_state(self):
        """Save project state including team insights."""
        state_file = self.project_root / ".ava_state.json"
        try:
            state_data = {
                "metadata": {k: (v.isoformat() if isinstance(v, datetime) else v)
                             for k, v in self.project_metadata.items()},
                "files": {path: fs.to_dict() for path, fs in self.files.items()},
                "patterns": {pid: p.to_dict() for pid, p in self.patterns.items()},
                "ai_decisions": [d.to_dict() for d in self.ai_decisions],
                "user_preferences": self.user_preferences,
                "team_insights": [i.to_dict() for i in self.team_insights]  # NEW
            }
            state_file.write_text(json.dumps(state_data, indent=2))
        except Exception as e:
            print(f"Failed to save project state: {e}")

    def load_state(self):
        """Load project state including team insights."""
        state_file = self.project_root / ".ava_state.json"
        if not state_file.exists():
            return

        try:
            state_data = json.loads(state_file.read_text())

            # Load existing data
            if "metadata" in state_data:
                for key, value in state_data["metadata"].items():
                    if key in ["created", "last_updated"] and isinstance(value, str):
                        try:
                            self.project_metadata[key] = datetime.fromisoformat(value)
                        except:
                            self.project_metadata[key] = datetime.now()
                    else:
                        self.project_metadata[key] = value

            if "user_preferences" in state_data:
                self.user_preferences = state_data["user_preferences"]

            # NEW: Load team insights
            if "team_insights" in state_data:
                self.team_insights = []
                for insight_data in state_data["team_insights"]:
                    try:
                        # Convert timestamp back to datetime
                        if "timestamp" in insight_data and isinstance(insight_data["timestamp"], str):
                            insight_data["timestamp"] = datetime.fromisoformat(insight_data["timestamp"])

                        insight = TeamInsight(
                            insight_type=insight_data.get("insight_type", "unknown"),
                            source_agent=insight_data.get("source_agent", "unknown"),
                            content=insight_data.get("content", ""),
                            impact_level=insight_data.get("impact_level", "medium"),
                            related_files=insight_data.get("related_files", []),
                            timestamp=insight_data.get("timestamp", datetime.now()),
                            applies_to_future=insight_data.get("applies_to_future", True)
                        )
                        self.team_insights.append(insight)
                    except Exception as e:
                        print(f"Failed to load team insight: {e}")

        except Exception as e:
            print(f"Failed to load project state: {e}")

    # All existing private methods remain exactly the same
    def _get_relative_path(self, file_path: str) -> str:
        path_obj = Path(file_path)
        if path_obj.is_absolute():
            try:
                return str(path_obj.relative_to(self.project_root))
            except ValueError:
                return path_obj.name
        return str(path_obj)

    def _extract_dependencies(self, content: str) -> List[str]:
        dependencies = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('from ') and 'import' in line:
                parts = line.split()
                if len(parts) >= 2 and not parts[1].startswith('.'):
                    dependencies.append(parts[1])
            elif line.startswith('import '):
                module = line.replace('import ', '').split()[0].split('.')[0]
                if module != '__future__':
                    dependencies.append(module)
        return list(set(dependencies))

    def _extract_exports(self, content: str) -> List[str]:
        exports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not node.name.startswith('_'):
                        exports.append(node.name)
        except:
            pass
        return exports

    def _extract_imports(self, content: str) -> List[str]:
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                imports.append(line)
        return imports

    def _scan_existing_project(self):
        if not self.project_root.exists() or not self.project_root.is_dir():
            print(f"Project root {self.project_root} does not exist or is not a directory.")
            return

        for py_file in self.project_root.rglob("*.py"):
            try:
                rel_path = str(py_file.relative_to(self.project_root))
                if not any(part.startswith('.') for part in py_file.parts):
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    self.add_file(str(py_file), content)
            except Exception as e:
                print(f"Failed to scan file {py_file}: {e}")

    def _detect_initial_patterns(self):
        if not self.files:
            return

        naming_examples = []
        for file_state in self.files.values():
            try:
                tree = ast.parse(file_state.content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        naming_examples.append(node.name)
            except:
                continue

        if naming_examples and all('_' in name for name in naming_examples[:5]):
            self.patterns["function_naming"] = ProjectPattern(
                "function_naming", "snake_case", naming_examples[:3], 0.9, len(naming_examples)
            )

    def _update_patterns(self):
        pass

    def _update_metadata(self):
        self.project_metadata["total_files"] = len(self.files)
        self.project_metadata["last_updated"] = datetime.now()
        self.project_metadata["ai_iterations"] = len(self.ai_decisions)

    def get_consistency_requirements(self, file_path: str) -> Dict[str, Any]:
        return {
            "naming_style": self._get_consistent_naming_style(),
            "import_organization": self._get_import_style(),
            "documentation_standards": self._get_doc_standards(),
            "error_handling_patterns": self._get_error_patterns()
        }

    def _get_interface_requirements(self, file_path: str) -> Dict[str, Any]:
        if file_path in self.files:
            return {"exports": self.files[file_path].exports,
                    "description": "Expected public API based on current exports."}
        return {"description": "No specific interface requirements defined yet for non-existent or untracked file."}

    def validate_file_consistency(self, file_path: str, content: str) -> Dict[str, Any]:
        issues = []
        suggestions = []
        issues.extend(self._check_naming_consistency(content))
        issues.extend(self._check_import_consistency(content))
        issues.extend(self._check_architectural_consistency(file_path, content))
        suggestions = self._generate_consistency_suggestions(file_path, issues)
        return {"is_consistent": not issues, "issues": issues, "suggestions": suggestions,
                "consistency_score": max(0, 1.0 - (len(issues) * 0.1))}

    def get_next_file_suggestions(self, current_files: List[str]) -> List[Dict[str, Any]]:
        suggestions = []
        missing_deps = self._find_missing_dependencies()
        for dep in missing_deps:
            suggestions.append({"file_path": f"{dep}.py", "reason": "Required by existing imports",
                                "priority": "high", "suggested_content": self._suggest_file_structure(dep)})
        suggestions.extend(self._suggest_architectural_files())
        priority_map = {"high": 0, "medium": 1, "low": 2}
        return sorted(suggestions, key=lambda x: priority_map.get(x.get("priority", "low").lower(), 99))

    def add_user_feedback(self, file_path: str, feedback_type: str, content: str, rating: int):
        rel_path = self._get_relative_path(file_path) if file_path else None
        feedback_timestamp = datetime.now()
        feedback = {"type": feedback_type, "content": content, "rating": rating,
                    "timestamp": feedback_timestamp.isoformat()}
        if rel_path and rel_path in self.files:
            self.files[rel_path].user_feedback.append(feedback)
        self._update_user_preferences(feedback_type, content, rating, feedback_timestamp)

    def get_improvement_opportunities(self) -> List[Dict[str, Any]]:
        opportunities = []
        low_quality_files = [f for f in self.files.values() if f.quality_score < 0.7]
        for file_state in low_quality_files:
            opportunities.append({"type": "quality_improvement", "file": file_state.path,
                                  "current_score": file_state.quality_score,
                                  "suggestions": self._get_quality_improvement_suggestions(file_state)})
        opportunities.extend(self._find_project_inconsistencies())
        return opportunities

    def _find_project_inconsistencies(self) -> List[Dict[str, Any]]:
        return []

    def _get_naming_patterns(self) -> Dict[str, Any]:
        return {pid: p.to_dict() for pid, p in self.patterns.items() if p.pattern_type == "naming"}

    def _get_structure_patterns(self) -> Dict[str, Any]:
        return {pid: p.to_dict() for pid, p in self.patterns.items() if p.pattern_type == "structure"}

    def _get_import_patterns(self) -> Dict[str, Any]:
        return {pid: p.to_dict() for pid, p in self.patterns.items() if p.pattern_type == "imports"}

    def _get_architectural_decisions(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self.ai_decisions if d.decision_type == "architecture"]

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        return {p: f_state.dependencies for p, f_state in self.files.items()}

    def _extract_coding_standards(self) -> Dict[str, Any]:
        return {"docstring_style": "google", "type_hints": True, "error_handling": "exceptions",
                "logging_style": "standard"}

    def _get_recent_ai_decisions(self, limit: int = 5) -> List[Dict[str, Any]]:
        sorted_decisions = sorted(self.ai_decisions, key=lambda d: d.timestamp, reverse=True)
        return [d.to_dict() for d in sorted_decisions[:limit]]

    def _get_file_specific_context(self, file_path: str) -> Dict[str, Any]:
        rel_path = self._get_relative_path(file_path)
        context = {"related_files": self._get_related_files(rel_path),
                   "consistency_requirements": self.get_consistency_requirements(file_path),
                   "required_interfaces": self._get_interface_requirements(rel_path)}
        if rel_path in self.files:
            fs = self.files[rel_path]
            context.update({"current_exports": fs.exports, "current_dependencies": fs.dependencies,
                            "previous_decisions": [d for d in fs.ai_decisions[-3:]],
                            "user_feedback": [f for f in fs.user_feedback[-2:]]})
        return context

    def _get_role_guidance(self, ai_role: str) -> Dict[str, Any]:
        guidance = {
            "planner": {"focus": "architecture, file organization, interface contracts",
                        "deliverables": ["project structure", "file specs"]},
            "coder": {"focus": "clean code, algorithm efficiency, error handling",
                      "deliverables": ["functional code snippets", "unit tests"]},
            "assembler": {"focus": "integration, consistency, final file structure",
                          "deliverables": ["complete files", "import resolution"]},
            "reviewer": {"focus": "quality, adherence to standards, best practices",
                         "deliverables": ["review feedback", "approval status"]}
        }
        return guidance.get(ai_role.lower(), {})

    def _get_related_files(self, file_path: str) -> List[str]:
        related = set()
        current_file = self.files.get(file_path)
        if not current_file: return []
        for path, f_state in self.files.items():
            if path == file_path: continue
            if current_file.exports and any(exp in f_state.imports for exp in current_file.exports): related.add(path)
        related.update(current_file.dependencies)
        return list(related - {file_path})

    def _get_consistent_naming_style(self) -> str:
        return self.patterns.get("function_naming", ProjectPattern("", "snake_case", [], 0, 0)).description

    def _get_import_style(self) -> str:
        return "standard library first, then third-party, then local"

    def _get_doc_standards(self) -> str:
        return "Google-style docstrings for public API"

    def _get_error_patterns(self) -> str:
        return "Use specific exceptions, log errors"

    def _check_naming_consistency(self, content: str) -> List[str]:
        return []

    def _check_import_consistency(self, content: str) -> List[str]:
        return []

    def _check_architectural_consistency(self, file_path: str, content: str) -> List[str]:
        return []

    def _generate_consistency_suggestions(self, file_path: str, issues: List[str]) -> List[str]:
        return ["Review naming against project patterns."]

    def _find_missing_dependencies(self) -> List[str]:
        return []

    def _suggest_architectural_files(self) -> List[Dict[str, Any]]:
        return []

    def _suggest_file_structure(self, module_name: str) -> str:
        return f'"""Module for {module_name}."""\n\npass\n'

    def _update_user_preferences(self, feedback_type: str, content: str, rating: int, timestamp: datetime):
        if feedback_type not in self.user_preferences: self.user_preferences[feedback_type] = []
        self.user_preferences[feedback_type].append(
            {"content": content, "rating": rating, "timestamp": timestamp.isoformat()})

    def _get_quality_improvement_suggestions(self, file_state: FileState) -> List[str]:
        return ["Refactor complex functions."]

    def _identify_main_files(self) -> List[str]:
        return [path for path, f_state in self.files.items() if 'if __name__ == "__main__":' in f_state.content]

    def _detect_architecture_type(self) -> str:
        counts = defaultdict(int)
        patterns = {"gui_application": ["gui", "ui", "view", "window", "pyside", "tkinter"],
                    "web_application": ["app", "api", "route", "view", "server", "flask", "django"],
                    "cli_application": ["cli", "main", "command"], "library": ["lib", "core", "util"]}
        for path in self.files.keys():
            for arch, keywords in patterns.items():
                if any(k in path.lower() for k in keywords): counts[arch] += 1
        return max(counts, key=counts.get) if counts else "library"