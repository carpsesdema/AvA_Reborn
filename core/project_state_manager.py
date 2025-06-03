# core/project_state_manager.py - Central Project Intelligence & State Management

import json
import hashlib
from datetime import datetime, date  # Import date for type checking
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict, field  # Import field
from collections import defaultdict


# Helper for JSON serialization of datetime
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


@dataclass
class FileState:
    """Represents the state of a single file in the project"""
    path: str
    content: str
    hash: str
    dependencies: List[str]
    exports: List[str]  # Functions, classes, constants this file provides
    imports: List[str]  # What this file imports
    last_modified: datetime
    ai_decisions: List[Dict[str, Any]] = field(default_factory=list)
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    review_status: str = "pending"  # pending, approved, needs_revision

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_modified'] = self.last_modified.isoformat() if isinstance(self.last_modified, datetime) else str(
            self.last_modified)

        for decision_list_key in ['ai_decisions', 'user_feedback']:
            if decision_list_key in data:
                for item in data[decision_list_key]:
                    if 'timestamp' in item and isinstance(item['timestamp'], datetime):
                        item['timestamp'] = item['timestamp'].isoformat()
                    elif 'timestamp' in item and not isinstance(item['timestamp'],
                                                                str):  # Ensure it's a string if not datetime
                        item['timestamp'] = str(item['timestamp'])
        return data


@dataclass
class ProjectPattern:
    """Represents a detected or established pattern in the project"""
    pattern_type: str  # naming, architecture, coding_style, etc.
    description: str
    examples: List[str]
    confidence: float
    usage_count: int

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


class ProjectStateManager:
    """
    🧠 Central Intelligence for Project-Aware AI Collaboration

    This manager maintains complete awareness of:
    - Project structure and file relationships
    - Coding patterns and conventions established
    - AI decisions and reasoning chains
    - User feedback and preferences
    - Quality metrics and improvement opportunities
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()  # Resolve path for consistency
        self.files: Dict[str, FileState] = {}
        self.patterns: Dict[str, ProjectPattern] = {}
        self.ai_decisions: List[AIDecision] = []
        self.user_preferences: Dict[str, Any] = {}
        self.project_metadata = {
            "name": self.project_root.name,
            "created": datetime.now(),
            "last_updated": datetime.now(),
            "total_files": 0,
            "ai_iterations": 0
        }

        self._scan_existing_project()
        self._detect_initial_patterns()
        self.load_state()  # Attempt to load state after init

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
                context=f"File {rel_path} created/updated.", reasoning=reasoning,
                timestamp=current_time, file_affected=rel_path
            )
            file_state.ai_decisions.append(decision.to_dict())
            self.ai_decisions.append(decision)

        self.files[rel_path] = file_state
        self._update_patterns()
        self._update_metadata()
        return file_state

    def get_project_context(self, for_file: str = None, ai_role: str = None) -> Dict[str, Any]:
        project_overview_serializable = {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in self.project_metadata.items()
        }
        # Ensure main_files and architecture_type are present even if calculated later
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
            confidence=confidence, timestamp=datetime.now(), file_affected=file_affected
        )
        self.ai_decisions.append(decision)
        if file_affected:
            rel_path = self._get_relative_path(file_affected)
            if rel_path in self.files:
                self.files[rel_path].ai_decisions.append(decision.to_dict())

    def get_consistency_requirements(self, file_path: str) -> Dict[str, Any]:
        rel_path = self._get_relative_path(file_path)
        return {
            "naming_style": self._get_consistent_naming_style(),
            "import_organization": self._get_import_style(),
            "documentation_level": self._get_doc_standards(),
            "error_handling_patterns": self._get_error_patterns(),
            "related_files": self._get_related_files(rel_path),
            "interface_contracts": self._get_interface_requirements(rel_path)
        }

    def _get_interface_requirements(self, file_path: str) -> Dict[str, Any]:
        """
        Placeholder or actual implementation for getting interface requirements.
        This method was missing, causing an AttributeError.
        """
        # print(f"DEBUG: _get_interface_requirements called for {file_path}") # Optional debug
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
        rel_path = self._get_relative_path(file_path) if file_path else None  # Handle optional file_path
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

    def _scan_existing_project(self):
        if not self.project_root.exists() or not self.project_root.is_dir():
            print(f"Project root {self.project_root} does not exist or is not a directory. Skipping scan.")
            return
        for item in self.project_root.rglob("*"):
            if self._should_ignore_file(item): continue
            if item.is_file():
                try:
                    content = item.read_text(encoding='utf-8', errors='ignore')
                    self.add_file(str(item), content)  # add_file handles relative path
                except Exception as e:
                    print(f"Warning: Could not read/process {item}: {e}")

    def _should_ignore_file(self, file_path: Path) -> bool:
        ignore_patterns = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache",
                           ".DS_Store", ".ava_project_state.json"}
        return any(pattern in file_path.parts or pattern == file_path.name for pattern in ignore_patterns)

    def _extract_dependencies(self, content: str) -> List[str]:
        import re
        deps = set()
        for match in re.finditer(r"^\s*import\s+([\w.]+)(?:[\s,]+([\w.]+))*", content, re.MULTILINE):
            for group_idx in range(1, match.lastindex + 1):
                if match.group(group_idx):
                    deps.add(match.group(group_idx).split('.')[0])
        for match in re.finditer(r"^\s*from\s+([\.\w]+)\s+import\s+", content, re.MULTILINE):
            module_part = match.group(1)
            if not module_part.startswith('.'): deps.add(module_part.split('.')[0])
        return list(deps)

    def _extract_exports(self, content: str) -> List[str]:
        import re
        exports = set()
        exports.update(re.findall(r"^\s*class\s+(\w+)\s*\(?.*?\)?:\s*$", content, re.MULTILINE))
        exports.update(m[1] for m in re.finditer(r"^\s*(async\s+)?def\s+(\w+)\s*\(", content, re.MULTILINE))
        exports.update(re.findall(r"^([A-Z_][A-Z0-9_]*)\s*[:=]", content, re.MULTILINE))
        return list(exports)

    def _extract_imports(self, content: str) -> List[str]:
        import re
        imports = set()
        for match in re.finditer(r"^\s*import\s+([^\n]+)", content, re.MULTILINE):
            for mod_alias in match.group(1).split(','):
                imports.add(mod_alias.split(' as ')[0].strip().split('.')[0])
        for match in re.finditer(r"^\s*from\s+([\.\w]+)\s+import\s+", content, re.MULTILINE):
            pkg = match.group(1)
            if not pkg.startswith('.'): imports.add(pkg.split('.')[0])
        return list(imports)

    def _get_relative_path(self, file_path_str: str) -> str:
        try:
            abs_file_path = Path(file_path_str).resolve()
            return str(abs_file_path.relative_to(self.project_root))
        except ValueError:
            return Path(file_path_str).name
        except Exception:  # Catch other potential Path errors
            return file_path_str  # Fallback to original string if path ops fail

    def _detect_initial_patterns(self):
        self._update_patterns()

    def _update_patterns(self):
        if not self.files: return
        self._detect_naming_patterns()
        self._detect_import_patterns()
        self._detect_architectural_patterns()

    def _detect_naming_patterns(self):
        function_names, class_names = [], []
        for f_state in self.files.values():
            content = f_state.content
            import re
            function_names.extend(
                m.group(2) for m in re.finditer(r"^\s*(async\s+)?def\s+(\w+)\s*\(", content, re.MULTILINE))
            class_names.extend(re.findall(r"^\s*class\s+(\w+)\s*\(?.*?\)?:\s*$", content, re.MULTILINE))
        if function_names:
            snake = sum(1 for n in function_names if '_' in n and n.islower())
            camel = sum(1 for n in function_names if n[0].islower() and any(c.isupper() for c in n[1:]))
            if snake > camel:
                self.patterns["function_naming"] = ProjectPattern("naming", "snake_case for functions",
                                                                  function_names[:2], 0.7, snake)
            elif camel > snake:
                self.patterns["function_naming"] = ProjectPattern("naming", "camelCase for functions",
                                                                  function_names[:2], 0.7, camel)
        if class_names:
            pascal = sum(1 for n in class_names if n[0].isupper() and not '_' in n)
            if pascal > len(class_names) / 2: self.patterns["class_naming"] = ProjectPattern("naming",
                                                                                             "PascalCase for classes",
                                                                                             class_names[:2], 0.7,
                                                                                             pascal)

    def _detect_import_patterns(self):
        pass

    def _detect_architectural_patterns(self):
        pass

    def _update_metadata(self):
        self.project_metadata.update(
            {"last_updated": datetime.now(), "total_files": len(self.files), "ai_iterations": len(self.ai_decisions)})

    def _identify_main_files(self) -> List[str]:
        return [p for p, f_state in self.files.items() if 'if __name__ == "__main__":' in f_state.content]

    def _detect_architecture_type(self) -> str:
        counts = defaultdict(int)
        patterns = {"gui_application": ["gui", "ui", "view", "window", "pyside", "tkinter"],
                    "web_application": ["app", "api", "route", "view", "server", "flask", "django"],
                    "cli_application": ["cli", "main", "command"], "library": ["lib", "core", "util"]}
        for path in self.files.keys():
            for arch, keywords in patterns.items():
                if any(k in path.lower() for k in keywords): counts[arch] += 1
        return max(counts, key=counts.get) if counts else "library"

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
                            "previous_decisions": [d for d in fs.ai_decisions[-3:]],  # Ensure serializable
                            "user_feedback": [f for f in fs.user_feedback[-2:]]})  # Ensure serializable
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
        return list(related - {file_path})  # Ensure self is not included

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

    def _find_project_inconsistencies(self) -> List[Dict[str, Any]]:
        return []

    def save_state(self, file_path: Optional[str] = None):
        state_file = Path(file_path).resolve() if file_path else (
                    self.project_root / ".ava_project_state.json").resolve()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_data = {
            "metadata": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in
                         self.project_metadata.items()},
            "files": {p: fs.to_dict() for p, fs in self.files.items()},
            "patterns": {pid: p.to_dict() for pid, p in self.patterns.items()},
            "ai_decisions": [d.to_dict() for d in self.ai_decisions],
            "user_preferences": self.user_preferences
        }
        try:
            state_file.write_text(json.dumps(state_data, indent=2, default=str), encoding='utf-8')
            # print(f"Project state saved to {state_file}") # Optional: for debugging
        except Exception as e:
            print(f"Error saving project state to {state_file}: {e}")

    def load_state(self, file_path: Optional[str] = None):
        state_file = Path(file_path).resolve() if file_path else (
                    self.project_root / ".ava_project_state.json").resolve()
        if not state_file.exists(): return
        try:
            data = json.loads(state_file.read_text(encoding='utf-8'))
            meta = data.get("metadata", {})
            self.project_metadata = {
                k: (datetime.fromisoformat(v) if k in ["created", "last_updated"] and isinstance(v, str) else v) for
                k, v in meta.items()}
            self.project_metadata.setdefault("name", self.project_root.name)
            self.project_metadata.setdefault("created", datetime.now())
            self.project_metadata.setdefault("last_updated", datetime.now())

            self.user_preferences = data.get("user_preferences", {})
            self.files.clear()
            self.patterns.clear()
            self.ai_decisions.clear()
            for p, fd in data.get("files", {}).items():
                if 'last_modified' in fd and isinstance(fd['last_modified'], str):
                    fd['last_modified'] = datetime.fromisoformat(fd['last_modified'])
                else:
                    fd['last_modified'] = datetime.now()
                for dl_key in ['ai_decisions', 'user_feedback']:
                    if dl_key in fd:
                        for item in fd[dl_key]:
                            if 'timestamp' in item and isinstance(item['timestamp'], str):
                                item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                            else:
                                item['timestamp'] = datetime.now()  # Fallback for missing/invalid
                self.files[p] = FileState(**fd)
            for pid, pd in data.get("patterns", {}).items(): self.patterns[pid] = ProjectPattern(**pd)
            for dd in data.get("ai_decisions", []):
                if 'timestamp' in dd and isinstance(dd['timestamp'], str):
                    dd['timestamp'] = datetime.fromisoformat(dd['timestamp'])
                else:
                    dd['timestamp'] = datetime.now()
                self.ai_decisions.append(AIDecision(**dd))
            # print(f"Project state loaded from {state_file}") # Optional: for debugging
        except Exception as e:
            print(f"Error loading project state from {state_file}: {e}")