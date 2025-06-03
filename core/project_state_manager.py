# core/project_state_manager.py - Central Project Intelligence & State Management

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from collections import defaultdict


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
    ai_decisions: List[Dict[str, Any]]  # Track AI reasoning for this file
    user_feedback: List[Dict[str, Any]]  # Track user feedback
    quality_score: float = 0.0
    review_status: str = "pending"  # pending, approved, needs_revision


@dataclass
class ProjectPattern:
    """Represents a detected or established pattern in the project"""
    pattern_type: str  # naming, architecture, coding_style, etc.
    description: str
    examples: List[str]
    confidence: float
    usage_count: int


@dataclass
class AIDecision:
    """Represents a decision made by an AI specialist"""
    ai_role: str  # planner, coder, assembler, reviewer
    decision_type: str  # architecture, implementation, naming, etc.
    context: str
    reasoning: str
    alternatives_considered: List[str]
    confidence: float
    timestamp: datetime
    file_affected: Optional[str] = None


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
        self.project_root = Path(project_root)
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

        # Initialize from existing files if any
        self._scan_existing_project()
        self._detect_initial_patterns()

    def add_file(self, file_path: str, content: str, ai_role: str = None,
                 reasoning: str = "") -> FileState:
        """Add or update a file with full context tracking"""
        rel_path = self._get_relative_path(file_path)
        file_hash = hashlib.md5(content.encode()).hexdigest()

        # Analyze file structure
        dependencies = self._extract_dependencies(content)
        exports = self._extract_exports(content)
        imports = self._extract_imports(content)

        # Create or update file state
        file_state = FileState(
            path=rel_path,
            content=content,
            hash=file_hash,
            dependencies=dependencies,
            exports=exports,
            imports=imports,
            last_modified=datetime.now(),
            ai_decisions=[],
            user_feedback=[]
        )

        # Record AI decision if provided
        if ai_role and reasoning:
            decision = AIDecision(
                ai_role=ai_role,
                decision_type="file_creation",
                context=f"Created {rel_path}",
                reasoning=reasoning,
                alternatives_considered=[],
                confidence=0.8,
                timestamp=datetime.now(),
                file_affected=rel_path
            )
            file_state.ai_decisions.append(asdict(decision))
            self.ai_decisions.append(decision)

        self.files[rel_path] = file_state
        self._update_patterns()
        self._update_metadata()

        return file_state

    def get_project_context(self, for_file: str = None, ai_role: str = None) -> Dict[str, Any]:
        """
        🎯 Get comprehensive project context for AI decision making
        This is the KEY method that makes AIs project-aware
        """
        context = {
            "project_overview": {
                "name": self.project_metadata["name"],
                "total_files": len(self.files),
                "main_files": self._identify_main_files(),
                "architecture_type": self._detect_architecture_type(),
            },
            "established_patterns": {
                "naming_conventions": self._get_naming_patterns(),
                "code_structure": self._get_structure_patterns(),
                "import_patterns": self._get_import_patterns(),
                "architectural_decisions": self._get_architectural_decisions(),
            },
            "file_relationships": self._build_dependency_graph(),
            "coding_standards": self._extract_coding_standards(),
            "recent_decisions": self._get_recent_ai_decisions(limit=10),
            "user_preferences": self.user_preferences.copy()
        }

        # Add specific context for the target file
        if for_file:
            context["target_file_context"] = self._get_file_specific_context(for_file)

        # Add role-specific context
        if ai_role:
            context["role_specific_guidance"] = self._get_role_guidance(ai_role)

        return context

    def record_ai_decision(self, ai_role: str, decision_type: str, context: str,
                           reasoning: str, confidence: float = 0.8,
                           alternatives: List[str] = None, file_affected: str = None):
        """Record an AI decision for future context"""
        decision = AIDecision(
            ai_role=ai_role,
            decision_type=decision_type,
            context=context,
            reasoning=reasoning,
            alternatives_considered=alternatives or [],
            confidence=confidence,
            timestamp=datetime.now(),
            file_affected=file_affected
        )

        self.ai_decisions.append(decision)

        # Add to specific file if provided
        if file_affected and file_affected in self.files:
            self.files[file_affected].ai_decisions.append(asdict(decision))

    def get_consistency_requirements(self, file_path: str) -> Dict[str, Any]:
        """Get consistency requirements for a specific file"""
        rel_path = self._get_relative_path(file_path)

        return {
            "naming_style": self._get_consistent_naming_style(),
            "import_organization": self._get_import_style(),
            "documentation_level": self._get_doc_standards(),
            "error_handling_patterns": self._get_error_patterns(),
            "related_files": self._get_related_files(rel_path),
            "interface_contracts": self._get_interface_requirements(rel_path)
        }

    def validate_file_consistency(self, file_path: str, content: str) -> Dict[str, Any]:
        """Validate a file against established project patterns"""
        issues = []
        suggestions = []

        # Check naming consistency
        naming_issues = self._check_naming_consistency(content)
        issues.extend(naming_issues)

        # Check import organization
        import_issues = self._check_import_consistency(content)
        issues.extend(import_issues)

        # Check architectural consistency
        arch_issues = self._check_architectural_consistency(file_path, content)
        issues.extend(arch_issues)

        # Generate suggestions
        suggestions = self._generate_consistency_suggestions(file_path, issues)

        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "consistency_score": max(0, 1.0 - (len(issues) * 0.1))
        }

    def get_next_file_suggestions(self, current_files: List[str]) -> List[Dict[str, Any]]:
        """Suggest what files should be created next based on project state"""
        suggestions = []

        # Analyze missing dependencies
        missing_deps = self._find_missing_dependencies()
        for dep in missing_deps:
            suggestions.append({
                "file_path": f"{dep}.py",
                "reason": f"Required by existing imports",
                "priority": "high",
                "suggested_content": self._suggest_file_structure(dep)
            })

        # Suggest architectural improvements
        arch_suggestions = self._suggest_architectural_files()
        suggestions.extend(arch_suggestions)

        return sorted(suggestions, key=lambda x: x.get("priority", "low"))

    def add_user_feedback(self, file_path: str, feedback_type: str, content: str, rating: int):
        """Record user feedback for continuous improvement"""
        rel_path = self._get_relative_path(file_path)

        feedback = {
            "type": feedback_type,
            "content": content,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }

        if rel_path in self.files:
            self.files[rel_path].user_feedback.append(feedback)

        # Update user preferences based on feedback
        self._update_user_preferences(feedback_type, content, rating)

    def get_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for code improvement across the project"""
        opportunities = []

        # Find files with low quality scores
        low_quality_files = [f for f in self.files.values() if f.quality_score < 0.7]
        for file_state in low_quality_files:
            opportunities.append({
                "type": "quality_improvement",
                "file": file_state.path,
                "current_score": file_state.quality_score,
                "suggestions": self._get_quality_improvement_suggestions(file_state)
            })

        # Find inconsistency patterns
        inconsistencies = self._find_project_inconsistencies()
        opportunities.extend(inconsistencies)

        return opportunities

    # Private helper methods
    def _scan_existing_project(self):
        """Scan existing project files to build initial state"""
        if not self.project_root.exists():
            return

        for py_file in self.project_root.rglob("*.py"):
            if self._should_ignore_file(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8')
                self.add_file(str(py_file), content)
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = {
            "__pycache__", ".git", ".venv", "venv",
            ".pytest_cache", ".mypy_cache", "node_modules"
        }
        return any(pattern in str(file_path) for pattern in ignore_patterns)

    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract file dependencies from imports"""
        import re
        deps = []

        # Find relative imports
        relative_imports = re.findall(r'from\s+\.(\w+)', content)
        deps.extend(relative_imports)

        # Find local imports (same project)
        local_imports = re.findall(r'from\s+(\w+)', content)
        deps.extend([imp for imp in local_imports if not imp.startswith('.')])

        return list(set(deps))

    def _extract_exports(self, content: str) -> List[str]:
        """Extract what this file exports (functions, classes, constants)"""
        import re
        exports = []

        # Find class definitions
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        exports.extend(classes)

        # Find function definitions
        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        exports.extend(functions)

        # Find constants (ALL_CAPS variables)
        constants = re.findall(r'^([A-Z_][A-Z0-9_]*)\s*=', content, re.MULTILINE)
        exports.extend(constants)

        return exports

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements"""
        import re
        imports = []

        import_lines = re.findall(r'^(?:from\s+\S+\s+)?import\s+(.+)', content, re.MULTILINE)
        for line in import_lines:
            imports.extend([item.strip() for item in line.split(',')])

        return imports

    def _get_relative_path(self, file_path: str) -> str:
        """Convert absolute path to relative path from project root"""
        try:
            return str(Path(file_path).relative_to(self.project_root))
        except ValueError:
            return str(Path(file_path).name)

    def _detect_initial_patterns(self):
        """Detect patterns from existing files"""
        self._update_patterns()

    def _update_patterns(self):
        """Update detected patterns based on current files"""
        if not self.files:
            return

        # Naming patterns
        self._detect_naming_patterns()

        # Import patterns
        self._detect_import_patterns()

        # Architectural patterns
        self._detect_architectural_patterns()

    def _detect_naming_patterns(self):
        """Detect naming conventions used in the project"""
        function_names = []
        class_names = []

        for file_state in self.files.values():
            content = file_state.content
            import re

            # Extract function names
            funcs = re.findall(r'def\s+(\w+)', content)
            function_names.extend(funcs)

            # Extract class names
            classes = re.findall(r'class\s+(\w+)', content)
            class_names.extend(classes)

        # Analyze patterns
        if function_names:
            snake_case_count = sum(1 for name in function_names if '_' in name)
            camel_case_count = sum(1 for name in function_names if name[0].islower() and any(c.isupper() for c in name))

            if snake_case_count > camel_case_count:
                self.patterns["function_naming"] = ProjectPattern(
                    pattern_type="naming",
                    description="snake_case for functions",
                    examples=function_names[:3],
                    confidence=0.8,
                    usage_count=snake_case_count
                )

    def _detect_import_patterns(self):
        """Detect import organization patterns"""
        # Implementation for import pattern detection
        pass

    def _detect_architectural_patterns(self):
        """Detect architectural patterns"""
        # Implementation for architectural pattern detection
        pass

    def _update_metadata(self):
        """Update project metadata"""
        self.project_metadata.update({
            "last_updated": datetime.now(),
            "total_files": len(self.files),
            "ai_iterations": len(self.ai_decisions)
        })

    def _identify_main_files(self) -> List[str]:
        """Identify main/entry point files"""
        main_files = []
        for path, file_state in self.files.items():
            if 'if __name__ == "__main__"' in file_state.content:
                main_files.append(path)
        return main_files

    def _detect_architecture_type(self) -> str:
        """Detect project architecture type"""
        file_names = list(self.files.keys())

        if any('gui' in f or 'ui' in f for f in file_names):
            return "gui_application"
        elif any('api' in f or 'routes' in f or 'views' in f for f in file_names):
            return "web_application"
        elif any('cli' in f or 'main' in f for f in file_names):
            return "cli_application"
        else:
            return "library"

    def _get_naming_patterns(self) -> Dict[str, Any]:
        """Get established naming patterns"""
        return {pattern_id: pattern for pattern_id, pattern in self.patterns.items()
                if pattern.pattern_type == "naming"}

    def _get_structure_patterns(self) -> Dict[str, Any]:
        """Get code structure patterns"""
        return {pattern_id: pattern for pattern_id, pattern in self.patterns.items()
                if pattern.pattern_type == "structure"}

    def _get_import_patterns(self) -> Dict[str, Any]:
        """Get import organization patterns"""
        return {pattern_id: pattern for pattern_id, pattern in self.patterns.items()
                if pattern.pattern_type == "imports"}

    def _get_architectural_decisions(self) -> List[Dict[str, Any]]:
        """Get architectural decisions made by AIs"""
        return [asdict(decision) for decision in self.ai_decisions
                if decision.decision_type == "architecture"]

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build file dependency graph"""
        graph = {}
        for path, file_state in self.files.items():
            graph[path] = file_state.dependencies
        return graph

    def _extract_coding_standards(self) -> Dict[str, Any]:
        """Extract coding standards from existing code"""
        return {
            "docstring_style": "google",  # Could be detected
            "type_hints": True,  # Could be detected
            "error_handling": "exceptions",  # Could be detected
            "logging_style": "structured"  # Could be detected
        }

    def _get_recent_ai_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent AI decisions for context"""
        recent = sorted(self.ai_decisions, key=lambda d: d.timestamp, reverse=True)[:limit]
        return [asdict(decision) for decision in recent]

    def _get_file_specific_context(self, file_path: str) -> Dict[str, Any]:
        """Get context specific to a target file"""
        rel_path = self._get_relative_path(file_path)

        context = {
            "related_files": self._get_related_files(rel_path),
            "required_interfaces": self._get_interface_requirements(rel_path),
            "consistency_requirements": self.get_consistency_requirements(file_path)
        }

        if rel_path in self.files:
            file_state = self.files[rel_path]
            context.update({
                "current_exports": file_state.exports,
                "current_dependencies": file_state.dependencies,
                "previous_decisions": file_state.ai_decisions[-5:],  # Last 5 decisions
                "user_feedback": file_state.user_feedback[-3:]  # Last 3 feedback items
            })

        return context

    def _get_role_guidance(self, ai_role: str) -> Dict[str, Any]:
        """Get role-specific guidance"""
        guidance = {
            "planner": {
                "focus": "architecture and file organization",
                "considerations": ["scalability", "maintainability", "separation of concerns"],
                "deliverables": ["project structure", "file responsibilities", "interface definitions"]
            },
            "coder": {
                "focus": "implementation details and code quality",
                "considerations": ["performance", "readability", "error handling"],
                "deliverables": ["clean functions", "proper error handling", "type hints"]
            },
            "assembler": {
                "focus": "integration and consistency",
                "considerations": ["interface compatibility", "naming consistency", "import organization"],
                "deliverables": ["cohesive files", "proper imports", "documentation"]
            },
            "reviewer": {
                "focus": "quality assurance and best practices",
                "considerations": ["code quality", "security", "performance", "maintainability"],
                "deliverables": ["quality assessment", "improvement suggestions", "approval decision"]
            }
        }

        return guidance.get(ai_role, {})

    # Additional helper methods for consistency checking and suggestions...
    def _get_related_files(self, file_path: str) -> List[str]:
        """Find files related to the given file"""
        related = []
        if file_path in self.files:
            current_file = self.files[file_path]

            # Files that import this file
            for path, file_state in self.files.items():
                if any(export in file_state.imports for export in current_file.exports):
                    related.append(path)

            # Files that this file imports from
            related.extend(current_file.dependencies)

        return list(set(related))

    def _check_naming_consistency(self, content: str) -> List[str]:
        """Check naming consistency issues"""
        # Implementation for naming consistency checks
        return []

    def _check_import_consistency(self, content: str) -> List[str]:
        """Check import consistency issues"""
        # Implementation for import consistency checks
        return []

    def _check_architectural_consistency(self, file_path: str, content: str) -> List[str]:
        """Check architectural consistency issues"""
        # Implementation for architectural consistency checks
        return []

    def _generate_consistency_suggestions(self, file_path: str, issues: List[str]) -> List[str]:
        """Generate suggestions based on consistency issues"""
        # Implementation for generating suggestions
        return []

    def _find_missing_dependencies(self) -> List[str]:
        """Find missing dependencies across the project"""
        # Implementation for finding missing deps
        return []

    def _suggest_architectural_files(self) -> List[Dict[str, Any]]:
        """Suggest additional architectural files"""
        # Implementation for architectural suggestions
        return []

    def _suggest_file_structure(self, module_name: str) -> str:
        """Suggest structure for a new file"""
        return f'"""\n{module_name}.py - Module description\n"""\n\n# Implementation here'

    def _update_user_preferences(self, feedback_type: str, content: str, rating: int):
        """Update user preferences based on feedback"""
        if feedback_type not in self.user_preferences:
            self.user_preferences[feedback_type] = []

        self.user_preferences[feedback_type].append({
            "content": content,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        })

    def _get_quality_improvement_suggestions(self, file_state: FileState) -> List[str]:
        """Get suggestions for improving file quality"""
        return ["Add more documentation", "Improve error handling", "Add type hints"]

    def _find_project_inconsistencies(self) -> List[Dict[str, Any]]:
        """Find inconsistencies across the project"""
        return []

    def save_state(self, file_path: str = None):
        """Save project state to disk"""
        state_file = file_path or (self.project_root / ".ava_project_state.json")

        state_data = {
            "metadata": self.project_metadata,
            "files": {path: asdict(file_state) for path, file_state in self.files.items()},
            "patterns": {pid: asdict(pattern) for pid, pattern in self.patterns.items()},
            "ai_decisions": [asdict(decision) for decision in self.ai_decisions],
            "user_preferences": self.user_preferences
        }

        Path(state_file).write_text(json.dumps(state_data, indent=2, default=str), encoding='utf-8')

    def load_state(self, file_path: str = None):
        """Load project state from disk"""
        state_file = file_path or (self.project_root / ".ava_project_state.json")

        if not Path(state_file).exists():
            return

        try:
            state_data = json.loads(Path(state_file).read_text(encoding='utf-8'))

            self.project_metadata = state_data.get("metadata", {})
            self.user_preferences = state_data.get("user_preferences", {})

            # Restore file states
            for path, file_data in state_data.get("files", {}).items():
                self.files[path] = FileState(**file_data)

            # Restore patterns
            for pid, pattern_data in state_data.get("patterns", {}).items():
                self.patterns[pid] = ProjectPattern(**pattern_data)

            # Restore AI decisions
            for decision_data in state_data.get("ai_decisions", []):
                self.ai_decisions.append(AIDecision(**decision_data))

        except Exception as e:
            print(f"Warning: Could not load project state: {e}")