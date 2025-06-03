# core/diff_engine.py - Intelligent Code Diffing & Merging

import difflib
import re
import ast
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ChangeType(Enum):
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    MOVE = "move"
    REFACTOR = "refactor"


class ChangeImpact(Enum):
    LOW = "low"  # Comments, formatting
    MEDIUM = "medium"  # Variable names, small logic changes
    HIGH = "high"  # Function signatures, major logic
    CRITICAL = "critical"  # Class structure, imports


@dataclass
class CodeChange:
    """Represents a single change in code"""
    change_type: ChangeType
    impact: ChangeImpact
    line_start: int
    line_end: int
    old_content: str
    new_content: str
    description: str
    ai_reasoning: str
    confidence: float
    affects_functions: List[str]
    affects_classes: List[str]


@dataclass
class FileDiff:
    """Complete diff information for a file"""
    file_path: str
    old_version: str
    new_version: str
    changes: List[CodeChange]
    summary: str
    overall_impact: ChangeImpact
    recommendation: str  # approve, review_needed, reject


class SmartDiffEngine:
    """
    🔍 Intelligent Code Diff Engine

    Goes beyond line-by-line diffs to understand:
    - Semantic changes (function renames, refactoring)
    - Impact analysis (what breaks, what improves)
    - AI reasoning integration (why changes were made)
    - Approval recommendations based on change safety
    """

    def __init__(self, project_state_manager=None):
        self.project_state = project_state_manager

    def create_semantic_diff(self, old_code: str, new_code: str,
                             file_path: str, ai_reasoning: str = "") -> FileDiff:
        """Create an intelligent diff that understands code semantics"""

        # Basic line diff
        line_changes = self._get_line_changes(old_code, new_code)

        # Semantic analysis
        semantic_changes = self._analyze_semantic_changes(old_code, new_code)

        # Combine and classify changes
        classified_changes = self._classify_changes(line_changes, semantic_changes, ai_reasoning)

        # Generate summary and recommendation
        summary = self._generate_change_summary(classified_changes)
        impact = self._determine_overall_impact(classified_changes)
        recommendation = self._generate_recommendation(classified_changes, impact)

        return FileDiff(
            file_path=file_path,
            old_version=old_code,
            new_version=new_code,
            changes=classified_changes,
            summary=summary,
            overall_impact=impact,
            recommendation=recommendation
        )

    def _get_line_changes(self, old_code: str, new_code: str) -> List[Dict]:
        """Get basic line-by-line changes"""

        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()

        changes = []
        differ = difflib.unified_diff(old_lines, new_lines, lineterm='')

        current_old_line = 0
        current_new_line = 0

        for line in differ:
            if line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
                if match:
                    current_old_line = int(match.group(1))
                    current_new_line = int(match.group(3))
            elif line.startswith('-'):
                changes.append({
                    'type': 'deletion',
                    'old_line': current_old_line,
                    'content': line[1:],
                    'line_number': current_old_line
                })
                current_old_line += 1
            elif line.startswith('+'):
                changes.append({
                    'type': 'addition',
                    'new_line': current_new_line,
                    'content': line[1:],
                    'line_number': current_new_line
                })
                current_new_line += 1
            elif line.startswith(' '):
                current_old_line += 1
                current_new_line += 1

        return changes

    def _analyze_semantic_changes(self, old_code: str, new_code: str) -> Dict[str, Any]:
        """Analyze semantic-level changes using AST"""

        semantic_info = {
            'functions_added': [],
            'functions_removed': [],
            'functions_modified': [],
            'classes_added': [],
            'classes_removed': [],
            'classes_modified': [],
            'imports_added': [],
            'imports_removed': []
        }

        try:
            old_ast = ast.parse(old_code)
            new_ast = ast.parse(new_code)

            # Extract function and class information
            old_functions = self._extract_functions(old_ast)
            new_functions = self._extract_functions(new_ast)
            old_classes = self._extract_classes(old_ast)
            new_classes = self._extract_classes(new_ast)
            old_imports = self._extract_imports(old_ast)
            new_imports = self._extract_imports(new_ast)

            # Compare functions
            old_func_names = set(old_functions.keys())
            new_func_names = set(new_functions.keys())

            semantic_info['functions_added'] = list(new_func_names - old_func_names)
            semantic_info['functions_removed'] = list(old_func_names - new_func_names)

            # Check for modified functions
            for func_name in old_func_names & new_func_names:
                if old_functions[func_name] != new_functions[func_name]:
                    semantic_info['functions_modified'].append(func_name)

            # Compare classes
            old_class_names = set(old_classes.keys())
            new_class_names = set(new_classes.keys())

            semantic_info['classes_added'] = list(new_class_names - old_class_names)
            semantic_info['classes_removed'] = list(old_class_names - new_class_names)

            # Check for modified classes
            for class_name in old_class_names & new_class_names:
                if old_classes[class_name] != new_classes[class_name]:
                    semantic_info['classes_modified'].append(class_name)

            # Compare imports
            semantic_info['imports_added'] = list(set(new_imports) - set(old_imports))
            semantic_info['imports_removed'] = list(set(old_imports) - set(new_imports))

        except SyntaxError:
            # If either version has syntax errors, skip semantic analysis
            pass

        return semantic_info

    def _extract_functions(self, tree: ast.AST) -> Dict[str, str]:
        """Extract function definitions and their signatures"""
        functions = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Create a signature string
                args = [arg.arg for arg in node.args.args]
                signature = f"{node.name}({', '.join(args)})"
                functions[node.name] = signature

        return functions

    def _extract_classes(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract class definitions and their methods"""
        classes = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                classes[node.name] = methods

        return classes

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")

        return imports

    def _classify_changes(self, line_changes: List[Dict], semantic_changes: Dict,
                          ai_reasoning: str) -> List[CodeChange]:
        """Classify changes by type and impact"""

        classified = []

        # Group line changes into logical changes
        change_groups = self._group_related_changes(line_changes)

        for group in change_groups:
            change = self._create_code_change(group, semantic_changes, ai_reasoning)
            classified.append(change)

        return classified

    def _group_related_changes(self, line_changes: List[Dict]) -> List[List[Dict]]:
        """Group related line changes together"""

        if not line_changes:
            return []

        groups = []
        current_group = [line_changes[0]]

        for i in range(1, len(line_changes)):
            current_change = line_changes[i]
            last_change = current_group[-1]

            # Check if changes are adjacent (within 3 lines)
            current_line = current_change.get('line_number',
                                              current_change.get('old_line', current_change.get('new_line', 0)))
            last_line = last_change.get('line_number', last_change.get('old_line', last_change.get('new_line', 0)))

            if abs(current_line - last_line) <= 3:
                current_group.append(current_change)
            else:
                groups.append(current_group)
                current_group = [current_change]

        groups.append(current_group)
        return groups

    def _create_code_change(self, change_group: List[Dict], semantic_changes: Dict,
                            ai_reasoning: str) -> CodeChange:
        """Create a CodeChange from a group of line changes"""

        # Determine change type
        has_additions = any(c['type'] == 'addition' for c in change_group)
        has_deletions = any(c['type'] == 'deletion' for c in change_group)

        if has_additions and has_deletions:
            change_type = ChangeType.MODIFICATION
        elif has_additions:
            change_type = ChangeType.ADDITION
        else:
            change_type = ChangeType.DELETION

        # Get line range
        all_lines = []
        for change in change_group:
            line_num = change.get('line_number', change.get('old_line', change.get('new_line', 0)))
            all_lines.append(line_num)

        line_start = min(all_lines) if all_lines else 0
        line_end = max(all_lines) if all_lines else 0

        # Build content strings
        old_content = '\n'.join(c['content'] for c in change_group if c['type'] == 'deletion')
        new_content = '\n'.join(c['content'] for c in change_group if c['type'] == 'addition')

        # Determine impact and description
        impact, description = self._analyze_change_impact(old_content, new_content, semantic_changes)

        # Find affected functions and classes
        affected_functions = self._find_affected_functions(line_start, line_end, semantic_changes)
        affected_classes = self._find_affected_classes(line_start, line_end, semantic_changes)

        return CodeChange(
            change_type=change_type,
            impact=impact,
            line_start=line_start,
            line_end=line_end,
            old_content=old_content,
            new_content=new_content,
            description=description,
            ai_reasoning=ai_reasoning,
            confidence=0.8,  # Could be determined by AI confidence
            affects_functions=affected_functions,
            affects_classes=affected_classes
        )

    def _analyze_change_impact(self, old_content: str, new_content: str,
                               semantic_changes: Dict) -> Tuple[ChangeImpact, str]:
        """Analyze the impact level of a change"""

        # Check for high-impact changes
        if any(keyword in new_content.lower() for keyword in ['class ', 'def ', 'import ', 'from ']):
            if semantic_changes['functions_added'] or semantic_changes['classes_added']:
                return ChangeImpact.HIGH, "Added new function or class"
            elif semantic_changes['functions_removed'] or semantic_changes['classes_removed']:
                return ChangeImpact.CRITICAL, "Removed function or class"
            elif semantic_changes['imports_added'] or semantic_changes['imports_removed']:
                return ChangeImpact.HIGH, "Modified imports"

        # Check for medium-impact changes
        if any(keyword in new_content.lower() for keyword in ['return ', 'if ', 'for ', 'while ', '=']):
            return ChangeImpact.MEDIUM, "Modified logic or variable assignment"

        # Check for low-impact changes (comments, whitespace, formatting)
        if new_content.strip().startswith('#') or old_content.strip().startswith('#'):
            return ChangeImpact.LOW, "Updated comments or documentation"

        if not new_content.strip() or not old_content.strip():
            return ChangeImpact.LOW, "Formatting or whitespace change"

        return ChangeImpact.MEDIUM, "Code modification"

    def _find_affected_functions(self, line_start: int, line_end: int,
                                 semantic_changes: Dict) -> List[str]:
        """Find functions affected by the change"""
        affected = []

        # This would need more sophisticated analysis to map line numbers to functions
        # For now, return functions that were modified
        affected.extend(semantic_changes.get('functions_modified', []))
        affected.extend(semantic_changes.get('functions_added', []))

        return affected

    def _find_affected_classes(self, line_start: int, line_end: int,
                               semantic_changes: Dict) -> List[str]:
        """Find classes affected by the change"""
        affected = []

        affected.extend(semantic_changes.get('classes_modified', []))
        affected.extend(semantic_changes.get('classes_added', []))

        return affected

    def _generate_change_summary(self, changes: List[CodeChange]) -> str:
        """Generate a human-readable summary of changes"""

        if not changes:
            return "No changes detected"

        summary_parts = []

        additions = len([c for c in changes if c.change_type == ChangeType.ADDITION])
        deletions = len([c for c in changes if c.change_type == ChangeType.DELETION])
        modifications = len([c for c in changes if c.change_type == ChangeType.MODIFICATION])

        if additions:
            summary_parts.append(f"{additions} addition{'s' if additions > 1 else ''}")
        if deletions:
            summary_parts.append(f"{deletions} deletion{'s' if deletions > 1 else ''}")
        if modifications:
            summary_parts.append(f"{modifications} modification{'s' if modifications > 1 else ''}")

        base_summary = ", ".join(summary_parts)

        # Add impact information
        high_impact = len([c for c in changes if c.impact in [ChangeImpact.HIGH, ChangeImpact.CRITICAL]])
        if high_impact:
            base_summary += f" ({high_impact} high-impact change{'s' if high_impact > 1 else ''})"

        return base_summary

    def _determine_overall_impact(self, changes: List[CodeChange]) -> ChangeImpact:
        """Determine the overall impact of all changes"""

        if not changes:
            return ChangeImpact.LOW

        impacts = [change.impact for change in changes]

        if ChangeImpact.CRITICAL in impacts:
            return ChangeImpact.CRITICAL
        elif ChangeImpact.HIGH in impacts:
            return ChangeImpact.HIGH
        elif ChangeImpact.MEDIUM in impacts:
            return ChangeImpact.MEDIUM
        else:
            return ChangeImpact.LOW

    def _generate_recommendation(self, changes: List[CodeChange],
                                 overall_impact: ChangeImpact) -> str:
        """Generate a recommendation for the changes"""

        if overall_impact == ChangeImpact.CRITICAL:
            return "reject"  # Too risky
        elif overall_impact == ChangeImpact.HIGH:
            return "review_needed"  # Requires careful review
        elif overall_impact == ChangeImpact.MEDIUM:
            if len(changes) > 5:
                return "review_needed"  # Many changes need review
            else:
                return "approve"  # Manageable changes
        else:
            return "approve"  # Low-impact changes are safe


class IntelligentMerger:
    """
    🔀 Intelligent Code Merger

    Handles merging of code changes with conflict resolution:
    - Automatic safe merges for non-conflicting changes
    - Intelligent conflict detection and resolution
    - Preserves user customizations while applying AI improvements
    - Rollback capability for failed merges
    """

    def __init__(self, diff_engine: SmartDiffEngine):
        self.diff_engine = diff_engine

    def merge_changes(self, base_code: str, user_changes: str, ai_changes: str,
                      file_path: str) -> Dict[str, Any]:
        """Intelligently merge user and AI changes"""

        # Create diffs
        user_diff = self.diff_engine.create_semantic_diff(base_code, user_changes, file_path, "User modifications")
        ai_diff = self.diff_engine.create_semantic_diff(base_code, ai_changes, file_path, "AI improvements")

        # Detect conflicts
        conflicts = self._detect_conflicts(user_diff, ai_diff)

        if not conflicts:
            # No conflicts - can auto-merge
            merged_code = self._auto_merge(base_code, user_diff, ai_diff)
            return {
                "success": True,
                "merged_code": merged_code,
                "conflicts": [],
                "strategy": "auto_merge"
            }
        else:
            # Has conflicts - need user resolution
            conflict_resolutions = self._suggest_conflict_resolutions(conflicts)
            return {
                "success": False,
                "merged_code": None,
                "conflicts": conflicts,
                "suggested_resolutions": conflict_resolutions,
                "strategy": "manual_resolution_needed"
            }

    def _detect_conflicts(self, user_diff: FileDiff, ai_diff: FileDiff) -> List[Dict[str, Any]]:
        """Detect conflicts between user and AI changes"""

        conflicts = []

        # Check for overlapping line changes
        for user_change in user_diff.changes:
            for ai_change in ai_diff.changes:
                if self._changes_overlap(user_change, ai_change):
                    conflicts.append({
                        "type": "line_overlap",
                        "user_change": user_change,
                        "ai_change": ai_change,
                        "severity": self._determine_conflict_severity(user_change, ai_change)
                    })

        # Check for semantic conflicts
        semantic_conflicts = self._detect_semantic_conflicts(user_diff, ai_diff)
        conflicts.extend(semantic_conflicts)

        return conflicts

    def _changes_overlap(self, change1: CodeChange, change2: CodeChange) -> bool:
        """Check if two changes overlap in terms of lines"""

        range1 = set(range(change1.line_start, change1.line_end + 1))
        range2 = set(range(change2.line_start, change2.line_end + 1))

        return bool(range1 & range2)

    def _determine_conflict_severity(self, user_change: CodeChange, ai_change: CodeChange) -> str:
        """Determine severity of conflict between changes"""

        if (user_change.impact == ChangeImpact.CRITICAL or
                ai_change.impact == ChangeImpact.CRITICAL):
            return "high"
        elif (user_change.impact == ChangeImpact.HIGH or
              ai_change.impact == ChangeImpact.HIGH):
            return "medium"
        else:
            return "low"

    def _detect_semantic_conflicts(self, user_diff: FileDiff, ai_diff: FileDiff) -> List[Dict[str, Any]]:
        """Detect semantic conflicts beyond line overlaps"""

        conflicts = []

        # Check for function signature conflicts
        user_functions = set()
        ai_functions = set()

        for change in user_diff.changes:
            user_functions.update(change.affects_functions)

        for change in ai_diff.changes:
            ai_functions.update(change.affects_functions)

        conflicting_functions = user_functions & ai_functions

        for func_name in conflicting_functions:
            conflicts.append({
                "type": "function_conflict",
                "function_name": func_name,
                "description": f"Both user and AI modified function '{func_name}'",
                "severity": "medium"
            })

        return conflicts

    def _auto_merge(self, base_code: str, user_diff: FileDiff, ai_diff: FileDiff) -> str:
        """Automatically merge non-conflicting changes"""

        lines = base_code.splitlines()

        # Apply user changes first (user takes precedence)
        for change in sorted(user_diff.changes, key=lambda c: c.line_start, reverse=True):
            lines = self._apply_change_to_lines(lines, change)

        # Apply non-conflicting AI changes
        for change in sorted(ai_diff.changes, key=lambda c: c.line_start, reverse=True):
            # Check if this change would conflict with already applied user changes
            if not self._would_conflict_with_applied_changes(change, user_diff.changes):
                lines = self._apply_change_to_lines(lines, change)

        return '\n'.join(lines)

    def _apply_change_to_lines(self, lines: List[str], change: CodeChange) -> List[str]:
        """Apply a single change to list of lines"""

        if change.change_type == ChangeType.ADDITION:
            # Insert new lines
            new_lines = change.new_content.splitlines()
            lines[change.line_start:change.line_start] = new_lines

        elif change.change_type == ChangeType.DELETION:
            # Remove lines
            del lines[change.line_start:change.line_end + 1]

        elif change.change_type == ChangeType.MODIFICATION:
            # Replace lines
            new_lines = change.new_content.splitlines()
            lines[change.line_start:change.line_end + 1] = new_lines

        return lines

    def _would_conflict_with_applied_changes(self, change: CodeChange,
                                             applied_changes: List[CodeChange]) -> bool:
        """Check if a change would conflict with already applied changes"""

        for applied_change in applied_changes:
            if self._changes_overlap(change, applied_change):
                return True

        return False

    def _suggest_conflict_resolutions(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest resolutions for conflicts"""

        resolutions = []

        for conflict in conflicts:
            if conflict["type"] == "line_overlap":
                user_change = conflict["user_change"]
                ai_change = conflict["ai_change"]

                resolutions.append({
                    "conflict_id": len(resolutions),
                    "description": f"Conflict at lines {user_change.line_start}-{user_change.line_end}",
                    "options": [
                        {
                            "name": "keep_user",
                            "description": "Keep user changes",
                            "preview": user_change.new_content
                        },
                        {
                            "name": "keep_ai",
                            "description": "Keep AI changes",
                            "preview": ai_change.new_content
                        },
                        {
                            "name": "manual_merge",
                            "description": "Manually merge both changes",
                            "preview": f"{user_change.new_content}\n{ai_change.new_content}"
                        }
                    ]
                })

        return resolutions


# Integration with existing workflow
class DiffWorkflowIntegration:
    """Integrates diff/merge system with enhanced workflow"""

    def __init__(self, diff_engine: SmartDiffEngine, merger: IntelligentMerger,
                 project_state_manager, terminal=None):
        self.diff_engine = diff_engine
        self.merger = merger
        self.project_state = project_state_manager
        self.terminal = terminal

        # Track file versions for diffing
        self.file_versions: Dict[str, List[str]] = {}

    def track_file_version(self, file_path: str, content: str, version_label: str = ""):
        """Track a version of a file for later diffing"""

        if file_path not in self.file_versions:
            self.file_versions[file_path] = []

        self.file_versions[file_path].append({
            "content": content,
            "timestamp": datetime.now(),
            "label": version_label or f"v{len(self.file_versions[file_path]) + 1}"
        })

    def get_iteration_diff(self, file_path: str, iteration: int) -> Optional[FileDiff]:
        """Get diff between iterations of a file"""

        if file_path not in self.file_versions:
            return None

        versions = self.file_versions[file_path]

        if iteration < 1 or iteration >= len(versions):
            return None

        old_version = versions[iteration - 1]["content"]
        new_version = versions[iteration]["content"]

        return self.diff_engine.create_semantic_diff(
            old_version, new_version, file_path,
            f"AI iteration {iteration}"
        )

    def get_all_changes_summary(self, file_path: str) -> Dict[str, Any]:
        """Get summary of all changes made to a file"""

        if file_path not in self.file_versions:
            return {"error": "No versions tracked for this file"}

        versions = self.file_versions[file_path]

        if len(versions) < 2:
            return {"changes": [], "summary": "No changes yet"}

        # Compare first version to latest
        first_version = versions[0]["content"]
        latest_version = versions[-1]["content"]

        overall_diff = self.diff_engine.create_semantic_diff(
            first_version, latest_version, file_path,
            "Overall changes"
        )

        return {
            "total_iterations": len(versions),
            "overall_diff": overall_diff,
            "iteration_summaries": [
                {
                    "iteration": i + 1,
                    "label": versions[i]["label"],
                    "timestamp": versions[i]["timestamp"].isoformat()
                }
                for i in range(len(versions))
            ]
        }

    def _log(self, message: str):
        """Log diff/merge operations"""
        if self.terminal and hasattr(self.terminal, 'log'):
            self.terminal.log(message)
        else:
            print(message)