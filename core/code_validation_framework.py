# core/code_validation_framework.py - Advanced Code Validation & Testing Framework

import ast
import asyncio
import subprocess
import tempfile
import sys
import os
import re
import json
import docker
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import queue
import time

try:
    import pylint.lint
    import pylint.reporters.text

    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    import mypy.api

    MYPY_AVAILABLE = True
except ImportError:
    MYPY_AVAILABLE = False

try:
    import black

    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


class ValidationLevel(Enum):
    BASIC = "basic"  # Syntax and basic checks
    STANDARD = "standard"  # + Style and type checking
    COMPREHENSIVE = "comprehensive"  # + Dynamic testing and security
    PRODUCTION = "production"  # + Performance and deployment checks


class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class IssueType(Enum):
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    STYLE_VIOLATION = "style_violation"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    LOGIC_ERROR = "logic_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    issue_type: IssueType
    severity: str  # error, warning, info
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    rule_id: Optional[str] = None
    suggestion: Optional[str] = None
    example_fix: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ValidationResult:
    """Results from code validation"""
    status: ValidationStatus
    overall_score: float  # 0.0 - 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.BASIC

    # Detailed results by category
    syntax_valid: bool = True
    style_score: float = 1.0
    type_safety_score: float = 1.0
    security_score: float = 1.0
    performance_score: float = 1.0

    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    test_coverage: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


class StaticAnalyzer:
    """Handles static code analysis using various tools"""

    def __init__(self):
        self.available_tools = self._check_available_tools()

    def _check_available_tools(self) -> Dict[str, bool]:
        """Check which static analysis tools are available"""
        return {
            'pylint': PYLINT_AVAILABLE,
            'mypy': MYPY_AVAILABLE,
            'black': BLACK_AVAILABLE,
            'ast': True  # Always available
        }

    async def analyze_code(self, code: str, file_path: str = "temp.py") -> List[ValidationIssue]:
        """Perform comprehensive static analysis"""
        issues = []

        # 1. Syntax validation using AST
        syntax_issues = await self._check_syntax(code)
        issues.extend(syntax_issues)

        # If syntax is invalid, skip other checks
        if any(issue.issue_type == IssueType.SYNTAX_ERROR for issue in syntax_issues):
            return issues

        # 2. Style checking with Black
        if self.available_tools['black']:
            style_issues = await self._check_style_black(code)
            issues.extend(style_issues)

        # 3. Linting with Pylint
        if self.available_tools['pylint']:
            lint_issues = await self._check_pylint(code, file_path)
            issues.extend(lint_issues)

        # 4. Type checking with MyPy
        if self.available_tools['mypy']:
            type_issues = await self._check_mypy(code, file_path)
            issues.extend(type_issues)

        # 5. Security analysis
        security_issues = await self._check_security(code)
        issues.extend(security_issues)

        # 6. Performance analysis
        performance_issues = await self._check_performance(code)
        issues.extend(performance_issues)

        return issues

    async def _check_syntax(self, code: str) -> List[ValidationIssue]:
        """Check syntax using Python AST"""
        issues = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.SYNTAX_ERROR,
                severity="error",
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                column_number=e.offset,
                suggestion="Fix the syntax error before proceeding",
                confidence=1.0
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.SYNTAX_ERROR,
                severity="error",
                message=f"Parse error: {str(e)}",
                suggestion="Check for malformed code structure",
                confidence=0.8
            ))

        return issues

    async def _check_style_black(self, code: str) -> List[ValidationIssue]:
        """Check code style using Black formatter"""
        if not BLACK_AVAILABLE:
            return []

        issues = []

        try:
            # Check if Black would make changes
            formatted_code = black.format_str(code, mode=black.FileMode())

            if formatted_code != code:
                # Calculate the differences
                original_lines = code.splitlines()
                formatted_lines = formatted_code.splitlines()

                differences = self._find_line_differences(original_lines, formatted_lines)

                for line_num, diff_type, suggestion in differences:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.STYLE_VIOLATION,
                        severity="warning",
                        message=f"Style issue: {diff_type}",
                        line_number=line_num,
                        suggestion=suggestion,
                        example_fix=formatted_lines[line_num - 1] if line_num <= len(formatted_lines) else None,
                        confidence=0.9
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.STYLE_VIOLATION,
                severity="info",
                message=f"Style check failed: {str(e)}",
                suggestion="Manual style review recommended",
                confidence=0.5
            ))

        return issues

    def _find_line_differences(self, original: List[str], formatted: List[str]) -> List[Tuple[int, str, str]]:
        """Find differences between original and formatted code"""
        differences = []

        # Simple line-by-line comparison
        max_lines = max(len(original), len(formatted))

        for i in range(max_lines):
            orig_line = original[i] if i < len(original) else ""
            fmt_line = formatted[i] if i < len(formatted) else ""

            if orig_line != fmt_line:
                if orig_line.strip() == fmt_line.strip():
                    diff_type = "whitespace formatting"
                    suggestion = "Adjust whitespace according to PEP 8"
                elif len(orig_line) > 88 and len(fmt_line) <= 88:
                    diff_type = "line too long"
                    suggestion = "Break long line for better readability"
                else:
                    diff_type = "formatting"
                    suggestion = "Apply Black formatting"

                differences.append((i + 1, diff_type, suggestion))

        return differences

    async def _check_pylint(self, code: str, file_path: str) -> List[ValidationIssue]:
        """Check code using Pylint"""
        if not PYLINT_AVAILABLE:
            return []

        issues = []

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Run Pylint
                pylint_output = subprocess.run([
                    sys.executable, '-m', 'pylint',
                    '--output-format=json',
                    '--disable=C0114,C0115,C0116',  # Disable some docstring warnings for micro-tasks
                    temp_file
                ], capture_output=True, text=True, timeout=30)

                # Parse Pylint JSON output
                if pylint_output.stdout:
                    pylint_results = json.loads(pylint_output.stdout)

                    for result in pylint_results:
                        issue_type = self._map_pylint_type(result.get('type', 'warning'))

                        issues.append(ValidationIssue(
                            issue_type=issue_type,
                            severity=result.get('type', 'warning'),
                            message=result.get('message', ''),
                            line_number=result.get('line'),
                            column_number=result.get('column'),
                            rule_id=result.get('message-id'),
                            suggestion=self._get_pylint_suggestion(result.get('message-id', '')),
                            confidence=0.8
                        ))

            finally:
                # Clean up temporary file
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                issue_type=IssueType.LOGIC_ERROR,
                severity="warning",
                message="Pylint analysis timed out",
                suggestion="Code may be too complex for analysis",
                confidence=0.5
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.LOGIC_ERROR,
                severity="info",
                message=f"Pylint analysis failed: {str(e)}",
                suggestion="Manual code review recommended",
                confidence=0.3
            ))

        return issues

    def _map_pylint_type(self, pylint_type: str) -> IssueType:
        """Map Pylint message types to our issue types"""
        mapping = {
            'error': IssueType.LOGIC_ERROR,
            'warning': IssueType.STYLE_VIOLATION,
            'refactor': IssueType.PERFORMANCE_ISSUE,
            'convention': IssueType.STYLE_VIOLATION,
            'info': IssueType.STYLE_VIOLATION
        }
        return mapping.get(pylint_type, IssueType.LOGIC_ERROR)

    def _get_pylint_suggestion(self, message_id: str) -> str:
        """Get suggestion for Pylint message"""
        suggestions = {
            'C0103': "Use snake_case naming convention",
            'C0301': "Break long lines (max 88 characters)",
            'W0613': "Remove unused function parameter or add underscore prefix",
            'W0622': "Avoid redefining built-in names",
            'R0903': "Add more methods to class or consider using a function",
            'R0913': "Reduce number of function parameters (max 5)",
            'W0611': "Remove unused import",
            'E1101': "Check object attributes and method calls"
        }
        return suggestions.get(message_id, "Review Pylint documentation for this issue")

    async def _check_mypy(self, code: str, file_path: str) -> List[ValidationIssue]:
        """Check types using MyPy"""
        if not MYPY_AVAILABLE:
            return []

        issues = []

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Run MyPy
                result = mypy.api.run([
                    '--no-error-summary',
                    '--show-column-numbers',
                    temp_file
                ])

                stdout, stderr, exit_code = result

                # Parse MyPy output
                for line in stdout.splitlines():
                    if ':' in line and ('error:' in line or 'warning:' in line):
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            try:
                                line_num = int(parts[1])
                                col_num = int(parts[2]) if parts[2].isdigit() else None
                                message = parts[3].strip()

                                severity = "error" if "error:" in message else "warning"
                                message = message.replace("error:", "").replace("warning:", "").strip()

                                issues.append(ValidationIssue(
                                    issue_type=IssueType.TYPE_ERROR,
                                    severity=severity,
                                    message=f"Type issue: {message}",
                                    line_number=line_num,
                                    column_number=col_num,
                                    suggestion=self._get_mypy_suggestion(message),
                                    confidence=0.9
                                ))
                            except ValueError:
                                continue

            finally:
                os.unlink(temp_file)

        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.TYPE_ERROR,
                severity="info",
                message=f"Type checking failed: {str(e)}",
                suggestion="Manual type review recommended",
                confidence=0.3
            ))

        return issues

    def _get_mypy_suggestion(self, message: str) -> str:
        """Get suggestion for MyPy error"""
        if "has no attribute" in message:
            return "Check object type and available attributes"
        elif "incompatible types" in message:
            return "Add type annotations or convert types"
        elif "Cannot determine type" in message:
            return "Add explicit type annotations"
        elif "imported but unused" in message:
            return "Remove unused import or use TYPE_CHECKING"
        else:
            return "Review type annotations and ensure type consistency"

    async def _check_security(self, code: str) -> List[ValidationIssue]:
        """Check for common security issues"""
        issues = []
        lines = code.splitlines()

        security_patterns = [
            (r'eval\s*\(', "Use of eval() is dangerous", "Use safer alternatives like ast.literal_eval()"),
            (r'exec\s*\(', "Use of exec() is dangerous", "Avoid dynamic code execution"),
            (r'input\s*\([^)]*\)', "input() without validation", "Validate and sanitize user input"),
            (r'os\.system\s*\(', "os.system() is vulnerable", "Use subprocess with shell=False"),
            (r'shell\s*=\s*True', "shell=True is dangerous", "Use shell=False and pass command as list"),
            (r'pickle\.loads?\s*\(', "Pickle is unsafe with untrusted data", "Use json or safer serialization"),
            (r'hashlib\.md5\s*\(', "MD5 is cryptographically broken", "Use hashlib.sha256() or stronger"),
            (r'random\.random\s*\(', "random is not cryptographically secure", "Use secrets module for security"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "Use environment variables or secure storage"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "Use environment variables or secure storage"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message, suggestion in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(ValidationIssue(
                        issue_type=IssueType.SECURITY_ISSUE,
                        severity="warning",
                        message=message,
                        line_number=i,
                        suggestion=suggestion,
                        confidence=0.8
                    ))

        return issues

    async def _check_performance(self, code: str) -> List[ValidationIssue]:
        """Check for common performance issues"""
        issues = []
        lines = code.splitlines()

        performance_patterns = [
            (r'\.append\s*\(.+\)\s*for\s+.+\s+in', "List comprehension more efficient",
             "Use list comprehension instead of append in loop"),
            (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(.+\)\s*\)', "Iterate directly over sequence",
             "Use 'for item in sequence' instead of range(len())"),
            (r'\+\s*=\s*["\'].*["\']', "String concatenation in loop", "Use join() for multiple concatenations"),
            (r'global\s+\w+', "Global variables can hurt performance",
             "Consider function parameters or class attributes"),
            (r'import\s+\*', "Star imports are inefficient", "Import specific names or use qualified imports"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, message, suggestion in performance_patterns:
                if re.search(pattern, line):
                    issues.append(ValidationIssue(
                        issue_type=IssueType.PERFORMANCE_ISSUE,
                        severity="info",
                        message=message,
                        line_number=i,
                        suggestion=suggestion,
                        confidence=0.7
                    ))

        return issues


class DynamicTester:
    """Handles dynamic code execution and testing"""

    def __init__(self):
        self.docker_available = DOCKER_AVAILABLE
        self.timeout = 30  # seconds

    async def test_code(self, code: str, test_cases: List[Dict] = None) -> List[ValidationIssue]:
        """Execute code dynamically and run tests"""
        issues = []

        if self.docker_available:
            # Use Docker for safe execution
            docker_issues = await self._test_with_docker(code, test_cases)
            issues.extend(docker_issues)
        else:
            # Use subprocess for safer execution
            subprocess_issues = await self._test_with_subprocess(code, test_cases)
            issues.extend(subprocess_issues)

        return issues

    async def _test_with_docker(self, code: str, test_cases: List[Dict] = None) -> List[ValidationIssue]:
        """Test code in Docker container for maximum safety"""
        issues = []

        try:
            client = docker.from_env()

            # Create test script
            test_script = self._create_test_script(code, test_cases)

            # Run in container with resource limits
            container = client.containers.run(
                'python:3.9-slim',
                command=['python', '-c', test_script],
                detach=True,
                mem_limit='128m',
                cpu_period=100000,
                cpu_quota=50000,  # 50% of one CPU
                network_disabled=True,
                remove=True
            )

            # Wait for completion with timeout
            try:
                result = container.wait(timeout=self.timeout)
                logs = container.logs().decode('utf-8')

                if result['StatusCode'] != 0:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.RUNTIME_ERROR,
                        severity="error",
                        message=f"Runtime error during execution: {logs}",
                        suggestion="Fix runtime errors before deployment",
                        confidence=0.9
                    ))
                else:
                    # Parse test results from logs
                    test_issues = self._parse_test_results(logs)
                    issues.extend(test_issues)

            except docker.errors.DockerException as e:
                issues.append(ValidationIssue(
                    issue_type=IssueType.RUNTIME_ERROR,
                    severity="warning",
                    message=f"Docker execution failed: {str(e)}",
                    suggestion="Code may have resource or security issues",
                    confidence=0.7
                ))

        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.RUNTIME_ERROR,
                severity="info",
                message=f"Dynamic testing failed: {str(e)}",
                suggestion="Manual testing recommended",
                confidence=0.5
            ))

        return issues

    async def _test_with_subprocess(self, code: str, test_cases: List[Dict] = None) -> List[ValidationIssue]:
        """Test code using subprocess with limited permissions"""
        issues = []

        try:
            test_script = self._create_test_script(code, test_cases)

            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_file = f.name

            try:
                # Run with timeout and limited resources
                result = subprocess.run([
                    sys.executable, temp_file
                ], capture_output=True, text=True, timeout=self.timeout)

                if result.returncode != 0:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.RUNTIME_ERROR,
                        severity="error",
                        message=f"Runtime error: {result.stderr}",
                        suggestion="Fix runtime errors before deployment",
                        confidence=0.9
                    ))
                else:
                    # Parse test results
                    test_issues = self._parse_test_results(result.stdout)
                    issues.extend(test_issues)

            finally:
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            issues.append(ValidationIssue(
                issue_type=IssueType.PERFORMANCE_ISSUE,
                severity="warning",
                message="Code execution timed out",
                suggestion="Check for infinite loops or optimize performance",
                confidence=0.8
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.RUNTIME_ERROR,
                severity="info",
                message=f"Execution test failed: {str(e)}",
                suggestion="Manual testing recommended",
                confidence=0.5
            ))

        return issues

    def _create_test_script(self, code: str, test_cases: List[Dict] = None) -> str:
        """Create a test script that includes the code and basic tests"""

        # Basic test template
        test_template = '''
import sys
import traceback
import json

# User code
{user_code}

# Basic validation tests
def run_basic_tests():
    results = []

    try:
        # Test 1: Import validation
        import ast
        ast.parse("""{user_code_escaped}""")
        results.append({"test": "syntax_validation", "status": "passed"})
    except Exception as e:
        results.append({"test": "syntax_validation", "status": "failed", "error": str(e)})

    # Test 2: Basic execution
    try:
        # Try to execute basic functions if they exist
        globals_dict = globals()
        functions = [name for name, obj in globals_dict.items() 
                    if callable(obj) and not name.startswith('_')]

        for func_name in functions[:3]:  # Test first 3 functions
            func = globals_dict[func_name]
            try:
                # Try calling with no arguments (if possible)
                import inspect
                sig = inspect.signature(func)
                if len(sig.parameters) == 0:
                    result = func()
                    results.append({"test": f"function_{func_name}", "status": "passed"})
                else:
                    results.append({"test": f"function_{func_name}", "status": "skipped", "reason": "requires_parameters"})
            except Exception as e:
                results.append({"test": f"function_{func_name}", "status": "failed", "error": str(e)})

    except Exception as e:
        results.append({"test": "execution_test", "status": "failed", "error": str(e)})

    return results

if __name__ == "__main__":
    try:
        test_results = run_basic_tests()
        print("TEST_RESULTS_START")
        print(json.dumps(test_results, indent=2))
        print("TEST_RESULTS_END")
    except Exception as e:
        print(f"Test execution failed: {{e}}")
        traceback.print_exc()
'''

        # Escape the user code for string insertion
        escaped_code = code.replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")

        return test_template.format(
            user_code=code,
            user_code_escaped=escaped_code
        )

    def _parse_test_results(self, output: str) -> List[ValidationIssue]:
        """Parse test results from execution output"""
        issues = []

        try:
            # Extract JSON results
            start_marker = "TEST_RESULTS_START"
            end_marker = "TEST_RESULTS_END"

            if start_marker in output and end_marker in output:
                start_idx = output.find(start_marker) + len(start_marker)
                end_idx = output.find(end_marker)
                json_str = output[start_idx:end_idx].strip()

                test_results = json.loads(json_str)

                for result in test_results:
                    if result.get('status') == 'failed':
                        issues.append(ValidationIssue(
                            issue_type=IssueType.RUNTIME_ERROR,
                            severity="error",
                            message=f"Test '{result['test']}' failed: {result.get('error', 'Unknown error')}",
                            suggestion="Fix the runtime error in the code",
                            confidence=0.9
                        ))
                    elif result.get('status') == 'skipped':
                        issues.append(ValidationIssue(
                            issue_type=IssueType.LOGIC_ERROR,
                            severity="info",
                            message=f"Test '{result['test']}' skipped: {result.get('reason', 'Unknown reason')}",
                            suggestion="Consider adding basic parameter validation",
                            confidence=0.5
                        ))

        except Exception:
            # If we can't parse results, check for obvious errors in output
            if "Traceback" in output or "Error:" in output:
                issues.append(ValidationIssue(
                    issue_type=IssueType.RUNTIME_ERROR,
                    severity="error",
                    message="Runtime error detected during execution",
                    suggestion="Review code for runtime issues",
                    confidence=0.7
                ))

        return issues


class AutoTestGenerator:
    """Generates test cases automatically based on code analysis"""

    def __init__(self):
        pass

    async def generate_tests(self, code: str, function_signatures: List[Dict] = None) -> List[Dict[str, Any]]:
        """Generate test cases for the given code"""
        tests = []

        # Parse code to find functions
        try:
            tree = ast.parse(code)
            functions = self._extract_functions(tree)

            for func_info in functions:
                func_tests = await self._generate_function_tests(func_info)
                tests.extend(func_tests)

        except Exception as e:
            print(f"Test generation failed: {e}")

        return tests

    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function information from AST"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'arg_types': {},
                    'return_type': None,
                    'docstring': ast.get_docstring(node)
                }

                # Extract type hints
                for arg in node.args.args:
                    if arg.annotation:
                        if isinstance(arg.annotation, ast.Name):
                            func_info['arg_types'][arg.arg] = arg.annotation.id
                        elif isinstance(arg.annotation, ast.Constant):
                            func_info['arg_types'][arg.arg] = arg.annotation.value

                if node.returns:
                    if isinstance(node.returns, ast.Name):
                        func_info['return_type'] = node.returns.id

                functions.append(func_info)

        return functions

    async def _generate_function_tests(self, func_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for a specific function"""
        tests = []
        func_name = func_info['name']
        args = func_info['args']
        arg_types = func_info['arg_types']

        # Skip private functions and special methods
        if func_name.startswith('_'):
            return tests

        # Generate basic test cases based on argument types
        test_cases = self._generate_test_values(args, arg_types)

        for i, test_case in enumerate(test_cases):
            tests.append({
                'function_name': func_name,
                'test_name': f"test_{func_name}_{i + 1}",
                'arguments': test_case,
                'expected_type': func_info.get('return_type'),
                'description': f"Test {func_name} with {', '.join(f'{k}={v}' for k, v in test_case.items())}"
            })

        return tests

    def _generate_test_values(self, args: List[str], arg_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate test values based on argument types"""
        test_values = []

        if not args:
            return [{}]  # No arguments

        # Generate different combinations of test values
        type_to_values = {
            'int': [0, 1, -1, 42, 100],
            'float': [0.0, 1.0, -1.0, 3.14, 100.5],
            'str': ["", "test", "hello world", "123", "special!@#"],
            'bool': [True, False],
            'list': [[], [1, 2, 3], ["a", "b"], [1, "mixed", True]],
            'dict': [{}, {"key": "value"}, {"a": 1, "b": 2}],
            'None': [None],
        }

        # Start with happy path values
        happy_case = {}
        for arg in args:
            arg_type = arg_types.get(arg, 'str')  # Default to str
            values = type_to_values.get(arg_type, ["default"])
            happy_case[arg] = values[1] if len(values) > 1 else values[0]

        test_values.append(happy_case)

        # Add edge cases for each argument
        for arg in args:
            arg_type = arg_types.get(arg, 'str')
            edge_values = type_to_values.get(arg_type, ["edge"])

            for edge_value in edge_values[:2]:  # Limit to 2 edge cases per arg
                edge_case = happy_case.copy()
                edge_case[arg] = edge_value
                test_values.append(edge_case)

        return test_values[:5]  # Limit total test cases


class CodeValidationFramework:
    """
    🔍 Comprehensive Code Validation Framework

    Provides multi-layer validation including:
    - Static analysis (syntax, style, types, security)
    - Dynamic testing (execution, automated tests)
    - Performance analysis
    - Security scanning
    - Intelligent feedback and suggestions
    """

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.static_analyzer = StaticAnalyzer()
        self.dynamic_tester = DynamicTester()
        self.test_generator = AutoTestGenerator()

        # Validation metrics
        self.validation_history: List[ValidationResult] = []

    async def validate_code(self, code: str, file_path: str = "temp.py",
                            context: Dict[str, Any] = None) -> ValidationResult:
        """
        Perform comprehensive code validation
        """
        start_time = time.time()

        result = ValidationResult(
            status=ValidationStatus.PASSED,
            overall_score=1.0,
            validation_level=self.validation_level
        )

        try:
            # 1. Static Analysis
            static_issues = await self.static_analyzer.analyze_code(code, file_path)
            result.issues.extend(static_issues)

            # 2. Dynamic Testing (if validation level allows)
            if self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
                # Generate tests
                auto_tests = await self.test_generator.generate_tests(code)

                # Run dynamic tests
                dynamic_issues = await self.dynamic_tester.test_code(code, auto_tests)
                result.issues.extend(dynamic_issues)

                # Update test metrics
                result.tests_passed = len([t for t in auto_tests if 'passed' in str(t)])
                result.tests_failed = len([t for t in auto_tests if 'failed' in str(t)])

            # 3. Calculate scores and overall result
            await self._calculate_validation_scores(result)

            # 4. Generate recommendations
            result.recommendations = await self._generate_recommendations(result, context)
            result.next_steps = await self._generate_next_steps(result)

            # 5. Update execution time
            result.execution_time = time.time() - start_time

            # 6. Store in history
            self.validation_history.append(result)

            return result

        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.overall_score = 0.0
            result.issues.append(ValidationIssue(
                issue_type=IssueType.LOGIC_ERROR,
                severity="error",
                message=f"Validation framework error: {str(e)}",
                suggestion="Manual code review required",
                confidence=1.0
            ))
            result.execution_time = time.time() - start_time
            return result

    async def _calculate_validation_scores(self, result: ValidationResult):
        """Calculate detailed scores for different aspects"""
        issues = result.issues

        # Check syntax validity
        syntax_errors = [i for i in issues if i.issue_type == IssueType.SYNTAX_ERROR]
        result.syntax_valid = len(syntax_errors) == 0

        # Calculate style score
        style_issues = [i for i in issues if i.issue_type == IssueType.STYLE_VIOLATION]
        result.style_score = max(0.0, 1.0 - (len(style_issues) * 0.1))

        # Calculate type safety score
        type_issues = [i for i in issues if i.issue_type == IssueType.TYPE_ERROR]
        result.type_safety_score = max(0.0, 1.0 - (len(type_issues) * 0.2))

        # Calculate security score
        security_issues = [i for i in issues if i.issue_type == IssueType.SECURITY_ISSUE]
        result.security_score = max(0.0, 1.0 - (len(security_issues) * 0.3))

        # Calculate performance score
        performance_issues = [i for i in issues if i.issue_type == IssueType.PERFORMANCE_ISSUE]
        result.performance_score = max(0.0, 1.0 - (len(performance_issues) * 0.1))

        # Calculate overall score
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])
        info_count = len([i for i in issues if i.severity == "info"])

        # Weight errors more heavily than warnings
        penalty = (error_count * 0.3) + (warning_count * 0.1) + (info_count * 0.05)
        result.overall_score = max(0.0, 1.0 - penalty)

        # Update overall status
        if error_count > 0:
            result.status = ValidationStatus.FAILED
        elif warning_count > 5:
            result.status = ValidationStatus.WARNING
        else:
            result.status = ValidationStatus.PASSED

        # Store metrics
        result.metrics = {
            'total_issues': len(issues),
            'error_count': error_count,
            'warning_count': warning_count,
            'info_count': info_count,
            'lines_of_code': len([line for line in issues[0].message.split('\n') if line.strip()]) if issues else 0,
            'issues_per_line': len(issues) / max(1, result.metrics.get('lines_of_code', 1))
        }

    async def _generate_recommendations(self, result: ValidationResult, context: Dict[str, Any] = None) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []

        # Syntax recommendations
        if not result.syntax_valid:
            recommendations.append("🔧 Fix syntax errors before proceeding with other improvements")

        # Style recommendations
        if result.style_score < 0.8:
            recommendations.append("📏 Run Black formatter to fix style issues automatically")
            recommendations.append("📝 Consider using an IDE with PEP 8 linting enabled")

        # Type safety recommendations
        if result.type_safety_score < 0.7:
            recommendations.append("🏷️ Add type hints to improve code clarity and catch type errors")
            recommendations.append("🔍 Run MyPy regularly to catch type inconsistencies")

        # Security recommendations
        if result.security_score < 0.9:
            recommendations.append("🔒 Address security issues before deploying to production")
            recommendations.append("🛡️ Use input validation and sanitization for all user inputs")

        # Performance recommendations
        if result.performance_score < 0.8:
            recommendations.append("⚡ Optimize performance-critical sections identified")
            recommendations.append("📊 Consider profiling the code for detailed performance analysis")

        # Testing recommendations
        if result.tests_failed > 0:
            recommendations.append("🧪 Fix failing tests before considering code complete")

        if result.tests_passed < 3:
            recommendations.append("🧪 Add more comprehensive test cases")

        # Context-specific recommendations
        if context:
            domain = context.get('domain', '')
            if 'api' in domain.lower():
                recommendations.append("🌐 Ensure proper error handling for API endpoints")
                recommendations.append("📡 Add input validation for all API parameters")
            elif 'gui' in domain.lower():
                recommendations.append("🖼️ Ensure proper event handling and user feedback")
                recommendations.append("🎨 Consider accessibility and user experience")

        return recommendations[:6]  # Limit to most important recommendations

    async def _generate_next_steps(self, result: ValidationResult) -> List[str]:
        """Generate next steps based on validation results"""
        next_steps = []

        if result.status == ValidationStatus.FAILED:
            next_steps.append("Fix critical errors before proceeding")
            next_steps.append("Re-run validation after fixes")
        elif result.status == ValidationStatus.WARNING:
            next_steps.append("Address warnings for better code quality")
            next_steps.append("Consider adding more tests")
        else:
            next_steps.append("Code validation passed - ready for integration")
            next_steps.append("Consider adding performance optimizations")

        if result.overall_score < 0.8:
            next_steps.append("Improve code quality to reach 80%+ score")

        return next_steps

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics about validation history"""
        if not self.validation_history:
            return {"total_validations": 0}

        recent_results = self.validation_history[-10:]  # Last 10 validations

        return {
            "total_validations": len(self.validation_history),
            "average_score": sum(r.overall_score for r in recent_results) / len(recent_results),
            "average_issues": sum(len(r.issues) for r in recent_results) / len(recent_results),
            "success_rate": len([r for r in recent_results if r.status == ValidationStatus.PASSED]) / len(
                recent_results),
            "average_execution_time": sum(r.execution_time for r in recent_results) / len(recent_results),
            "common_issue_types": self._get_common_issue_types(recent_results)
        }

    def _get_common_issue_types(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Get most common issue types from recent validations"""
        issue_counts = {}

        for result in results:
            for issue in result.issues:
                issue_type = issue.issue_type.value
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        # Return top 5 most common issues
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])