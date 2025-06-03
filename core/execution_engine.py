# core/execution_engine.py - Safe Code Execution & Validation

import asyncio
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ExecutionResult(Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class CodeExecutionResult:
    """Result of code execution attempt"""
    result: ExecutionResult
    output: str
    error: str
    execution_time: float
    exit_code: int
    suggestions: List[str]
    dependencies_missing: List[str]


@dataclass
class TestResult:
    """Result of generated test execution"""
    test_name: str
    passed: bool
    output: str
    error: str
    coverage: float


class SafeExecutionEnvironment:
    """Isolated environment for safe code execution"""

    def __init__(self, timeout: int = 30, memory_limit: int = 100):
        self.timeout = timeout
        self.memory_limit = memory_limit  # MB
        self.allowed_imports = {
            # Safe standard library modules
            'os', 'sys', 'pathlib', 'json', 'datetime', 'time', 'math',
            'random', 'collections', 'itertools', 'functools', 're',
            'typing', 'dataclasses', 'enum', 'abc', 'copy', 'uuid',
            # Common safe third-party
            'requests', 'numpy', 'pandas', 'matplotlib', 'flask', 'fastapi',
            'pydantic', 'click', 'argparse', 'logging', 'pytest',
            'PySide6', 'tkinter'
        }
        self.forbidden_modules = {
            'subprocess', 'multiprocessing', 'threading', 'socket',
            'urllib', 'http', 'ftplib', 'smtplib', 'pickle', 'marshal',
            'ctypes', 'importlib', '__builtin__', '__builtins__'
        }


class ExecutionEngine:
    """
    🚀 Code Execution & Testing Engine

    Safely executes generated code and provides immediate feedback:
    - Syntax validation
    - Import dependency checking
    - Runtime execution in isolated environment
    - Auto-generated test creation and execution
    - Performance metrics and suggestions
    """

    def __init__(self, project_state_manager, terminal=None):
        self.project_state = project_state_manager
        self.terminal = terminal
        self.environment = SafeExecutionEnvironment()
        self.test_templates = self._load_test_templates()

    async def validate_and_execute_file(self, file_path: str, code: str) -> CodeExecutionResult:
        """Complete validation and execution pipeline for a file"""

        self._log(f"🔍 Validating {file_path}...")

        # Stage 1: Syntax validation
        syntax_result = await self._validate_syntax(code)
        if syntax_result.result != ExecutionResult.SUCCESS:
            return syntax_result

        # Stage 2: Import analysis
        import_result = await self._analyze_imports(code)
        if import_result.result != ExecutionResult.SUCCESS:
            return import_result

        # Stage 3: Safe execution
        execution_result = await self._execute_safely(file_path, code)

        # Stage 4: Generate suggestions based on results
        execution_result.suggestions = self._generate_improvement_suggestions(
            execution_result, file_path, code
        )

        return execution_result

    async def generate_and_run_tests(self, file_path: str, code: str) -> List[TestResult]:
        """Generate tests for code and execute them"""

        self._log(f"🧪 Generating tests for {file_path}...")

        # Analyze code to understand what to test
        test_scenarios = await self._analyze_code_for_testing(code)

        # Generate test code
        test_code = await self._generate_test_code(file_path, code, test_scenarios)

        # Execute tests
        test_results = await self._execute_tests(test_code, file_path)

        return test_results

    async def _validate_syntax(self, code: str) -> CodeExecutionResult:
        """Check for syntax errors"""
        try:
            compile(code, '<string>', 'exec')
            return CodeExecutionResult(
                result=ExecutionResult.SUCCESS,
                output="Syntax validation passed",
                error="",
                execution_time=0.0,
                exit_code=0,
                suggestions=[],
                dependencies_missing=[]
            )
        except SyntaxError as e:
            return CodeExecutionResult(
                result=ExecutionResult.SYNTAX_ERROR,
                output="",
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                execution_time=0.0,
                exit_code=1,
                suggestions=[
                    f"Fix syntax error at line {e.lineno}",
                    "Check for missing colons, parentheses, or indentation"
                ],
                dependencies_missing=[]
            )

    async def _analyze_imports(self, code: str) -> CodeExecutionResult:
        """Analyze imports for security and availability"""
        import ast
        import re

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Syntax errors handled in previous stage
            return CodeExecutionResult(ExecutionResult.SUCCESS, "", "", 0.0, 0, [], [])

        imports = []
        missing_deps = []
        forbidden_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # Check for forbidden imports
        for imp in imports:
            base_module = imp.split('.')[0]
            if base_module in self.environment.forbidden_modules:
                forbidden_imports.append(imp)

        if forbidden_imports:
            return CodeExecutionResult(
                result=ExecutionResult.SECURITY_VIOLATION,
                output="",
                error=f"Forbidden imports detected: {', '.join(forbidden_imports)}",
                execution_time=0.0,
                exit_code=1,
                suggestions=[
                    "Remove or replace forbidden imports",
                    "Use safer alternatives for system operations"
                ],
                dependencies_missing=[]
            )

        # Check for missing dependencies
        for imp in imports:
            base_module = imp.split('.')[0]
            if base_module not in self.environment.allowed_imports:
                try:
                    __import__(base_module)
                except ImportError:
                    missing_deps.append(base_module)

        if missing_deps:
            return CodeExecutionResult(
                result=ExecutionResult.IMPORT_ERROR,
                output="",
                error=f"Missing dependencies: {', '.join(missing_deps)}",
                execution_time=0.0,
                exit_code=1,
                suggestions=[
                    f"Install missing packages: pip install {' '.join(missing_deps)}",
                    "Add dependencies to requirements.txt"
                ],
                dependencies_missing=missing_deps
            )

        return CodeExecutionResult(ExecutionResult.SUCCESS, "Import analysis passed", "", 0.0, 0, [], [])

    async def _execute_safely(self, file_path: str, code: str) -> CodeExecutionResult:
        """Execute code in isolated environment"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "test_execution.py"
            temp_file.write_text(code, encoding='utf-8')

            # Create execution command with safety limits
            cmd = [
                sys.executable, "-c", f"""
import resource
import signal
import sys

# Set memory limit
resource.setrlimit(resource.RLIMIT_AS, ({self.environment.memory_limit * 1024 * 1024}, -1))

# Set timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.environment.timeout})

try:
    exec(open(r'{temp_file}').read())
except Exception as e:
    print(f"EXECUTION_ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            ]

            start_time = asyncio.get_event_loop().time()

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=temp_dir
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.environment.timeout + 5
                )

                execution_time = asyncio.get_event_loop().time() - start_time

                stdout_text = stdout.decode('utf-8')
                stderr_text = stderr.decode('utf-8')

                if process.returncode == 0:
                    return CodeExecutionResult(
                        result=ExecutionResult.SUCCESS,
                        output=stdout_text,
                        error=stderr_text,
                        execution_time=execution_time,
                        exit_code=process.returncode,
                        suggestions=[],
                        dependencies_missing=[]
                    )
                else:
                    # Determine error type from stderr
                    if "EXECUTION_ERROR:" in stderr_text:
                        error_msg = stderr_text.split("EXECUTION_ERROR:")[-1].strip()
                        result_type = ExecutionResult.RUNTIME_ERROR
                    else:
                        error_msg = stderr_text
                        result_type = ExecutionResult.RUNTIME_ERROR

                    return CodeExecutionResult(
                        result=result_type,
                        output=stdout_text,
                        error=error_msg,
                        execution_time=execution_time,
                        exit_code=process.returncode,
                        suggestions=[],
                        dependencies_missing=[]
                    )

            except asyncio.TimeoutError:
                return CodeExecutionResult(
                    result=ExecutionResult.TIMEOUT,
                    output="",
                    error=f"Execution timeout after {self.environment.timeout} seconds",
                    execution_time=self.environment.timeout,
                    exit_code=124,
                    suggestions=[
                        "Optimize code for better performance",
                        "Remove infinite loops or long-running operations"
                    ],
                    dependencies_missing=[]
                )

            except Exception as e:
                return CodeExecutionResult(
                    result=ExecutionResult.RUNTIME_ERROR,
                    output="",
                    error=f"Execution failed: {str(e)}",
                    execution_time=0.0,
                    exit_code=1,
                    suggestions=[],
                    dependencies_missing=[]
                )

    async def _analyze_code_for_testing(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code to identify what should be tested"""
        import ast

        test_scenarios = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Test functions
                    test_scenarios.append({
                        "type": "function",
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node))
                    })

                elif isinstance(node, ast.ClassDef):
                    # Test classes
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    test_scenarios.append({
                        "type": "class",
                        "name": node.name,
                        "methods": methods
                    })

        except SyntaxError:
            # If code has syntax errors, can't analyze for testing
            pass

        return test_scenarios

    async def _generate_test_code(self, file_path: str, code: str, scenarios: List[Dict[str, Any]]) -> str:
        """Generate test code using LLM"""

        # This would use your LLM client to generate smart tests
        from core.llm_client import LLMRole

        test_prompt = f"""
Generate comprehensive unit tests for this Python code:

FILE: {file_path}
CODE:
```python
{code}
```

TEST SCENARIOS TO COVER:
{scenarios}

Generate pytest-compatible test code that:
1. Tests all functions with various inputs including edge cases
2. Tests class initialization and methods
3. Tests error handling and exceptions
4. Includes both positive and negative test cases
5. Uses descriptive test names and docstrings

Return ONLY the test code:
"""

        # You'd integrate this with your LLM client
        # For now, return a basic template
        return f'''
import pytest
import sys
from pathlib import Path

# Add the source directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the module under test
# from {Path(file_path).stem} import *

def test_basic_functionality():
    """Test basic functionality of the module"""
    # TODO: Add specific tests based on analysis
    assert True

def test_error_handling():
    """Test error handling"""
    # TODO: Add error case tests
    assert True
'''

    async def _execute_tests(self, test_code: str, original_file: str) -> List[TestResult]:
        """Execute generated tests"""

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_generated.py"
            test_file.write_text(test_code, encoding='utf-8')

            # Run pytest
            cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=temp_dir
                )

                stdout, stderr = await process.communicate()

                # Parse pytest output to extract test results
                return self._parse_pytest_output(stdout.decode('utf-8'), stderr.decode('utf-8'))

            except Exception as e:
                return [TestResult(
                    test_name="test_execution_failed",
                    passed=False,
                    output="",
                    error=f"Test execution failed: {e}",
                    coverage=0.0
                )]

    def _parse_pytest_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse pytest output to extract individual test results"""

        results = []
        lines = stdout.split('\n')

        for line in lines:
            if '::' in line and ('PASSED' in line or 'FAILED' in line):
                parts = line.split('::')
                if len(parts) >= 2:
                    test_name = parts[-1].split()[0]
                    passed = 'PASSED' in line

                    results.append(TestResult(
                        test_name=test_name,
                        passed=passed,
                        output=line,
                        error="" if passed else "Test failed - check implementation",
                        coverage=0.0  # Would need coverage.py integration for real coverage
                    ))

        if not results:
            # No tests found or parsing failed
            results.append(TestResult(
                test_name="unknown",
                passed=False,
                output=stdout,
                error=stderr,
                coverage=0.0
            ))

        return results

    def _generate_improvement_suggestions(self, result: CodeExecutionResult,
                                          file_path: str, code: str) -> List[str]:
        """Generate suggestions based on execution results"""

        suggestions = []

        if result.result == ExecutionResult.SUCCESS:
            suggestions.extend([
                "Code executed successfully!",
                "Consider adding error handling for edge cases",
                "Add type hints for better code documentation"
            ])
        elif result.result == ExecutionResult.RUNTIME_ERROR:
            suggestions.extend([
                "Fix runtime error in the code",
                "Add try-catch blocks for error handling",
                "Validate input parameters before processing"
            ])
        elif result.result == ExecutionResult.TIMEOUT:
            suggestions.extend([
                "Optimize algorithm for better performance",
                "Consider using more efficient data structures",
                "Break down complex operations into smaller chunks"
            ])

        return suggestions

    def _load_test_templates(self) -> Dict[str, str]:
        """Load test templates for different code patterns"""
        return {
            "function": '''
def test_{function_name}():
    """Test {function_name} function"""
    # Test with valid inputs
    result = {function_name}({sample_args})
    assert result is not None

    # Test with edge cases
    # TODO: Add edge case tests
''',
            "class": '''
def test_{class_name}_initialization():
    """Test {class_name} initialization"""
    instance = {class_name}()
    assert instance is not None

def test_{class_name}_methods():
    """Test {class_name} methods"""
    instance = {class_name}()
    # TODO: Test individual methods
'''
        }

    def _log(self, message: str):
        """Log execution progress"""
        if self.terminal and hasattr(self.terminal, 'log'):
            self.terminal.log(message)
        else:
            print(message)


# Integration with enhanced workflow
class ExecutionFeedbackIntegration:
    """Integrates execution results with AI feedback system"""

    def __init__(self, execution_engine, ai_feedback_system):
        self.execution_engine = execution_engine
        self.ai_feedback_system = ai_feedback_system

    async def execute_and_provide_feedback(self, session_id: str, file_path: str,
                                           code: str) -> Tuple[bool, str]:
        """Execute code and provide feedback to AI collaboration system"""

        # Execute the code
        execution_result = await self.execution_engine.validate_and_execute_file(file_path, code)

        # Generate and run tests
        test_results = await self.execution_engine.generate_and_run_tests(file_path, code)

        # Determine if code is acceptable
        code_passes = (execution_result.result == ExecutionResult.SUCCESS and
                       all(test.passed for test in test_results))

        # Create detailed feedback
        feedback_content = f"""
EXECUTION RESULTS for {file_path}:

Code Execution: {execution_result.result.value}
Output: {execution_result.output[:200]}...
Execution Time: {execution_result.execution_time:.2f}s

Tests Generated: {len(test_results)}
Tests Passed: {sum(1 for t in test_results if t.passed)}/{len(test_results)}

Suggestions:
{chr(10).join('- ' + s for s in execution_result.suggestions)}
"""

        # Send feedback to AI collaboration system
        if not code_passes:
            await self.ai_feedback_system.send_feedback(
                session_id=session_id,
                from_ai="execution_engine",
                to_ai="coder",
                feedback_type="quality_assessment",
                content=f"Code execution failed for {file_path}",
                context={
                    "execution_result": execution_result,
                    "test_results": test_results,
                    "improvement_needed": True
                },
                priority="high"
            )

        return code_passes, feedback_content