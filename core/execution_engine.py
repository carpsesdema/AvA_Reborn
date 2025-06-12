# core/execution_engine.py - Safe Code Execution & Validation

import asyncio
import subprocess
import tempfile
import sys
import logging
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
            'PySide6', 'tkinter', 'ursina', 'noise' # Added game libs
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

    def __init__(self, project_state_manager, terminal=None, stream_emitter=None):
        self.project_state = project_state_manager
        self.terminal = terminal
        self.stream_emitter = stream_emitter
        self.logger = logging.getLogger(__name__)
        self.environment = SafeExecutionEnvironment()

    async def validate_code(self, file_path: str, code: str) -> CodeExecutionResult:
        """Lightweight validation pipeline for a file's code."""
        if self.stream_emitter:
            self.stream_emitter("ExecutionEngine", "info", f"Validating syntax for {file_path}", "3")

        # Stage 1: Syntax validation
        try:
            compile(code, file_path, 'exec')
        except SyntaxError as e:
            error_msg = f"Syntax error in {file_path} on line {e.lineno}: {e.msg}"
            if self.stream_emitter:
                self.stream_emitter("ExecutionEngine", "error", error_msg, "3")
            return CodeExecutionResult(result=ExecutionResult.SYNTAX_ERROR, output="", error=error_msg,
                                       execution_time=0.0, exit_code=1, suggestions=[], dependencies_missing=[])

        if self.stream_emitter:
            self.stream_emitter("ExecutionEngine", "success", f"Syntax OK for {file_path}", "3")
        return CodeExecutionResult(result=ExecutionResult.SUCCESS, output="Syntax validation passed", error="",
                                   execution_time=0.0, exit_code=0, suggestions=[], dependencies_missing=[])