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
    exit_code: int


class ExecutionEngine:
    """
    🚀 Code Execution & Validation Engine

    Safely executes generated code in a subprocess to provide immediate feedback on:
    - Syntax validation (implicit in execution attempt)
    - Import dependency checking
    - Runtime errors
    """

    def __init__(self, project_state_manager, stream_emitter=None):
        self.project_state = project_state_manager
        self.stream_emitter = stream_emitter
        self.logger = logging.getLogger(__name__)
        self.timeout = 15  # seconds

    def _log(self, message: str, level: str = "info", indent: int = 3):
        if self.stream_emitter:
            self.stream_emitter("ExecutionEngine", level, message, str(indent))

    async def validate_code(self, file_path: str, code: str) -> CodeExecutionResult:
        """
        Robust validation pipeline. Writes code to a temporary file and executes it
        in a separate, timed-out process to catch runtime and import errors.
        """
        self._log(f"Starting robust validation for {file_path}...")

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            temp_file = temp_dir / Path(file_path).name  # Use only the filename
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(code, encoding='utf-8')

            python_executable = sys.executable
            command = [python_executable, str(temp_file)]
            self._log(f"Executing command: {' '.join(command)}", "debug")

            try:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(temp_dir)
                )

                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)

                stdout = stdout_bytes.decode('utf-8', errors='ignore')
                stderr = stderr_bytes.decode('utf-8', errors='ignore')
                exit_code = proc.returncode

                if exit_code == 0:
                    self._log(f"✅ Validation successful for {file_path}. Exit code: 0.", "success")
                    return CodeExecutionResult(result=ExecutionResult.SUCCESS, output=stdout, error="", exit_code=0)
                else:
                    self._log(f"❌ Validation failed for {file_path}. Exit code: {exit_code}.", "error")
                    self._log(f"Stderr: {stderr.strip()}", "debug", 4)

                    if "SyntaxError:" in stderr:
                        result_type = ExecutionResult.SYNTAX_ERROR
                    elif "ImportError:" in stderr or "ModuleNotFoundError:" in stderr:
                        result_type = ExecutionResult.IMPORT_ERROR
                    else:
                        result_type = ExecutionResult.RUNTIME_ERROR

                    return CodeExecutionResult(result=result_type, output=stdout, error=stderr, exit_code=exit_code)

            except asyncio.TimeoutError:
                self._log(f"⏰ Validation timed out for {file_path} after {self.timeout}s.", "error")
                proc.kill()
                await proc.wait()
                return CodeExecutionResult(
                    result=ExecutionResult.TIMEOUT,
                    output="",
                    error=f"Execution timed out after {self.timeout} seconds. The code may contain an infinite loop.",
                    exit_code=-1
                )
            except FileNotFoundError:
                self._log(f"❌ Command not found: {python_executable}. Is Python installed correctly?", "error")
                return CodeExecutionResult(
                    result=ExecutionResult.RUNTIME_ERROR,
                    output="",
                    error=f"Python executable not found at '{python_executable}'.",
                    exit_code=127
                )
            except Exception as e:
                self._log(f"An unexpected error occurred during validation: {e}", "error")
                return CodeExecutionResult(
                    result=ExecutionResult.RUNTIME_ERROR,
                    output="",
                    error=f"An unexpected exception occurred during execution: {str(e)}",
                    exit_code=-1
                )