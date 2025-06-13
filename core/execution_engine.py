# core/execution_engine.py - Safe Code Execution & Validation

import asyncio
import subprocess
import tempfile
import sys
import logging
import re
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
    SKIPPED = "skipped"  # For game loops


@dataclass
class CodeExecutionResult:
    """Result of code execution attempt"""
    result: ExecutionResult
    output: str
    error: str
    exit_code: int
    # --- FIX: Add a clean, parsed error message ---
    clean_error: str


class ExecutionEngine:
    """
    🚀 Code Execution & Validation Engine

    Safely executes generated code in a subprocess to provide immediate feedback on:
    - Syntax validation
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

    def _is_game_loop_file(self, code: str) -> bool:
        """Detects if the code likely contains a main game loop that would run forever."""
        game_loop_patterns = ['app.run()', 'pygame.event.get', 'while running:', 'app.exec()']
        return any(pattern in code for pattern in game_loop_patterns)

    def _parse_error(self, stderr: str) -> Tuple[ExecutionResult, str]:
        """Parses stderr to determine a clean error type and message."""
        if "SyntaxError:" in stderr:
            match = re.search(r"SyntaxError: (.*)", stderr)
            clean_message = f"Syntax Error: {match.group(1)}" if match else "Syntax Error: Unspecified"
            return ExecutionResult.SYNTAX_ERROR, clean_message
        if "ImportError:" in stderr:
            match = re.search(r"ImportError: (.*)", stderr)
            clean_message = f"Import Error: {match.group(1)}" if match else "Import Error: Unspecified"
            return ExecutionResult.IMPORT_ERROR, clean_message
        if "ModuleNotFoundError:" in stderr:
            match = re.search(r"ModuleNotFoundError: No module named '(.*)'", stderr)
            clean_message = f"Module Not Found Error: The module '{match.group(1)}' is missing." if match else "Module Not Found Error: Unspecified"
            return ExecutionResult.IMPORT_ERROR, clean_message

        # Fallback for generic runtime errors
        last_line = stderr.strip().split('\n')[-1]
        return ExecutionResult.RUNTIME_ERROR, f"Runtime Error: {last_line}"

    async def validate_code(self, file_path: str, code: str) -> CodeExecutionResult:
        """
        Robust validation pipeline. Writes code to a temporary file and executes it
        in a separate, timed-out process. Skips runtime execution for game loop files.
        """
        self._log(f"Starting robust validation for {file_path}...")

        if self._is_game_loop_file(code):
            self._log(f"Game loop detected in {file_path}. Performing syntax-only check.", "info")
            try:
                compile(code, file_path, 'exec')
                self._log(f"✅ Syntax OK for game loop file: {file_path}", "success")
                return CodeExecutionResult(result=ExecutionResult.SUCCESS,
                                           output="Syntax check passed (runtime skipped).", error="", exit_code=0,
                                           clean_error="")
            except SyntaxError as e:
                error_msg = f"Syntax error in {file_path} on line {e.lineno}: {e.msg}"
                self._log(error_msg, "error")
                return CodeExecutionResult(result=ExecutionResult.SYNTAX_ERROR, output="", error=error_msg, exit_code=1,
                                           clean_error=f"Syntax Error: {e.msg}")

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            temp_file = temp_dir / Path(file_path).name
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(code, encoding='utf-8')

            python_executable = sys.executable
            command = [python_executable, str(temp_file)]
            self._log(f"Executing command: {' '.join(command)}", "debug")

            try:
                proc = await asyncio.create_subprocess_exec(
                    *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(temp_dir)
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
                stdout = stdout_bytes.decode('utf-8', errors='ignore')
                stderr = stderr_bytes.decode('utf-8', errors='ignore')

                if proc.returncode == 0:
                    self._log(f"✅ Validation successful for {file_path}. Exit code: 0.", "success")
                    return CodeExecutionResult(result=ExecutionResult.SUCCESS, output=stdout, error="", exit_code=0,
                                               clean_error="")
                else:
                    self._log(f"❌ Validation failed for {file_path}. Exit code: {proc.returncode}.", "error")
                    self._log(f"Stderr: {stderr.strip()}", "debug", 4)
                    result_type, clean_error_msg = self._parse_error(stderr)
                    return CodeExecutionResult(result=result_type, output=stdout, error=stderr,
                                               exit_code=proc.returncode, clean_error=clean_error_msg)

            except asyncio.TimeoutError:
                self._log(f"⏰ Validation timed out for {file_path} after {self.timeout}s.", "error")
                proc.kill()
                await proc.wait()
                clean_msg = f"Execution timed out after {self.timeout} seconds, possibly due to an infinite loop."
                return CodeExecutionResult(result=ExecutionResult.TIMEOUT, output="", error=clean_msg, exit_code=-1,
                                           clean_error=clean_msg)

            except Exception as e:
                self._log(f"An unexpected error occurred during validation: {e}", "error")
                clean_msg = f"An unexpected exception occurred during execution: {str(e)}"
                return CodeExecutionResult(result=ExecutionResult.RUNTIME_ERROR, output="", error=clean_msg,
                                           exit_code=-1, clean_error=clean_msg)