# core/enhanced_workflow_engine.py - Complete V6 "Surgical Modification" Engine with Error Analysis

import asyncio
import json
import logging
import traceback
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal

from core.workflow_services import ArchitectService, CoderService, ReviewerService
from core.project_state_manager import ProjectStateManager


class EnhancedWorkflowEngine(QObject):
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    workflow_progress = Signal(str, str)
    file_generated = Signal(str)
    project_loaded = Signal(str)
    detailed_log_event = Signal(str, str, str, str)
    task_progress = Signal(int, int)
    analysis_started = Signal(str)
    analysis_completed = Signal(str, dict)

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.terminal_window = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

        self.project_state_manager: ProjectStateManager = None
        self.current_tech_spec: dict = None

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        # Error analysis prompt templates
        self.error_analysis_prompts = {
            "syntax_error": """
You are Ava, an expert debugging AI. A Python syntax error occurred during execution.

ERROR CONTEXT:
{error_context}

PROJECT FILES:
{file_list}

TASK: Analyze the syntax error and provide a complete fixed version of the problematic file(s).

RESPONSE FORMAT:
1. Brief explanation of what caused the error
2. Complete corrected file contents (full file, not just the fix)
3. Any additional files that need to be created or modified

Be thorough and ensure your fix addresses the root cause.
""",
            "runtime_error": """
You are Ava, an expert debugging AI. A runtime error occurred during project execution.

ERROR CONTEXT:
{error_context}

PROJECT FILES:
{file_list}

RECENT FILE STATES:
{file_states}

TASK: Analyze the runtime error and provide complete fixed files.

RESPONSE FORMAT:
1. Root cause analysis of the runtime error
2. Complete corrected file contents (full files)
3. Any new dependencies or requirements needed
4. Testing suggestions to prevent similar errors

Focus on creating robust, error-free code.
""",
            "import_error": """
You are Ava, an expert Python dependency resolver. An import/module error occurred.

ERROR CONTEXT:
{error_context}

PROJECT FILES:
{file_list}

TASK: Fix the import/dependency issues and provide complete corrected files.

RESPONSE FORMAT:
1. Analysis of missing dependencies or import issues
2. Updated requirements.txt (if needed)
3. Complete corrected file contents with proper imports
4. Installation instructions if new packages are required

Ensure all imports are properly structured and available.
""",
            "general_error": """
You are Ava, an expert debugging AI. An error occurred during project execution.

ERROR CONTEXT:
{error_context}

PROJECT FILES:
{file_list}

TASK: Analyze and fix the error, providing complete corrected files.

RESPONSE FORMAT:
1. Error analysis and root cause identification
2. Complete fixed file contents (full files, ready to copy-paste)
3. Any additional changes needed (new files, dependencies, etc.)
4. Prevention strategies for similar issues

Be comprehensive and provide production-ready fixes.
"""
        }

        self._connect_terminal_signals()
        self.logger.info("✅ V6 'Surgical Modification' Workflow Engine initialized.")

    def _connect_terminal_signals(self):
        if not self.terminal_window:
            return
        try:
            self.workflow_progress.connect(self.terminal_window.update_workflow_progress)
            self.task_progress.connect(self.terminal_window.update_task_progress)
            self.detailed_log_event.connect(self.terminal_window.stream_log_rich)
        except Exception as e:
            self.logger.error(f"❌ Failed to connect terminal signals: {e}")

    async def execute_analysis_workflow(self, project_path_str: str):
        self.logger.info(f"🚀 Starting Analysis workflow for: {project_path_str}...")
        self.analysis_started.emit(project_path_str)
        self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                     f"🚀 Initializing Analysis for '{Path(project_path_str).name}'...", "0")
        try:
            self.project_state_manager = ProjectStateManager(Path(project_path_str))
            self.detailed_log_event.emit("WorkflowEngine", "info", "Project State Manager initialized. Scanned files.",
                                         "1")
            tech_spec = await self.architect_service.analyze_and_create_spec_from_project(self.project_state_manager)
            if not tech_spec:
                raise Exception(
                    "Analysis failed. Architect could not produce a technical specification from the project files.")
            self.current_tech_spec = tech_spec
            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ Analysis complete! Technical spec created.",
                                         "0")
            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
            self.project_loaded.emit(project_path_str)
        except Exception as e:
            self.logger.error(f"❌ Analysis Workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Analysis Error: {str(e)}", "0")
            self.project_loaded.emit(project_path_str)

    def _get_files_to_modify(self, user_prompt: str, all_files: List[str]) -> List[str]:
        """Identifies which files should be modified based on the user's prompt."""
        files_to_modify = []
        # A simple but effective heuristic: if a filename is mentioned, it's a target.
        for filename in all_files:
            if re.search(r'\b' + re.escape(filename) + r'\b', user_prompt, re.IGNORECASE):
                files_to_modify.append(filename)

        # If no files are explicitly mentioned, we might need a more advanced AI step.
        # For now, if none are mentioned, we'll assume we modify everything (the old behavior).
        if not files_to_modify:
            self.logger.warning("No specific files mentioned in modification prompt. Will attempt to regenerate all.")
            return all_files

        self.logger.info(f"Identified files to modify from prompt: {files_to_modify}")
        return files_to_modify

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        """Enhanced workflow execution with error analysis capabilities"""
        self.logger.info(f"🚀 Starting enhanced workflow: {user_prompt[:100]}...")

        try:
            # Check if this is an error analysis request
            if await self._is_error_analysis_request(user_prompt):
                await self._handle_error_analysis(user_prompt, conversation_context)
            else:
                # Regular workflow execution
                await self._execute_standard_workflow(user_prompt, conversation_context)

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"Workflow failed: {e}", "3")

    async def _is_error_analysis_request(self, prompt: str) -> bool:
        """Determine if the prompt is requesting error analysis"""
        error_keywords = [
            'error', 'fix', 'debug', 'problem', 'issue', 'failed',
            'exception', 'traceback', 'crash', 'broken', 'not working'
        ]
        return any(keyword in prompt.lower() for keyword in error_keywords)

    async def _handle_error_analysis(self, user_prompt: str, conversation_context: List[Dict]):
        """Handle error analysis and fixing requests"""
        self.detailed_log_event.emit("WorkflowEngine", "stage_start", "🔍 Analyzing error for fixes", "1")

        # Get error context from the application
        app = self._get_ava_application()
        if not app or not app.last_error_context:
            await self._send_response(
                "No recent error context available. Please run the project first to capture errors.")
            return

        error_context = app.last_error_context

        # Determine error type and select appropriate prompt
        error_type = self._classify_error(error_context)
        prompt_template = self.error_analysis_prompts.get(error_type, self.error_analysis_prompts["general_error"])

        # Gather project context
        project_files = self._get_project_file_list(app.current_project_path)
        file_states_info = self._format_file_states(error_context.file_states)

        # Format the analysis prompt
        analysis_prompt = prompt_template.format(
            error_context=error_context.get_error_summary(),
            file_list=project_files,
            file_states=file_states_info
        )

        self.detailed_log_event.emit("WorkflowEngine", "ai_query", f"Sending error analysis to AI: {error_type}", "1")

        # Get AI analysis and fixes
        try:
            response = await self.llm_client.generate_response(
                analysis_prompt,
                model=app.current_config.get("chat_model", "gemini-2.5-pro-preview-06-05"),
                temperature=0.3  # Lower temperature for more precise fixes
            )

            # Process the response and extract fixes
            await self._process_error_fix_response(response, app.current_project_path)

        except Exception as e:
            self.logger.error(f"Error analysis failed: {e}")
            await self._send_response(f"Error analysis failed: {e}")

    def _classify_error(self, error_context) -> str:
        """Classify the type of error for appropriate handling"""
        stderr = error_context.stderr.lower()

        if "syntaxerror" in stderr or "invalid syntax" in stderr:
            return "syntax_error"
        elif "modulenotfounderror" in stderr or "importerror" in stderr or "no module named" in stderr:
            return "import_error"
        elif any(keyword in stderr for keyword in ["exception", "error", "traceback"]):
            return "runtime_error"
        else:
            return "general_error"

    def _get_project_file_list(self, project_path: Path) -> str:
        """Get a formatted list of project files"""
        if not project_path.exists():
            return "Project directory not found"

        try:
            files = []
            for file_path in project_path.rglob("*.py"):
                rel_path = file_path.relative_to(project_path)
                files.append(f"  - {rel_path}")

            if not files:
                files.append("  - No Python files found")

            # Also check for requirements.txt and other important files
            important_files = ["requirements.txt", "setup.py", "pyproject.toml", "README.md"]
            for important_file in important_files:
                if (project_path / important_file).exists():
                    files.append(f"  - {important_file}")

            return "\n".join(files)
        except Exception as e:
            return f"Error listing files: {e}"

    def _format_file_states(self, file_states: Dict[str, float]) -> str:
        """Format file states information for AI context"""
        if not file_states:
            return "No file state information available"

        states = []
        for file_path, mtime in file_states.items():
            # Convert timestamp to readable format
            try:
                time_str = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
                states.append(f"  - {file_path} (modified: {time_str})")
            except Exception:
                states.append(f"  - {file_path}")

        return "\n".join(states)

    async def _process_error_fix_response(self, ai_response: str, project_path: Path):
        """Process AI response containing error fixes and apply them"""
        self.detailed_log_event.emit("WorkflowEngine", "stage_start", "📝 Processing error fixes", "1")

        # Try to extract file contents from the AI response
        fixes_applied = 0

        # Look for code blocks in the response
        import re

        # Pattern to match file headers and code blocks
        file_pattern = r'(?:```python\s*#\s*(.+?)```|```python\s*(.+?)```|(?:^|\n)([a-zA-Z_][a-zA-Z0-9_]*\.py):?\s*```python\s*(.+?)```)'

        # Also look for explicit file mentions
        file_mentions = re.findall(r'(?:^|\n)([a-zA-Z_][a-zA-Z0-9_]*\.py):?\s*```(?:python)?\s*(.+?)```', ai_response,
                                   re.DOTALL | re.MULTILINE)

        for file_match in file_mentions:
            filename, code_content = file_match
            if filename and code_content:
                try:
                    # Clean up the code content
                    code_content = code_content.strip()

                    # Write the fixed file
                    file_path = project_path / filename
                    file_path.write_text(code_content, encoding='utf-8')

                    self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Fixed file: {filename}", "1")
                    self.file_generated.emit(str(file_path))
                    fixes_applied += 1

                except Exception as e:
                    self.logger.error(f"Failed to write fixed file {filename}: {e}")

        # Look for requirements.txt updates
        req_pattern = r'requirements\.txt:?\s*```(?:txt|text)?\s*(.+?)```'
        req_matches = re.findall(req_pattern, ai_response, re.DOTALL)

        for req_content in req_matches:
            try:
                req_file = project_path / "requirements.txt"
                req_file.write_text(req_content.strip(), encoding='utf-8')
                self.detailed_log_event.emit("WorkflowEngine", "file_op", "Updated requirements.txt", "1")
                fixes_applied += 1
            except Exception as e:
                self.logger.error(f"Failed to update requirements.txt: {e}")

        # If no explicit files were found, try to parse the response differently
        if fixes_applied == 0:
            # Look for any Python code blocks and try to determine filename from context
            code_blocks = re.findall(r'```python\s*(.+?)```', ai_response, re.DOTALL)

            if code_blocks:
                # If there's only one code block and a main.py exists, assume it's main.py
                main_py = project_path / "main.py"
                if len(code_blocks) == 1 and main_py.exists():
                    try:
                        main_py.write_text(code_blocks[0].strip(), encoding='utf-8')
                        self.detailed_log_event.emit("WorkflowEngine", "file_op", "Fixed main.py", "1")
                        self.file_generated.emit(str(main_py))
                        fixes_applied += 1
                    except Exception as e:
                        self.logger.error(f"Failed to write main.py: {e}")

        # Send status message
        if fixes_applied > 0:
            status_msg = f"✅ Applied {fixes_applied} fix(es) to your project files!"
            if fixes_applied > 1:
                status_msg += "\n💡 Try running the project again to test the fixes."
            else:
                status_msg += "\n💡 Try running the project again to test the fix."
        else:
            status_msg = "⚠️ Couldn't automatically apply fixes. Please review the analysis above and apply changes manually."

        await self._send_response(f"{ai_response}\n\n{status_msg}")

    async def _execute_standard_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        """Execute standard workflow for non-error requests"""
        self.logger.info(f"🚀 Starting V6 workflow: {user_prompt[:100]}...")
        workflow_start_time = datetime.now()
        self.workflow_started.emit(user_prompt)
        try:
            is_modification = self.project_state_manager is not None and self.current_tech_spec is not None

            # --- PHASE 1: GET TECH SPEC ---
            tech_spec = self.current_tech_spec if is_modification else await self.architect_service.create_tech_spec(
                user_prompt, conversation_context)
            if not tech_spec or 'technical_specs' not in tech_spec:
                raise Exception("Architecture phase failed. No valid technical specification available.")

            # --- PHASE 2: SETUP PROJECT DIRECTORY ---
            project_name = tech_spec.get("project_name", "ai-project").replace(" ", "-")
            new_project_dir_name = f"{project_name}-mod-{datetime.now().strftime('%Y%m%d_%H%M%S')}" if is_modification else f"{project_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            project_dir = Path("./workspace") / new_project_dir_name
            project_dir.mkdir(parents=True, exist_ok=True)

            # --- NEW: WRITE REQUIREMENTS.TXT ---
            requirements = tech_spec.get("requirements", [])
            if requirements:
                req_content = "\n".join(requirements)
                self._write_file(project_dir, "requirements.txt", req_content)
                self.detailed_log_event.emit("WorkflowEngine", "file_op", "Generated requirements.txt", "1")

            # --- PHASE 3: FILE HANDLING (MODIFICATION-AWARE) ---
            build_order = tech_spec.get("dependency_order", [])
            files_in_spec = list(tech_spec.get("technical_specs", {}).keys())
            files_to_actively_modify = self._get_files_to_modify(user_prompt,
                                                                 files_in_spec) if is_modification else files_in_spec

            knowledge_packets = {}
            results = {"files_created": [], "project_dir": str(project_dir), "failed_files": []}

            self.detailed_log_event.emit("WorkflowEngine", "info",
                                         f"Files targeted for modification: {files_to_actively_modify}", "1")

            for filename in build_order:
                file_spec = tech_spec["technical_specs"].get(filename)
                if not file_spec:
                    continue

                # Determine if we should generate this file or copy it
                if filename in files_to_actively_modify:
                    # GENERATE the modified file
                    self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                                 f"🧬 Generating modified file: {filename}", "1")
                    dependency_context = self._build_dependency_context(file_spec.get("dependencies", []),
                                                                        knowledge_packets)
                    project_context = {"description": tech_spec.get("project_description", "")}
                    generated_code = await self.coder_service.generate_file_from_spec(filename, file_spec,
                                                                                      project_context,
                                                                                      dependency_context)

                    if "# FALLBACK" in generated_code:
                        results["failed_files"].append(filename)
                        continue

                    self._write_file(project_dir, filename, generated_code)
                    knowledge_packets[filename] = {"spec": file_spec, "source_code": generated_code}
                elif is_modification:
                    # COPY the original, unchanged file
                    original_file_path = self.project_state_manager.project_root / filename
                    if original_file_path.exists():
                        destination_path = project_dir / filename
                        shutil.copy2(original_file_path, destination_path)
                        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Copied unchanged file: {filename}",
                                                     "1")
                        knowledge_packets[filename] = {"spec": file_spec, "source_code": original_file_path.read_text()}
                    else:
                        self.logger.warning(f"Wanted to copy {filename} but it doesn't exist in the source project.")
                # (The case for a new project is handled by the initial `if` block)

                results["files_created"].append(filename)

            # --- PHASE 4: FINALIZATION ---
            final_result = await self._finalize_project(results, (datetime.now() - workflow_start_time).total_seconds())
            self.workflow_completed.emit(final_result)
            return final_result
        except Exception as e:
            self.logger.error(f"❌ AI Workflow failed: {e}", exc_info=True)
            self.workflow_completed.emit({"success": False, "error": str(e),
                                          "elapsed_time": (datetime.now() - workflow_start_time).total_seconds()})
            raise

    def _build_dependency_context(self, dependency_files: List[str], knowledge_packets: Dict[str, Dict]) -> str:
        if not dependency_files:
            return "This file has no dependencies."
        context_str = ""
        for dep_file in dependency_files:
            if dep_file in knowledge_packets:
                packet = knowledge_packets[dep_file]
                context_str += f"\n\n--- CONTEXT FOR DEPENDENCY: {dep_file} ---\n"
                context_str += f"SPECIFICATION:\n```json\n{json.dumps(packet['spec'], indent=2)}\n```\n"
                context_str += f"FULL SOURCE CODE:\n```python\n{packet['source_code']}\n```\n"
            else:
                self.logger.warning(
                    f"Dependency '{dep_file}' was not found in knowledge packets. Context will be incomplete.")
        return context_str

    def _write_file(self, project_dir: Path, filename: str, content: str):
        file_path_obj = project_dir / filename
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_path_obj.write_text(content, encoding='utf-8')
        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"File written/overwritten: {file_path_obj}", "2")
        self.file_generated.emit(str(file_path_obj))

    async def _finalize_project(self, results: Dict[str, Any], elapsed_time: float) -> Dict[str, Any]:
        project_dir = results.get("project_dir")
        if project_dir:
            self.project_loaded.emit(project_dir)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project loaded into UI: {project_dir}", "1")
        return {"success": len(results.get("failed_files", [])) == 0,
                "project_name": Path(project_dir).name if project_dir else "Unknown",
                "project_dir": project_dir,
                "num_files": len(results.get("files_created", [])),
                "files_created": results.get("files_created", []),
                "failed_files": results.get("failed_files", []),
                "elapsed_time": elapsed_time,
                "strategy": "V6 Surgical Modification with Error Analysis"}

    async def _send_response(self, message: str):
        """Send response back to the chat interface"""
        app = self._get_ava_application()
        if app:
            app.chat_message_received.emit(message)

    def _get_ava_application(self):
        """Get reference to the main AvA application"""
        # This assumes the application is accessible through some mechanism
        # You might need to adjust this based on your architecture
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if hasattr(app, 'ava_app'):
            return app.ava_app
        return None