# enhanced_workflow_engine.py - COMPLETE WORKING FILE with Role-Based LLM

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import tempfile
import shutil
import json
import traceback

from PySide6.QtCore import QObject, Signal, QTimer, QThread

from core.llm_client import LLMRole
# Import services from their new locations
from core.workflow_services import EnhancedPlannerService, EnhancedCoderService, EnhancedAssemblerService


class EnhancedWorkflowEngine(QObject):
    """
    🚀 Enhanced Workflow Engine with Role-Based AI Code Generation
    """
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    workflow_progress = Signal(str, str)  # stage, description (overall stage)
    file_generated = Signal(str)
    project_loaded = Signal(str)

    # RENAMED: This is the primary signal for detailed, hierarchical logging
    # agent_name (e.g., "Planner", "Coder"), type_key ("thought", "code_chunk", "status", "error"), content, indent_level_str
    detailed_log_event = Signal(str, str, str, str)

    task_progress = Signal(int, int)

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.terminal_window = terminal_window  # For direct access if needed, though signals are preferred
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

        self._debug_llm_client()

        self.workflow_stats = {
            "cache_stats": {"cache_size": 0, "hit_rate": 0.0},
            "workflow_state": {"stage": "idle", "completed_tasks": 0, "total_tasks": 0}
        }

        # --- Service Instantiation with Emitter ---
        # The emitter function that services will call
        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.planner_service = EnhancedPlannerService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = EnhancedCoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.assembler_service = EnhancedAssemblerService(self.llm_client, service_log_emitter, self.rag_manager)
        # self.orchestrator = WorkflowOrchestrator(self.planner_service, self.coder_service, self.assembler_service, service_log_emitter)

        self._connect_terminal_signals()
        self.logger.info("✅ Enhanced Workflow Engine initialized with Role-Based AI and Service Emitters")

    def _debug_llm_client(self):
        try:
            self.logger.info("🔍 DEBUGGING LLM CLIENT...")
            if not self.llm_client: self.logger.error("❌ LLM CLIENT IS NONE!"); return
            self.logger.info(f"✅ LLM Client exists: {type(self.llm_client)}")
            methods = [m for m in dir(self.llm_client) if not m.startswith('_')]
            self.logger.info(f"📋 LLM Client methods: {methods}")
            if hasattr(self.llm_client, 'chat'):
                self.logger.info("✅ chat method exists")
            else:
                self.logger.error("❌ chat method MISSING!")
            if hasattr(self.llm_client, 'get_available_models'):
                models = self.llm_client.get_available_models()
                self.logger.info(f"📊 Available models: {models}")
                if not models or models == ["No LLM services available"]:
                    self.logger.error("❌ NO LLM MODELS AVAILABLE!")
                else:
                    self.logger.info("✅ LLM models are available")
            else:
                self.logger.warning("⚠️ get_available_models method missing")
        except Exception as e:
            self.logger.error(f"❌ LLM DEBUG FAILED: {e}\n{traceback.format_exc()}")

    def _connect_terminal_signals(self):
        if not self.terminal_window: return
        try:
            # Overall stage progress
            self.workflow_progress.connect(self.terminal_window.update_workflow_progress)
            # Overall task count progress
            self.task_progress.connect(self.terminal_window.update_task_progress)

            # Detailed hierarchical logging
            # The terminal's stream_log method will need to handle these args
            if hasattr(self.terminal_window, 'stream_log_rich'):  # Expecting a new method for rich logs
                self.detailed_log_event.connect(self.terminal_window.stream_log_rich)
            elif hasattr(self.terminal_window, 'stream_log'):  # Fallback to simpler stream_log
                # Adapt the signal to what stream_log expects if it's simpler
                # For now, assume stream_log_rich or direct handling in terminal
                self.detailed_log_event.connect(
                    lambda agent, type_key, content, indent:
                    self.terminal_window.stream_log(f"[{agent}|{type_key}] {content}", indent=int(indent))
                )
                self.logger.warning("Terminal does not have stream_log_rich, using fallback stream_log.")
            else:
                self.logger.error("Terminal does not have a suitable method to connect detailed_log_event.")

            # File generation status (for indicators, etc.)
            # self.file_progress.connect(self._handle_file_progress) # Assuming you have this signal & handler

            self.logger.info("✅ Terminal signals connected for streaming")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect terminal signals: {e}")

    # Example _handle_file_progress if you add a file_progress signal
    # def _handle_file_progress(self, file_path: str, status: str):
    #     if hasattr(self.terminal_window, 'start_file_generation') and hasattr(self.terminal_window, 'complete_file_generation'):
    #         if status == "started": self.terminal_window.start_file_generation(file_path)
    #         elif status in ["completed", "error"]: self.terminal_window.complete_file_generation(file_path, status == "completed")

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        self.logger.info(f"🚀 Starting Role-Based AI workflow: {user_prompt[:100]}...")
        self.workflow_stats["workflow_state"] = {"stage": "initializing", "completed_tasks": 0, "total_tasks": 6}

        try:
            self.workflow_started.emit(user_prompt)
            # Emit detailed log for workflow start
            self.detailed_log_event.emit("WorkflowEngine", "stage_start", "🚀 Initializing AvA AI workflow system", "0")

            # Stage 1: Context Discovery (Engine's own LLM call)
            self.workflow_progress.emit("context_discovery", "Analyzing project requirements with AI")
            self.detailed_log_event.emit("WorkflowEngine", "stage_detail",
                                         "🔍 AI analyzing project context and requirements", "0")
            context_info = await self._discover_context_with_ai(user_prompt, conversation_context)
            self._update_task_progress(1, 6)

            # Stage 2: Planning (Engine's own LLM call)
            self.workflow_progress.emit("planning", "Creating intelligent development plan")
            self.detailed_log_event.emit("WorkflowEngine", "stage_detail", "🧠 AI creating intelligent development plan",
                                         "0")
            plan = await self._create_ai_development_plan(user_prompt, context_info)
            self._update_task_progress(2, 6)

            # Stage 3: Architecture Design (Can be simple or LLM-based)
            self.workflow_progress.emit("design", "Designing project architecture")
            self.detailed_log_event.emit("WorkflowEngine", "stage_detail", "🏗️ Designing project architecture", "0")
            architecture = await self._design_project_architecture(user_prompt, plan)
            self._update_task_progress(3, 6)

            # Stage 4: Code Generation (Orchestrated, services will emit their own detailed logs)
            self.workflow_progress.emit("generation", "Generating code with AI assistance")
            self.detailed_log_event.emit("WorkflowEngine", "stage_detail",
                                         "⚡ AI generating professional code (orchestrating services)", "0")

            # --- Code Generation Phase using Services ---
            results = {"files_created": [], "project_dir": None}
            project_name = plan.get("project_name", "ai_project")
            output_dir = Path("./workspace")
            project_dir = output_dir / f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            project_dir.mkdir(parents=True, exist_ok=True)
            results["project_dir"] = str(project_dir)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project directory created: {project_dir}", "1")

            files_to_create = []
            if 'files' in plan and isinstance(plan['files'], dict):  # New plan structure
                files_to_create = [{"filename": fn, **fi} for fn, fi in plan['files'].items()]
            elif 'files_to_create' in plan and isinstance(plan['files_to_create'], list):  # Old plan structure
                files_to_create = plan['files_to_create']

            # Ensure ContextCache is defined or imported correctly if used
            # For simplicity, if ContextCache is complex, pass None or a dummy for now.
            # from core.some_module import ContextCache # Or define it here
            class ContextCache:  # Dummy for now
                def get(self, key): return None

                def set(self, key, value): pass

            context_cache = ContextCache()

            for i, file_info_dict in enumerate(files_to_create):
                filename = file_info_dict.get("filename", f"file_{i}.py")
                self.detailed_log_event.emit("WorkflowEngine", "status",
                                             f"Processing file: {filename} ({i + 1}/{len(files_to_create)})", "1")

                micro_tasks = await self.planner_service.create_micro_tasks(filename, plan, context_cache)

                def coder_progress_update(msg: str):
                    self.detailed_log_event.emit("WorkflowEngine", "status", f"Coder Update for {filename}: {msg}", "2")

                task_results = await self.coder_service.execute_tasks_parallel(micro_tasks, f"Context for {filename}",
                                                                               context_cache, coder_progress_update)

                assembled_code, review_ok, review_msg = await self.assembler_service.assemble_file(filename,
                                                                                                   task_results, plan,
                                                                                                   context_cache)

                file_path_obj = project_dir / filename
                file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                file_path_obj.write_text(assembled_code, encoding='utf-8')
                self.detailed_log_event.emit("WorkflowEngine", "file_op",
                                             f"File written: {file_path_obj} (Review: {'OK' if review_ok else 'Needs Attention'})",
                                             "2")
                self.file_generated.emit(str(file_path_obj))
                results["files_created"].append(filename)
                # Spread task 4 & 5 over file generation
                # Iteration i goes from 0 to len(files_to_create)-1
                # Progress for tasks 4 and 5 should span from just after task 3 (architecture) up to task 5 completion
                # Task 3 completion is at progress point 3.
                # Tasks 4 & 5 span from progress point 3 to 5 (a total of 2 progress units)
                # For each file, it contributes a fraction of these 2 units.
                if files_to_create:  # Avoid division by zero
                    progress_for_current_file = (2.0 * (i + 1)) / len(files_to_create)
                    self._update_task_progress(3 + int(progress_for_current_file), 6)

            # --- End Code Generation Phase ---
            # Ensure task progress for generation phase is marked as 5 if all files processed
            if files_to_create:
                self._update_task_progress(5, 6)  # Mark generation phase as complete (tasks 4&5)
            else:  # If no files, jump to task 5 directly
                self._update_task_progress(5, 6)

            # Stage 5: Finalization
            self.workflow_progress.emit("finalization", "Finalizing project")
            self.detailed_log_event.emit("WorkflowEngine", "stage_detail", "📄 Finalizing AI-generated project", "0")
            final_result = await self._finalize_project(results)
            self._update_task_progress(6, 6)

            self.workflow_progress.emit("complete", "AI workflow completed successfully")
            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ AI workflow completed successfully!", "0")
            self.workflow_completed.emit(final_result)

            self.logger.info("✅ Enhanced AI workflow completed successfully")
            return final_result

        except Exception as e:
            self.logger.error(f"❌ AI Workflow failed: {e}", exc_info=True)
            self.workflow_progress.emit("error", f"AI workflow failed: {str(e)}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Workflow Error: {str(e)}", "0")
            self.detailed_log_event.emit("WorkflowEngine", "debug", traceback.format_exc(), "1")
            self.workflow_completed.emit({"success": False, "error": str(e)})
            raise

    def _update_task_progress(self, completed: int, total: int):
        self.workflow_stats["workflow_state"]["completed_tasks"] = completed
        self.workflow_stats["workflow_state"]["total_tasks"] = total
        self.task_progress.emit(completed, total)

    async def _discover_context_with_ai(self, prompt: str, conversation_context: List[Dict]) -> Dict[str, Any]:
        self.detailed_log_event.emit("WorkflowEngine", "thought",
                                     "Analyzing user requirements for context discovery...", "1")
        analysis_prompt = f"Analyze this development request: \"{prompt}\". Provide JSON: {{\"project_type\": \"app_type\", \"main_technology\": \"tech\", \"frameworks\": [], \"complexity\": \"level\", \"key_features\": [], \"estimated_files\": N}}"

        try:
            ai_response_text = await self._call_llm_async(analysis_prompt, LLMRole.PLANNER, "ContextAnalyzer", 1)
            start_idx = ai_response_text.find('{')
            end_idx = ai_response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                ai_analysis = json.loads(ai_response_text[start_idx:end_idx])
                self.detailed_log_event.emit("WorkflowEngine", "success", "AI context analysis complete.", "1")
                # Ensure all expected keys from the prompt are in ai_analysis or have defaults
                result_context = {
                    "project_type": ai_analysis.get("project_type", "desktop_app"),
                    "main_technology": ai_analysis.get("main_technology", "Python"),
                    "frameworks": ai_analysis.get("frameworks", ["PySide6"] if "gui" in prompt.lower() else []),
                    "complexity": ai_analysis.get("complexity", "moderate"),
                    "key_features": ai_analysis.get("key_features", []),
                    "estimated_files": ai_analysis.get("estimated_files", 1)
                }
                return {"prompt": prompt, "conversation_history": conversation_context or [],
                        "ai_analysis": ai_analysis, **result_context}
            else:
                self.detailed_log_event.emit("WorkflowEngine", "warning",
                                             "No valid JSON found in AI context analysis response.", "1")
                raise ValueError("No valid JSON found in context analysis response")
        except Exception as e:
            self.detailed_log_event.emit("WorkflowEngine", "warning",
                                         f"AI context analysis failed ({e}), using fallback.", "1")

        is_gui = "gui" in prompt.lower() or "pyside" in prompt.lower()
        return {"prompt": prompt, "conversation_history": conversation_context or [],
                "project_type": "desktop_app" if is_gui else "cli_tool", "main_technology": "Python",
                "frameworks": ["PySide6"] if is_gui else [], "complexity": "moderate", "key_features": [],
                "estimated_files": 1}

    async def _create_ai_development_plan(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.detailed_log_event.emit("WorkflowEngine", "thought", "Creating AI development strategy and plan...", "1")
        planning_prompt = f"""
Request: "{prompt}"
Project Type: {context.get('project_type', 'unknown')}
Complexity: {context.get('complexity', 'moderate')}
Detected Frameworks: {context.get('frameworks', [])}

Create a detailed JSON plan. For a calculator, suggest modular files like main.py, calculator_gui.py, calculator_logic.py.
The JSON should include: "project_name", "description", "architecture_type", "main_requirements", "files" (as a dictionary of filename: {{details}}), "dependencies", "execution_notes".
Example for "files": {{ "main.py": {{ "priority": 1, "description": "Main entry point and GUI setup for a PySide6 calculator", "key_features": ["QApplication init", "MainWindow instantiation", "Button grid layout", "Display field"] }} }}
"""

        try:
            ai_response_text = await self._call_llm_async(planning_prompt, LLMRole.PLANNER, "DevPlanner", 1)
            start_idx = ai_response_text.find('{')
            end_idx = ai_response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                ai_plan = json.loads(ai_response_text[start_idx:end_idx])
                self.detailed_log_event.emit("WorkflowEngine", "success", "AI development plan created.", "1")
                return ai_plan
            else:
                self.detailed_log_event.emit("WorkflowEngine", "warning",
                                             "No valid JSON found in AI planning response.", "1")
                raise ValueError("No valid JSON found in planning response")
        except Exception as e:
            self.detailed_log_event.emit("WorkflowEngine", "warning",
                                         f"AI planning failed ({e}), using smart fallback.", "1")

        if "calculator" in prompt.lower():
            return {"project_name": "calculator_app_fb", "description": "PySide6 Calculator App",
                    "architecture_type": "gui_calculator", "main_requirements": ["GUI", "Math ops"],
                    "files": {"main.py": {"priority": 1, "description": "GUI and logic"},
                              "calculator_logic.py": {"priority": 2, "description": "Core math functions"}},
                    "dependencies": ["PySide6"], "execution_notes": "Run main.py"}
        return {"project_name": "ai_generated_app_fb", "description": prompt[:50], "architecture_type": "generic_app",
                "main_requirements": ["Core logic"],
                "files": {"main.py": {"priority": 1, "description": "Main application"}}, "dependencies": ["Python"],
                "execution_notes": "Run main.py"}

    async def _design_project_architecture(self, prompt: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        self.detailed_log_event.emit("WorkflowEngine", "thought", "Designing system architecture (simplified)...", "1")
        await asyncio.sleep(0.1)
        architecture = {"pattern": plan.get("architecture_type", "modular"), "entry_point": "main.py",
                        "dependencies": plan.get("dependencies", ["Python"]), "structure": "Clean, modular design"}
        self.detailed_log_event.emit("WorkflowEngine", "success", "Architecture design complete.", "1")
        return architecture

    async def _call_llm_async(self, prompt: str, role: LLMRole, agent_name_for_log: str, indent_level: int) -> str:
        self.detailed_log_event.emit(agent_name_for_log, "thought", f"Preparing LLM request for role {role.value}...",
                                     str(indent_level))

        if not self.llm_client:
            self.detailed_log_event.emit(agent_name_for_log, "error", "LLM client is not available.", str(indent_level))
            raise Exception("LLM client is None - not initialized properly")

        self.detailed_log_event.emit(agent_name_for_log, "status", f"Awaiting LLM ({role.value})...",
                                     str(indent_level + 1))

        response_chunks = []
        stream_generator = None
        try:
            stream_generator = self.llm_client.stream_chat(prompt, role)
            chunk_idx = 0
            async for chunk in stream_generator:
                response_chunks.append(chunk)
                log_type = "code_chunk" if role == LLMRole.CODER else "llm_chunk"
                if role != LLMRole.CODER:
                    self.detailed_log_event.emit(agent_name_for_log, log_type, chunk, str(indent_level + 2))

                if chunk_idx % 10 == 0:
                    await asyncio.sleep(0.001)
                chunk_idx += 1
        except Exception as e:
            self.detailed_log_event.emit(agent_name_for_log, "error", f"LLM stream error: {e}", str(indent_level + 1))
            self.logger.error(f"LLM stream error during _call_llm_async for {agent_name_for_log} ({role.value}): {e}",
                              exc_info=True)
            raise
        finally:
            if stream_generator and hasattr(stream_generator, 'aclose'):
                try:
                    await stream_generator.aclose()
                except Exception:
                    pass

        response_text = ''.join(response_chunks)
        if not response_text.strip():
            self.detailed_log_event.emit(agent_name_for_log, "warning", "LLM returned an empty response.",
                                         str(indent_level + 1))
            raise Exception(f"LLM returned empty response for role {role.value}")

        self.detailed_log_event.emit(agent_name_for_log, "success",
                                     f"LLM response received (length: {len(response_text)}).", str(indent_level))
        return response_text

    async def _finalize_project(self, results: Dict[str, Any]) -> Dict[str, Any]:
        self.detailed_log_event.emit("WorkflowEngine", "thought", "Creating project summary...", "1")
        await asyncio.sleep(0.1)

        project_dir = results.get("project_dir")
        if project_dir:
            self.project_loaded.emit(project_dir)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project loaded into UI: {project_dir}", "1")

        final_result = {
            "success": True, "project_name": Path(project_dir).name if project_dir else "Unknown",
            "project_dir": project_dir, "file_count": len(results.get("files_created", [])),
            "files_created": results.get("files_created", [])
        }
        self.detailed_log_event.emit("WorkflowEngine", "success", "Project finalization complete.", "1")
        return final_result

    def get_workflow_stats(self) -> Dict[str, Any]:
        return self.workflow_stats.copy()

    def execute_workflow(self, prompt: str):  # Legacy compatibility
        asyncio.create_task(self.execute_enhanced_workflow(prompt, []))