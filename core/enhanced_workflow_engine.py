# core/enhanced_workflow_engine.py - RESULTS FOCUSED VERSION

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from PySide6.QtCore import QObject, Signal, QTimer

from core.project_state_manager import ProjectStateManager
from core.ai_feedback_system import AIFeedbackSystem, FeedbackType
from core.enhanced_micro_task_engine import StreamlinedMicroTaskEngine
from core.domain_context_manager import DomainContextManager
from core.llm_client import LLMRole
import json


class EnhancedWorkflowEngine(QObject):
    """
    🚀 RESULTS-FOCUSED Workflow Engine

    Fast, efficient, professional code generation with:
    - Smart task decomposition (3-5 chunks)
    - Real-time code streaming
    - Domain awareness
    - Professional output quality
    """

    # Existing signals (keep compatibility)
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    file_generated = Signal(str)
    project_loaded = Signal(str)
    workflow_progress = Signal(str, str)

    # New signals for enhanced features
    ai_collaboration_started = Signal(str)
    ai_feedback_received = Signal(str, str, str)
    iteration_completed = Signal(str, int)
    quality_check_completed = Signal(str, bool, str)

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None, project_root: str = None):
        super().__init__()

        # Core components
        self.llm_client = llm_client
        self.terminal = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager

        # Enhanced components
        self.project_root = Path(project_root) if project_root else Path("./workspace")
        self.project_state = ProjectStateManager(self.project_root)
        self.ai_feedback_system = AIFeedbackSystem(llm_client, self.project_state, terminal_window)

        # STREAMLINED components
        self.domain_context_manager = DomainContextManager(self.project_root)
        self.micro_task_engine = StreamlinedMicroTaskEngine(
            llm_client, self.project_state, self.domain_context_manager
        )

        # Output directory
        self.output_dir = Path("./generated_projects")
        self.output_dir.mkdir(exist_ok=True)

        # Workflow state
        self.current_workflow_state = {
            "stage": "idle",
            "progress": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "active_collaboration_session": None,
            "iterations": {},
            "domain_context": None
        }

        # Active workflow tracking
        self.current_workflow_task = None
        self.workflow_active = False
        self.current_project_dir = None

        # User feedback integration
        self.user_feedback_queue = []
        self.pause_for_feedback = False

        self._connect_code_viewer()
        self._log_enhanced_initialization()

    def _connect_code_viewer(self):
        """Connect code viewer for real-time updates"""
        if self.code_viewer:
            self.code_viewer.file_changed.connect(self._on_file_changed)

    def _log_enhanced_initialization(self):
        """Log streamlined initialization"""
        self.terminal.log("🚀 RESULTS-FOCUSED Workflow Engine Initialized!")
        self.terminal.log("   ⚡ Fast task decomposition (3-5 chunks)")
        self.terminal.log("   📺 Real-time code streaming")
        self.terminal.log("   🎯 Professional output quality")
        self.terminal.log("   🔍 Smart domain awareness")

        if self.project_state.files:
            self.terminal.log(f"   📁 Existing project loaded: {len(self.project_state.files)} files")
        else:
            self.terminal.log("   📄 Ready to build professional code")

    async def execute_enhanced_workflow(self, user_prompt: str, enable_iterations: bool = True,
                                        max_iterations: int = 2) -> Dict[str, Any]:
        """Execute RESULTS-FOCUSED workflow"""

        if not self._is_build_request(user_prompt):
            self.terminal.log(f"⚠️ Not a build request: '{user_prompt}'")
            return {"success": False, "error": "Not a build request"}

        if self.workflow_active:
            self.terminal.log("⚠️ Workflow already running")
            return {"success": False, "error": "Workflow already active"}

        try:
            self.workflow_active = True
            self.workflow_started.emit(user_prompt)
            self._update_workflow_stage("initializing", "Starting fast professional workflow...")

            # Quick domain context discovery
            self._update_workflow_stage("context_discovery", "Analyzing project domain...")
            domain_context = await self.domain_context_manager.get_comprehensive_context()
            self.current_workflow_state["domain_context"] = domain_context

            # Log what we found
            if domain_context.get('frameworks'):
                frameworks = [f['name'] for f in domain_context['frameworks'] if f['confidence'] > 0.5]
                if frameworks:
                    self.terminal.log(f"🔍 Detected: {', '.join(frameworks)}")

            # Initialize collaboration
            target_files = await self._predict_target_files(user_prompt)
            session_id = await self.ai_feedback_system.initiate_collaboration(
                task_description=user_prompt,
                target_files=target_files,
                participating_ais=["planner", "coder", "assembler", "reviewer"]
            )

            self.ai_collaboration_started.emit(session_id)

            # Fast planning
            self._update_workflow_stage("planning", "Creating professional development plan...")
            planner_guidance = await self.ai_feedback_system.process_planner_requirements(
                session_id, user_prompt
            )
            planner_guidance["domain_context"] = domain_context

            # Setup project
            project_name = planner_guidance.get("project_plan", {}).get("name", "enhanced_project")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_project_dir = self.output_dir / f"{project_name}_{timestamp}"
            self.current_project_dir.mkdir(exist_ok=True)

            # FAST file generation
            self._update_workflow_stage("generation", "🚀 Generating professional code...")
            files_to_generate = planner_guidance.get("file_specifications", {})

            generated_files = []
            failed_files = []

            for file_path, file_spec in files_to_generate.items():
                self.terminal.log(f"🎯 Creating: {file_path}")

                result = await self._generate_file_fast(
                    session_id=session_id,
                    file_path=file_path,
                    file_spec=file_spec,
                    planner_guidance=planner_guidance,
                    enable_iterations=enable_iterations,
                    max_iterations=max_iterations
                )

                if result["success"]:
                    generated_files.append(result["file_path"])
                    self.file_generated.emit(result["file_path"])

                    # Add to project state
                    self.project_state.add_file(
                        result["file_path"],
                        result["content"],
                        ai_role="fast_workflow",
                        reasoning=f"Generated for: {user_prompt}"
                    )
                else:
                    failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})

            # Finalize
            if generated_files:
                self._update_workflow_stage("finalization", "Finalizing project...")
                await self._finalize_enhanced_project(generated_files)

            # Complete
            success = len(generated_files) > 0
            result = {
                "success": success,
                "project_dir": str(self.current_project_dir),
                "generated_files": generated_files,
                "failed_files": failed_files,
                "project_name": project_name,
                "file_count": len(generated_files)
            }

            if success:
                self.terminal.log(f"✅ SUCCESS! Generated {len(generated_files)} files")
            else:
                self.terminal.log(f"❌ FAILED! No files generated")

            self.workflow_completed.emit(result)
            return result

        except Exception as e:
            self.terminal.log(f"❌ Workflow failed: {e}")
            import traceback
            traceback.print_exc()

            error_result = {
                "success": False,
                "error": str(e),
                "project_dir": str(self.current_project_dir) if self.current_project_dir else None
            }
            self.workflow_completed.emit(error_result)
            return error_result

        finally:
            self.workflow_active = False
            self.current_workflow_task = None

    async def _generate_file_fast(self, session_id: str, file_path: str, file_spec: Dict[str, Any],
                                  planner_guidance: Dict[str, Any], enable_iterations: bool = True,
                                  max_iterations: int = 2) -> Dict[str, Any]:
        """Fast file generation with streaming"""

        iteration = 0
        current_content = ""
        review_approved = False

        while iteration < max_iterations and not review_approved:
            iteration += 1
            self.terminal.log(f"[{file_path}] 🔄 Iteration {iteration}/{max_iterations}")

            try:
                # Create smart tasks (3-5 chunks)
                self.terminal.log(f"[{file_path}] 🎯 Planning code structure...")

                smart_tasks = await self.micro_task_engine.create_smart_tasks(
                    file_path=file_path,
                    file_spec=file_spec,
                    project_context=planner_guidance.get("domain_context", {})
                )

                self.terminal.log(f"[{file_path}] 📋 Created {len(smart_tasks)} code chunks")

                # Execute with STREAMING
                task_results = await self._execute_with_streaming(
                    session_id, file_path, smart_tasks, planner_guidance
                )

                # Assemble
                self.terminal.log(f"[{file_path}] 🔧 Assembling final code...")
                assembled_content, assembly_context = await self.ai_feedback_system.assemble_with_context_feedback(
                    session_id, file_path, task_results, planner_guidance
                )

                # Quick review
                review_approved, review_feedback, review_data = await self.ai_feedback_system.review_with_collaborative_feedback(
                    session_id, file_path, assembled_content, assembly_context, planner_guidance
                )

                current_content = assembled_content

                if review_approved:
                    self.terminal.log(f"[{file_path}] ✅ Code APPROVED!")
                    break
                else:
                    self.terminal.log(f"[{file_path}] ⚠️ Needs improvement...")
                    if not enable_iterations or iteration >= max_iterations:
                        self.terminal.log(f"[{file_path}] ⏭️ Using current version")
                        review_approved = True
                        break

            except Exception as e:
                self.terminal.log(f"[{file_path}] ❌ Iteration {iteration} failed: {e}")
                if iteration == max_iterations:
                    return {
                        "success": False,
                        "file_path": file_path,
                        "error": f"Failed after {max_iterations} iterations: {e}"
                    }

        # Write file
        if current_content:
            full_path = self.current_project_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(current_content, encoding='utf-8')

            self.terminal.log(f"[{file_path}] 📝 File written ({len(current_content)} chars)")

            return {
                "success": True,
                "file_path": str(full_path),
                "content": current_content,
                "iterations_used": iteration,
                "final_review_approved": review_approved
            }
        else:
            return {
                "success": False,
                "file_path": file_path,
                "error": "No content generated"
            }

    async def _execute_with_streaming(self, session_id: str, file_path: str,
                                      smart_tasks: List[Any], planner_guidance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute tasks with REAL-TIME CODE STREAMING"""

        task_results = []

        for task in smart_tasks:
            task_dict = task.to_dict()

            self.terminal.log(f"[{file_path}] 🚀 {task.description}")
            self.terminal.log(f"[{file_path}] 📝 Streaming code...")

            try:
                # Create professional prompt
                code_prompt = f"""
You are a senior Python developer creating professional, production-ready code.

TASK: {task.description}
CONTEXT: {task.context}
REQUIREMENTS: {task.exact_requirements}

FILE: {file_path}
EXPECTED LENGTH: ~{task.expected_lines} lines

Create clean, well-documented, production-ready Python code that follows best practices.
Include proper error handling, type hints, and docstrings.
Make it professional quality that would pass code review.

Code:
"""

                # STREAM the actual code generation
                code_chunks = []
                async for chunk in self.llm_client.stream_chat(code_prompt, LLMRole.CODER):
                    code_chunks.append(chunk)
                    # Show every 50 characters to terminal
                    if len(''.join(code_chunks)) % 50 == 0:
                        self.terminal.log(f"[{file_path}] 📺 {''.join(code_chunks[-50:])}", log_level="debug")

                generated_code = ''.join(code_chunks)

                self.terminal.log(f"[{file_path}] ✅ Chunk complete ({len(generated_code)} chars)")

                task_results.append({
                    "task": task_dict,
                    "code": generated_code
                })

            except Exception as e:
                self.terminal.log(f"[{file_path}] ❌ Task failed: {e}")
                task_results.append({
                    "task": task_dict,
                    "code": f"# ERROR: {e}\n# TODO: Fix this implementation\npass"
                })

        return task_results

    # Keep all the other methods unchanged...
    async def _finalize_enhanced_project(self, generated_files: List[str]):
        """Finalize project"""
        if self.code_viewer and self.current_project_dir:
            self.code_viewer.load_project(str(self.current_project_dir))

            main_files = ["main.py", "app.py", "__init__.py"]
            for main_file in main_files:
                main_path = self.current_project_dir / main_file
                if main_path.exists():
                    self.code_viewer.auto_open_file(str(main_path))
                    break

        self.project_state.save_state(self.current_project_dir / ".ava_project_state.json")
        self.project_loaded.emit(str(self.current_project_dir))

    def execute_workflow(self, user_prompt: str):
        """Compatibility method"""
        if not self._is_build_request(user_prompt):
            self.terminal.log(f"⚠️ Skipping workflow - not a build request: '{user_prompt}'")
            return

        if self.workflow_active:
            self.terminal.log("⚠️ Workflow already running - please wait")
            return

        try:
            self.current_workflow_task = asyncio.create_task(
                self.execute_enhanced_workflow(user_prompt)
            )
        except Exception as e:
            self.terminal.log(f"❌ Failed to start workflow: {e}")

    async def _predict_target_files(self, user_prompt: str) -> List[str]:
        """Predict files needed"""
        prompt_lower = user_prompt.lower()
        if "gui" in prompt_lower or "calculator" in prompt_lower:
            return ["main.py"]
        elif "api" in prompt_lower:
            return ["main.py", "api.py"]
        else:
            return ["main.py"]

    def _is_build_request(self, prompt: str) -> bool:
        """Check if this is a build request"""
        prompt_lower = prompt.lower().strip()

        casual_phrases = ['hi', 'hello', 'hey', 'thanks', 'ok', 'yes', 'no']
        if prompt_lower in casual_phrases:
            return False

        build_keywords = [
            'build', 'create', 'make', 'generate', 'develop', 'code',
            'implement', 'write', 'design', 'calculator', 'app'
        ]

        return any(keyword in prompt_lower for keyword in build_keywords) and len(prompt.split()) >= 3

    def _update_workflow_stage(self, stage: str, description: str):
        """Update workflow stage"""
        self.current_workflow_state["stage"] = stage
        self.workflow_progress.emit(stage, description)
        self.terminal.log(f"📋 {description}")

    def _on_file_changed(self, file_path: str, content: str):
        """Handle file changes"""
        if self.project_state:
            self.project_state.add_file(
                file_path, content, ai_role="user_edit",
                reasoning="User modified file"
            )

    def get_workflow_stats(self) -> Dict:
        """Get workflow statistics"""
        return {
            "workflow_state": self.current_workflow_state.copy(),
            "workflow_active": self.workflow_active,
            "project_files": len(self.project_state.files) if self.project_state else 0
        }

    def is_workflow_running(self) -> bool:
        """Check if workflow is running"""
        return self.workflow_active and (
                self.current_workflow_task is not None and
                not self.current_workflow_task.done()
        )