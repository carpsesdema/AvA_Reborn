# core/workflow_engine.py - Enhanced with Conversation Context Support

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from PySide6.QtCore import QObject, Signal, QTimer

from core.workflow_services import (
    EnhancedPlannerService, EnhancedCoderService, WorkflowOrchestrator
)
from core.assembler_service import AssemblerService


class ContextCache:
    """Smart context cache for RAG results with ranking and pruning"""

    def __init__(self, max_cache_size: int = 100):
        self.cache: Dict[str, Dict] = {}
        self.access_count: Dict[str, int] = {}
        self.max_cache_size = max_cache_size

    def _generate_cache_key(self, query: str, context_type: str = "general") -> str:
        """Generate cache key from query and context type"""
        import hashlib
        combined = f"{context_type}:{query.lower()}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def get_context(self, query: str, context_type: str = "general"):
        """Get cached context if available"""
        cache_key = self._generate_cache_key(query, context_type)
        if cache_key in self.cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]["context"]
        return None

    def store_context(self, query: str, context: str, context_type: str = "general",
                      relevance_score: float = 1.0):
        """Store context in cache with relevance score"""
        if not context or len(context.strip()) < 10:
            return

        cache_key = self._generate_cache_key(query, context_type)

        if len(self.cache) >= self.max_cache_size:
            self._prune_cache()

        self.cache[cache_key] = {
            "context": context,
            "query": query,
            "context_type": context_type,
            "relevance_score": relevance_score,
            "timestamp": datetime.now()
        }
        self.access_count[cache_key] = 1

    def _prune_cache(self):
        """Remove least frequently accessed items"""
        if not self.cache:
            return
        sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) // 4
        for cache_key, _ in sorted_items[:items_to_remove]:
            self.cache.pop(cache_key, None)
            self.access_count.pop(cache_key, None)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "total_accesses": sum(self.access_count.values()),
            "hit_rate": len([k for k in self.access_count if self.access_count[k] > 1]) / max(len(self.cache), 1)
        }


class WorkflowEngine(QObject):
    """🎼 Enhanced Workflow Engine with Conversation Context Intelligence"""

    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    file_generated = Signal(str)
    project_loaded = Signal(str)
    workflow_progress = Signal(str, str)  # (stage, description)

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.terminal = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.output_dir = Path("./generated_projects")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize ENHANCED modular AI services
        self.planner_service = EnhancedPlannerService(llm_client, rag_manager)
        self.coder_service = EnhancedCoderService(llm_client, rag_manager)
        self.assembler_service = AssemblerService(llm_client, rag_manager)

        # Create enhanced orchestrator
        self.orchestrator = WorkflowOrchestrator(
            self.planner_service,
            self.coder_service,
            self.assembler_service,
            terminal_window
        )

        # Context cache for RAG optimization
        self.context_cache = ContextCache(max_cache_size=150)

        # Workflow state tracking
        self.current_workflow_state = {
            "stage": "idle",
            "progress": 0,
            "total_tasks": 0,
            "completed_tasks": 0
        }

        # OPTIMIZATION: Add workflow task tracking
        self.current_workflow_task = None
        self.workflow_active = False

        self._connect_code_viewer()
        self._log_service_info()

    def _connect_code_viewer(self):
        if self.code_viewer:
            self.code_viewer.file_changed.connect(self._on_file_changed)

    def _log_service_info(self):
        """Log information about the enhanced modular services"""
        self.terminal.log("🤖 Enhanced AI Services with Context Intelligence:")
        self.terminal.log("   🧠 EnhancedPlannerService: Context-aware planning & review")
        self.terminal.log("   ⚙️ EnhancedCoderService: Intelligent code generation")
        self.terminal.log("   📄 AssemblerService: Professional code assembly")
        self.terminal.log("   🎼 WorkflowOrchestrator: Context-driven workflow")

        if hasattr(self.llm_client, 'get_role_assignments'):
            assignments = self.llm_client.get_role_assignments()
            self.terminal.log("🎯 Enhanced Role Assignments:")
            self.terminal.log(f"   🧠 Planner: {assignments.get('planner', 'Not assigned')}")
            self.terminal.log(f"   ⚙️ Coder: {assignments.get('coder', 'Not assigned')}")
            self.terminal.log(f"   📄 Assembler: {assignments.get('assembler', 'Not assigned')}")

    def execute_workflow(self, user_prompt: str):
        """Execute standard workflow (compatibility method)"""
        self.execute_enhanced_workflow(user_prompt, [])

    def execute_enhanced_workflow(self, user_prompt: str, conversation_history: List = None):
        """Execute ENHANCED workflow with conversation context"""

        # Validation
        if not self._is_build_request(user_prompt):
            self.terminal.log(f"⚠️ Skipping workflow - not a build request: '{user_prompt}'")
            return

        # Check if workflow is already running
        if self.workflow_active:
            self.terminal.log("⚠️ Enhanced workflow already running - please wait for completion")
            return

        self.terminal.log("🚀 Starting Enhanced Context-Aware Workflow...")
        self.terminal.log(f"📝 Build request: {user_prompt}")

        if conversation_history:
            self.terminal.log(f"🧠 Using conversation context: {len(conversation_history)} messages")

        # Display system status
        cache_stats = self.context_cache.get_stats()
        self.terminal.log(f"📊 Context cache: {cache_stats['cache_size']} items, {cache_stats['hit_rate']:.1%} hit rate")

        if self.rag_manager and self.rag_manager.is_ready:
            self.terminal.log("🧠 RAG system ready - enhancing all services with 700k chunks")
        else:
            self.terminal.log("⚠️ RAG system not ready - using base knowledge")

        self.workflow_started.emit(user_prompt)
        self.workflow_active = True

        # Reset workflow state
        self.current_workflow_state = {
            "stage": "starting",
            "progress": 0,
            "total_tasks": 0,
            "completed_tasks": 0
        }

        # ENHANCED: Use asyncio.create_task with conversation context
        try:
            self.current_workflow_task = asyncio.create_task(
                self._execute_enhanced_workflow_async(user_prompt, conversation_history or [])
            )
        except Exception as e:
            self.terminal.log(f"❌ Failed to start enhanced workflow task: {e}")
            self.workflow_active = False
            self._update_workflow_stage("error", f"Failed to start: {e}")

    async def _execute_enhanced_workflow_async(self, user_prompt: str, conversation_history: List):
        """Enhanced async workflow with conversation context intelligence"""
        try:
            # Update workflow stage
            self._update_workflow_stage("planning", "Initializing context-aware services...")

            # OPTIMIZATION: Add small delays to allow UI updates
            await asyncio.sleep(0.1)

            # Execute workflow using enhanced orchestrator with conversation context
            self._update_workflow_stage("generation", "Processing with conversation intelligence...")
            await asyncio.sleep(0.1)  # Allow UI to update

            # CRITICAL: Pass conversation history to orchestrator
            result = await self.orchestrator.execute_workflow(
                user_prompt,
                self.output_dir,
                conversation_history
            )

            # Stage 4: Project finalization
            if result["success"]:
                self._update_workflow_stage("finalization", "Setting up code viewer...")
                await asyncio.sleep(0.1)  # Allow UI to update
                self._setup_code_viewer_project(result)

            self._update_workflow_stage("complete", "Enhanced context-aware workflow completed!")

            if result["success"]:
                self.terminal.log("✅ Enhanced context-aware workflow completed successfully!")
                self.terminal.log(
                    f"📁 Generated: {result.get('project_name', 'Project')} with {result.get('file_count', 0)} files")
            else:
                self.terminal.log(f"❌ Enhanced workflow failed: {result.get('error', 'Unknown error')}")

            # OPTIMIZATION: Emit completion signal after small delay
            await asyncio.sleep(0.1)
            self.workflow_completed.emit(result)

        except asyncio.CancelledError:
            self.terminal.log("⚠️ Enhanced workflow was cancelled")
            self._update_workflow_stage("cancelled", "Enhanced workflow cancelled by user")
            error_result = {"success": False, "error": "Enhanced workflow cancelled", "cancelled": True}
            self.workflow_completed.emit(error_result)

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()

            self.terminal.log(f"❌ Enhanced context-aware workflow failed: {e}")
            self.terminal.log("📝 Full error traceback:")
            # Split traceback into lines for better terminal display
            for line in error_traceback.split('\n'):
                if line.strip():
                    self.terminal.log(f"   {line}")

            self._update_workflow_stage("error", f"Enhanced workflow failed: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "traceback": error_traceback,
                "enhanced": True
            }
            self.workflow_completed.emit(error_result)

        finally:
            # OPTIMIZATION: Always clean up workflow state
            self.workflow_active = False
            self.current_workflow_task = None

    def cancel_workflow(self):
        """OPTIMIZATION: Allow users to cancel running workflows"""
        if self.current_workflow_task and not self.current_workflow_task.done():
            self.terminal.log("🛑 Cancelling enhanced workflow...")
            self.current_workflow_task.cancel()
            self.workflow_active = False
            return True
        return False

    def _is_build_request(self, prompt: str) -> bool:
        """Determine if this is actually a request to build something"""
        prompt_lower = prompt.lower().strip()

        # Ignore casual chat
        casual_phrases = [
            'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay',
            'yes', 'no', 'sure', 'cool', 'nice', 'good', 'great'
        ]

        if prompt_lower in casual_phrases:
            return False

        # Ignore questions without build intent
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(prompt_lower.startswith(word) for word in question_words):
            build_question_patterns = ['how to build', 'how to create', 'how to make', 'what should i build']
            if not any(pattern in build_question_patterns for pattern in build_question_patterns):
                return False

        # Require explicit build keywords
        build_keywords = [
            'build', 'create', 'make', 'generate', 'develop', 'code',
            'implement', 'write', 'design', 'construct', 'program',
            'application', 'app', 'website', 'tool', 'script', 'project'
        ]

        has_build_keyword = any(keyword in prompt_lower for keyword in build_keywords)
        is_substantial = len(prompt.split()) >= 3

        return has_build_keyword and is_substantial

    def _update_workflow_stage(self, stage: str, description: str):
        """Update workflow stage and emit progress signal"""
        self.current_workflow_state["stage"] = stage
        self.workflow_progress.emit(stage, description)
        self.terminal.log(f"📋 {description}")

    def _setup_code_viewer_project(self, result: dict):
        """OPTIMIZATION: Smooth code viewer setup with error handling"""
        if not result.get("success") or not self.code_viewer:
            return

        try:
            project_dir = result.get("project_dir")
            if not project_dir:
                self.terminal.log("⚠️ No project directory in result")
                return

            self.code_viewer.load_project(project_dir)
            self.terminal.log(f"📂 Code viewer loaded project: {Path(project_dir).name}")

            # OPTIMIZATION: Find and open the most likely main file
            main_files = ["main.py", "app.py", "calculator.py", "__init__.py", "gui.py"]
            project_path = Path(project_dir)

            for main_file in main_files:
                main_path = project_path / main_file
                if main_path.exists():
                    self.code_viewer.auto_open_file(str(main_path))
                    self.terminal.log(f"📄 Opened main file: {main_file}")
                    break
            else:
                # If no main file found, open the first Python file
                py_files = list(project_path.glob("*.py"))
                if py_files:
                    self.code_viewer.auto_open_file(str(py_files[0]))
                    self.terminal.log(f"📄 Opened file: {py_files[0].name}")

            self.project_loaded.emit(project_dir)

        except Exception as e:
            self.terminal.log(f"⚠️ Error setting up code viewer: {e}")

    def _on_file_changed(self, file_path: str, content: str):
        """Handle file changes in code viewer"""
        self.terminal.log(f"📝 File modified: {Path(file_path).name}")

    def get_workflow_stats(self) -> Dict:
        """Get current workflow statistics"""
        cache_stats = self.context_cache.get_stats()
        return {
            "workflow_state": self.current_workflow_state.copy(),
            "cache_stats": cache_stats,
            "workflow_active": self.workflow_active,
            "services": {
                "planner": "EnhancedPlannerService",
                "coder": "EnhancedCoderService",
                "assembler": "AssemblerService",
                "orchestrator": "EnhancedWorkflowOrchestrator"
            },
            "enhanced": True
        }

    def is_workflow_running(self) -> bool:
        """OPTIMIZATION: Check if workflow is currently running"""
        return self.workflow_active and (
                self.current_workflow_task is not None and
                not self.current_workflow_task.done()
        )

    def get_workflow_status(self) -> str:
        """OPTIMIZATION: Get human-readable workflow status"""
        if not self.workflow_active:
            return "Idle"
        elif self.current_workflow_task and self.current_workflow_task.done():
            return "Completed"
        else:
            stage = self.current_workflow_state.get("stage", "unknown")
            return stage.title()