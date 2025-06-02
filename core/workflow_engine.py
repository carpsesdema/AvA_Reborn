# core/workflow_engine.py - Updated with Modular Services

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict
from PySide6.QtCore import QObject, QThread, Signal

from core.workflow_services import (
    PlannerService, CoderService, AssemblerService, WorkflowOrchestrator
)


# Keep the existing ContextCache class
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


class StreamingWorkflowThread(QThread):
    """Async workflow thread for streaming execution"""

    def __init__(self, workflow_engine, user_prompt):
        super().__init__()
        self.workflow_engine = workflow_engine
        self.user_prompt = user_prompt

    def run(self):
        # Run async workflow in thread
        asyncio.run(self.workflow_engine._execute_enhanced_workflow_async(self.user_prompt))


class WorkflowEngine(QObject):
    """🎼 Main Workflow Engine - Now with Modular Services"""

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

        # Initialize modular AI services
        self.planner_service = PlannerService(llm_client, rag_manager)
        self.coder_service = CoderService(llm_client, rag_manager)
        self.assembler_service = AssemblerService(llm_client, rag_manager)

        # Create orchestrator
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

        self._connect_code_viewer()
        self._log_service_info()

    def _connect_code_viewer(self):
        if self.code_viewer:
            self.code_viewer.file_changed.connect(self._on_file_changed)

    def _log_service_info(self):
        """Log information about the modular services"""
        self.terminal.log("🤖 Enhanced Modular AI Services Initialized:")
        self.terminal.log("   🧠 PlannerService: High-level planning & review")
        self.terminal.log("   ⚙️ CoderService: Parallel micro-task execution")
        self.terminal.log("   📄 AssemblerService: Code assembly & integration")
        self.terminal.log("   🎼 WorkflowOrchestrator: Parallel file processing")

        if hasattr(self.llm_client, 'get_role_assignments'):
            assignments = self.llm_client.get_role_assignments()
            self.terminal.log("🎯 Role Assignments:")
            self.terminal.log(f"   🧠 Planner: {assignments.get('planner', 'Not assigned')}")
            self.terminal.log(f"   ⚙️ Coder: {assignments.get('coder', 'Not assigned')}")
            self.terminal.log(f"   📄 Assembler: {assignments.get('assembler', 'Not assigned')}")

    def execute_workflow(self, user_prompt: str):
        """Execute the enhanced modular workflow"""
        # Validation
        if not self._is_build_request(user_prompt):
            self.terminal.log(f"⚠️ Skipping workflow - not a build request: '{user_prompt}'")
            return

        self.terminal.log("🚀 Starting Enhanced Modular Workflow...")
        self.terminal.log(f"📝 Build request: {user_prompt}")

        # Display system status
        cache_stats = self.context_cache.get_stats()
        self.terminal.log(f"📊 Context cache: {cache_stats['cache_size']} items, {cache_stats['hit_rate']:.1%} hit rate")

        if self.rag_manager and self.rag_manager.is_ready:
            self.terminal.log("🧠 RAG system ready - enhancing all services")
        else:
            self.terminal.log("⚠️ RAG system not ready - using base knowledge")

        self.workflow_started.emit(user_prompt)

        # Reset workflow state
        self.current_workflow_state = {
            "stage": "starting",
            "progress": 0,
            "total_tasks": 0,
            "completed_tasks": 0
        }

        # Use streaming workflow thread
        self.workflow_thread = StreamingWorkflowThread(self, user_prompt)
        self.workflow_thread.start()

    async def _execute_enhanced_workflow_async(self, user_prompt: str):
        """Enhanced async workflow with modular services"""
        try:
            # Update workflow stage
            self._update_workflow_stage("planning", "Initializing modular services...")

            # Execute workflow using orchestrator
            self._update_workflow_stage("generation", "Processing files in parallel...")
            result = await self.orchestrator.execute_workflow(user_prompt, self.output_dir)

            # Stage 4: Project finalization
            if result["success"]:
                self._update_workflow_stage("finalization", "Setting up code viewer...")
                self._setup_code_viewer_project(result)

            self._update_workflow_stage("complete", "Modular workflow completed!")
            self.terminal.log("✅ Enhanced modular workflow completed successfully!")
            self.workflow_completed.emit(result)

        except Exception as e:
            self._update_workflow_stage("error", f"Workflow failed: {e}")
            self.terminal.log(f"❌ Enhanced workflow failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.workflow_completed.emit(error_result)

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
            if not any(pattern in prompt_lower for pattern in build_question_patterns):
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
        if result["success"] and self.code_viewer:
            self.code_viewer.load_project(result["project_dir"])

            main_files = ["main.py", "app.py", "__init__.py"]
            project_path = Path(result["project_dir"])

            for main_file in main_files:
                main_path = project_path / main_file
                if main_path.exists():
                    self.code_viewer.auto_open_file(str(main_path))
                    break

            self.project_loaded.emit(result["project_dir"])

    def _on_file_changed(self, file_path: str, content: str):
        self.terminal.log(f"📝 File modified: {Path(file_path).name}")

    def get_workflow_stats(self) -> Dict:
        """Get current workflow statistics"""
        cache_stats = self.context_cache.get_stats()
        return {
            "workflow_state": self.current_workflow_state,
            "cache_stats": cache_stats,
            "services": {
                "planner": "ModularPlannerService",
                "coder": "ParallelCoderService",
                "assembler": "ModularAssemblerService",
                "orchestrator": "ParallelWorkflowOrchestrator"
            }
        }