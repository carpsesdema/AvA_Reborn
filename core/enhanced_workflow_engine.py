# core/enhanced_workflow_engine.py - Project-Aware Collaborative Workflow Engine

import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from PySide6.QtCore import QObject, Signal, QTimer

from core.project_state_manager import ProjectStateManager
from core.ai_feedback_system import AIFeedbackSystem, FeedbackType
from core.llm_client import LLMRole
import json


class EnhancedWorkflowEngine(QObject):
    """
    🚀 Enhanced Workflow Engine with Project Awareness & AI Collaboration

    This integrates with your existing workflow but adds:
    - Complete project state awareness
    - AI-to-AI collaboration and feedback
    - Iterative improvement cycles
    - Real-time streaming with context
    - User feedback integration
    """

    # Existing signals (keep compatibility)
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    file_generated = Signal(str)
    project_loaded = Signal(str)
    workflow_progress = Signal(str, str)  # (stage, description)

    # New signals for enhanced features
    ai_collaboration_started = Signal(str)  # session_id
    ai_feedback_received = Signal(str, str, str)  # from_ai, to_ai, content
    iteration_completed = Signal(str, int)  # file_path, iteration_number
    quality_check_completed = Signal(str, bool, str)  # file_path, approved, feedback

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None, project_root: str = None):
        super().__init__()

        # Core components (existing)
        self.llm_client = llm_client
        self.terminal = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager

        # Enhanced components (new)
        self.project_root = Path(project_root) if project_root else Path("./workspace")
        self.project_state = ProjectStateManager(self.project_root)
        self.ai_feedback_system = AIFeedbackSystem(llm_client, self.project_state, terminal_window)

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
            "iterations": {}  # file_path -> iteration_count
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
        """Log enhanced system initialization"""
        self.terminal.log("🚀 Enhanced Project-Aware Workflow Engine Initialized!")
        self.terminal.log("   🧠 Project State Manager: Full project awareness")
        self.terminal.log("   🤝 AI Feedback System: Specialist collaboration")
        self.terminal.log("   🔄 Iterative Refinement: Quality improvement loops")
        self.terminal.log("   👤 User Feedback: Interactive guidance")

        # Show project context if available
        if self.project_state.files:
            self.terminal.log(f"   📁 Existing project loaded: {len(self.project_state.files)} files")
        else:
            self.terminal.log("   📄 New project - will establish patterns as we build")

    async def execute_enhanced_workflow(self, user_prompt: str, enable_iterations: bool = True,
                                        max_iterations: int = 3) -> Dict[str, Any]:
        """
        🎯 Execute enhanced workflow with project awareness and AI collaboration

        This is the main entry point that orchestrates the entire process:
        1. Project-aware planning with full context
        2. AI specialist collaboration with feedback
        3. Iterative improvement based on review
        4. Real-time streaming and user interaction
        """

        if not self._is_build_request(user_prompt):
            self.terminal.log(f"⚠️ Not a build request: '{user_prompt}'")
            return {"success": False, "error": "Not a build request"}

        if self.workflow_active:
            self.terminal.log("⚠️ Workflow already running")
            return {"success": False, "error": "Workflow already active"}

        try:
            self.workflow_active = True
            self.workflow_started.emit(user_prompt)
            self._update_workflow_stage("initializing", "Starting enhanced collaborative workflow...")

            # Stage 1: Initialize collaboration session
            target_files = await self._predict_target_files(user_prompt)
            session_id = await self.ai_feedback_system.initiate_collaboration(
                task_description=user_prompt,
                target_files=target_files,
                participating_ais=["planner", "coder", "assembler", "reviewer"]
            )

            self.current_workflow_state["active_collaboration_session"] = session_id
            self.ai_collaboration_started.emit(session_id)

            # Stage 2: Project-aware planning
            self._update_workflow_stage("planning", "Creating project-aware plan with AI collaboration...")
            planner_guidance = await self.ai_feedback_system.process_planner_requirements(
                session_id, user_prompt
            )

            # Stage 3: Setup project directory
            project_name = planner_guidance.get("project_plan", {}).get("name", "enhanced_project")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_project_dir = self.output_dir / f"{project_name}_{timestamp}"
            self.current_project_dir.mkdir(exist_ok=True)

            # Stage 4: Collaborative file generation with iterations
            self._update_workflow_stage("generation", "Generating files with AI collaboration...")
            files_to_generate = planner_guidance.get("file_specifications", {})

            generated_files = []
            failed_files = []

            for file_path, file_spec in files_to_generate.items():
                self.terminal.log(f"🎯 Processing file: {file_path}")

                # Generate file with iterative improvement
                result = await self._generate_file_with_collaboration(
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

                    # Add to project state for future context
                    self.project_state.add_file(
                        result["file_path"],
                        result["content"],
                        ai_role="collaborative_workflow",
                        reasoning=f"Generated through enhanced workflow for: {user_prompt}"
                    )
                else:
                    failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})

            # Stage 5: Project finalization
            if generated_files:
                self._update_workflow_stage("finalization", "Finalizing project and updating code viewer...")
                await self._finalize_enhanced_project(generated_files)

            # Stage 6: Generate insights and recommendations
            insights = self._generate_project_insights()

            # Complete workflow
            success = len(generated_files) > 0
            result = {
                "success": success,
                "project_dir": str(self.current_project_dir),
                "generated_files": generated_files,
                "failed_files": failed_files,
                "project_name": project_name,
                "file_count": len(generated_files),
                "collaboration_session": session_id,
                "iterations_used": {f: self.current_workflow_state["iterations"].get(f, 0)
                                    for f in generated_files},
                "project_insights": insights
            }

            self._update_workflow_stage("complete",
                                        f"Enhanced workflow completed! Generated {len(generated_files)} files")
            self.workflow_completed.emit(result)

            return result

        except Exception as e:
            self.terminal.log(f"❌ Enhanced workflow failed: {e}")
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

    async def _generate_file_with_collaboration(self, session_id: str, file_path: str,
                                                file_spec: Dict[str, Any], planner_guidance: Dict[str, Any],
                                                enable_iterations: bool = True, max_iterations: int = 3) -> Dict[
        str, Any]:
        """Generate a single file with AI collaboration and iterative improvement"""

        iteration = 0
        current_content = ""
        review_approved = False

        while iteration < max_iterations and not review_approved:
            iteration += 1
            self.current_workflow_state["iterations"][file_path] = iteration

            self.terminal.log(f"[{file_path}] 🔄 Iteration {iteration}/{max_iterations}")

            try:
                # Step 1: Break into micro-tasks (project-aware)
                micro_tasks = await self._create_project_aware_micro_tasks(
                    session_id, file_path, file_spec, planner_guidance
                )

                # Step 2: Execute micro-tasks with AI collaboration
                task_results = await self._execute_collaborative_micro_tasks(
                    session_id, file_path, micro_tasks, planner_guidance
                )

                # Step 3: Assemble with context and feedback
                assembled_content, assembly_context = await self.ai_feedback_system.assemble_with_context_feedback(
                    session_id, file_path, task_results, planner_guidance
                )

                # Step 4: Comprehensive collaborative review
                review_approved, review_feedback, review_data = await self.ai_feedback_system.review_with_collaborative_feedback(
                    session_id, file_path, assembled_content, assembly_context, planner_guidance
                )

                current_content = assembled_content

                self.quality_check_completed.emit(file_path, review_approved, review_feedback)

                if review_approved:
                    self.terminal.log(f"[{file_path}] ✅ Review APPROVED on iteration {iteration}")
                    break
                else:
                    self.terminal.log(f"[{file_path}] ⚠️ Review suggests improvements (iteration {iteration})")

                    if not enable_iterations:
                        self.terminal.log(f"[{file_path}] ⏭️ Iterations disabled - using current version")
                        review_approved = True  # Accept as-is
                        break

                    if iteration < max_iterations:
                        # Process feedback for next iteration
                        improvement_plan = await self.ai_feedback_system.process_iterative_feedback(
                            session_id, file_path, review_data
                        )
                        self.terminal.log(f"[{file_path}] 🔧 Improvement plan: {improvement_plan.get('action')}")

                        # Brief pause for potential user feedback
                        await self._check_for_user_feedback(file_path, review_data)

            except Exception as e:
                self.terminal.log(f"[{file_path}] ❌ Iteration {iteration} failed: {e}")
                if iteration == max_iterations:
                    return {
                        "success": False,
                        "file_path": file_path,
                        "error": f"Failed after {max_iterations} iterations: {e}"
                    }

        # Write the final file
        if current_content:
            full_path = self.current_project_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(current_content, encoding='utf-8')

            self.iteration_completed.emit(file_path, iteration)

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

    async def _create_project_aware_micro_tasks(self, session_id: str, file_path: str,
                                                file_spec: Dict[str, Any], planner_guidance: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """Create micro-tasks with full project awareness"""

        # Get comprehensive project context for this specific file
        project_context = self.project_state.get_project_context(for_file=file_path, ai_role="planner")

        # Get consistency requirements
        consistency_requirements = self.project_state.get_consistency_requirements(file_path)

        # Enhanced micro-task creation prompt with full context
        task_prompt = f"""
As the Senior Planner AI, create ATOMIC micro-tasks for {file_path} with full project awareness:

FILE SPECIFICATION:
{json.dumps(file_spec, indent=2)}

PROJECT CONTEXT:
{json.dumps(project_context, indent=2)}

CONSISTENCY REQUIREMENTS:
{json.dumps(consistency_requirements, indent=2)}

PLANNER GUIDANCE:
{json.dumps(planner_guidance.get("specialist_guidance", {}).get("planner", {}), indent=2)}

Create micro-tasks that:
1. RESPECT existing project patterns and conventions
2. INTEGRATE seamlessly with related files: {consistency_requirements.get("related_files", [])}
3. MAINTAIN consistency with established coding standards
4. FOLLOW the interface contracts specified
5. CONSIDER dependencies and integration points

Return JSON array of ATOMIC tasks (5-15 lines each):
[
    {{
        "id": "unique_task_id",
        "type": "imports|constants|class|function|main",
        "description": "Clear task description",
        "priority": 1-10,
        "expected_lines": 5-15,
        "dependencies": ["list of other task ids this depends on"],
        "integration_notes": "How this integrates with existing code",
        "consistency_requirements": ["specific requirements for this task"]
    }}
]

Make each task ATOMIC and project-aware.
"""

        try:
            # Send task creation through feedback system for collaboration context
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(task_prompt, LLMRole.PLANNER):
                response_chunks.append(chunk)
                if len(response_chunks) % 10 == 0:
                    await asyncio.sleep(0.01)  # Allow UI updates

            response_text = ''.join(response_chunks)
            tasks = self._extract_json_array(response_text)

            # Add file context to each task
            for task in tasks:
                task["file_path"] = file_path
                task["project_context"] = project_context

            self.terminal.log(f"[{file_path}] 📋 Created {len(tasks)} project-aware micro-tasks")
            return tasks

        except Exception as e:
            self.terminal.log(f"[{file_path}] ❌ Micro-task creation failed: {e}")
            # Return basic fallback task
            return [{
                "id": f"implement_{file_path.replace('.py', '')}",
                "type": "complete",
                "description": f"Implement {file_path}",
                "file_path": file_path,
                "expected_lines": 50
            }]

    async def _execute_collaborative_micro_tasks(self, session_id: str, file_path: str,
                                                 micro_tasks: List[Dict[str, Any]],
                                                 planner_guidance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute micro-tasks with AI collaboration and feedback"""

        task_results = []

        # Group tasks by dependencies for optimal execution order
        independent_tasks = [t for t in micro_tasks if not t.get('dependencies')]
        dependent_tasks = [t for t in micro_tasks if t.get('dependencies')]

        # Execute independent tasks in parallel
        if independent_tasks:
            self.terminal.log(f"[{file_path}] ⚡ Executing {len(independent_tasks)} independent tasks in parallel")

            semaphore = asyncio.Semaphore(3)  # Limit concurrency

            async def execute_task_with_collaboration(task):
                async with semaphore:
                    return await self.ai_feedback_system.execute_coder_task_with_feedback(
                        session_id, task, planner_guidance
                    )

            # Execute in parallel with proper exception handling
            parallel_results = await asyncio.gather(
                *[execute_task_with_collaboration(task) for task in independent_tasks],
                return_exceptions=True
            )

            # Process results
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    self.terminal.log(f"[{file_path}] ❌ Task failed: {result}")
                    task_results.append({
                        "task": independent_tasks[i],
                        "code": f"# ERROR: {result}\npass"
                    })
                else:
                    task_results.append({
                        "task": independent_tasks[i],
                        "code": result
                    })

        # Execute dependent tasks sequentially
        for task in dependent_tasks:
            self.terminal.log(f"[{file_path}] 🔗 Executing dependent task: {task['description']}")

            try:
                code = await self.ai_feedback_system.execute_coder_task_with_feedback(
                    session_id, task, planner_guidance
                )
                task_results.append({"task": task, "code": code})
            except Exception as e:
                self.terminal.log(f"[{file_path}] ❌ Dependent task failed: {e}")
                task_results.append({"task": task, "code": f"# ERROR: {e}\npass"})

        return task_results

    async def _check_for_user_feedback(self, file_path: str, review_data: Dict[str, Any]):
        """Check for user feedback and pause if needed"""
        if self.pause_for_feedback:
            self.terminal.log(f"[{file_path}] ⏸️ Pausing for user feedback...")

            # Wait for user feedback or timeout
            timeout = 30  # 30 seconds
            for _ in range(timeout):
                if self.user_feedback_queue:
                    feedback = self.user_feedback_queue.pop(0)
                    self.terminal.log(f"[{file_path}] 👤 User feedback: {feedback['content']}")

                    # Add to project state
                    self.project_state.add_user_feedback(
                        file_path,
                        feedback['type'],
                        feedback['content'],
                        feedback.get('rating', 5)
                    )
                    break
                await asyncio.sleep(1)

    async def _finalize_enhanced_project(self, generated_files: List[str]):
        """Finalize project with enhanced features"""

        # Update code viewer with new project
        if self.code_viewer and self.current_project_dir:
            self.code_viewer.load_project(str(self.current_project_dir))

            # Auto-open the main file
            main_files = ["main.py", "app.py", "__init__.py"]
            for main_file in main_files:
                main_path = self.current_project_dir / main_file
                if main_path.exists():
                    self.code_viewer.auto_open_file(str(main_path))
                    break

        # Save project state
        self.project_state.save_state(self.current_project_dir / ".ava_project_state.json")

        # Emit project loaded signal
        self.project_loaded.emit(str(self.current_project_dir))

        self.terminal.log(f"📁 Project finalized: {self.current_project_dir.name}")

    def _generate_project_insights(self) -> Dict[str, Any]:
        """Generate insights about the project and workflow"""

        insights = {
            "project_stats": {
                "total_files": len(self.project_state.files),
                "patterns_detected": len(self.project_state.patterns),
                "ai_decisions_made": len(self.project_state.ai_decisions)
            },
            "collaboration_insights": self.ai_feedback_system.get_collaboration_insights(),
            "quality_metrics": self._calculate_quality_metrics(),
            "improvement_opportunities": self.project_state.get_improvement_opportunities(),
            "next_file_suggestions": self.project_state.get_next_file_suggestions(
                list(self.project_state.files.keys())
            )
        }

        return insights

    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate overall project quality metrics"""
        if not self.project_state.files:
            return {"overall_score": 0}

        total_score = sum(f.quality_score for f in self.project_state.files.values())
        avg_score = total_score / len(self.project_state.files)

        return {
            "overall_score": avg_score,
            "files_needing_improvement": len([f for f in self.project_state.files.values() if f.quality_score < 0.7]),
            "consistency_score": self._calculate_consistency_score()
        }

    def _calculate_consistency_score(self) -> float:
        """Calculate project consistency score"""
        # Simple consistency metric based on pattern adherence
        if not self.project_state.files:
            return 1.0

        consistent_files = 0
        for file_path, file_state in self.project_state.files.items():
            consistency_check = self.project_state.validate_file_consistency(file_path, file_state.content)
            if consistency_check["is_consistent"]:
                consistent_files += 1

        return consistent_files / len(self.project_state.files)

    # User interaction methods
    def add_user_feedback(self, feedback_type: str, content: str, rating: int = 5, file_path: str = None):
        """Add user feedback to the workflow"""
        feedback = {
            "type": feedback_type,
            "content": content,
            "rating": rating,
            "file_path": file_path,
            "timestamp": datetime.now()
        }

        self.user_feedback_queue.append(feedback)
        self.terminal.log(f"👤 User feedback added: {feedback_type}")

    def set_pause_for_feedback(self, enabled: bool):
        """Enable/disable pausing for user feedback"""
        self.pause_for_feedback = enabled
        self.terminal.log(f"⏸️ Pause for feedback: {'ENABLED' if enabled else 'DISABLED'}")

    async def request_file_iteration(self, file_path: str, feedback: str) -> bool:
        """Request another iteration for a specific file"""
        if not self.workflow_active:
            return False

        self.terminal.log(f"🔄 Iteration requested for {file_path}: {feedback}")

        # Add feedback and trigger re-processing
        self.add_user_feedback("iteration_request", feedback, file_path=file_path)

        # TODO: Implement file re-processing logic
        return True

    # Compatibility methods (keep existing interface)
    def execute_workflow(self, user_prompt: str):
        """Compatibility method - delegates to enhanced workflow"""
        if not self._is_build_request(user_prompt):
            self.terminal.log(f"⚠️ Skipping workflow - not a build request: '{user_prompt}'")
            return

        if self.workflow_active:
            self.terminal.log("⚠️ Workflow already running - please wait for completion")
            return

        # Create async task for enhanced workflow
        try:
            self.current_workflow_task = asyncio.create_task(
                self.execute_enhanced_workflow(user_prompt)
            )
        except Exception as e:
            self.terminal.log(f"❌ Failed to start enhanced workflow: {e}")

    async def _predict_target_files(self, user_prompt: str) -> List[str]:
        """Predict what files will be needed based on the prompt"""
        # Simple prediction - could be enhanced with ML
        prompt_lower = user_prompt.lower()

        if "gui" in prompt_lower or "interface" in prompt_lower:
            return ["main.py", "gui.py", "components.py"]
        elif "api" in prompt_lower or "web" in prompt_lower:
            return ["main.py", "api.py", "models.py", "routes.py"]
        elif "cli" in prompt_lower or "command" in prompt_lower:
            return ["main.py", "cli.py", "commands.py"]
        else:
            return ["main.py", "utils.py"]

    def _extract_json_array(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON array from text"""
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            pass

        return []

    def _is_build_request(self, prompt: str) -> bool:
        """Determine if this is a build request"""
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

    def _on_file_changed(self, file_path: str, content: str):
        """Handle file changes in code viewer"""
        self.terminal.log(f"📝 File modified: {Path(file_path).name}")

        # Update project state with changes
        if self.project_state:
            self.project_state.add_file(
                file_path,
                content,
                ai_role="user_edit",
                reasoning="User modified file in code viewer"
            )

    def get_workflow_stats(self) -> Dict:
        """Get current workflow statistics with enhanced info"""
        base_stats = {
            "workflow_state": self.current_workflow_state.copy(),
            "workflow_active": self.workflow_active,
            "project_files": len(self.project_state.files) if self.project_state else 0,
            "ai_decisions": len(self.project_state.ai_decisions) if self.project_state else 0
        }

        if hasattr(self, 'ai_feedback_system'):
            base_stats["collaboration_insights"] = self.ai_feedback_system.get_collaboration_insights()

        return base_stats

    def is_workflow_running(self) -> bool:
        """Check if workflow is currently running"""
        return self.workflow_active and (
                self.current_workflow_task is not None and
                not self.current_workflow_task.done()
        )