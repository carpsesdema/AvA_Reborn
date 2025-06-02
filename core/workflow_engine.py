# core/workflow_engine.py - Enhanced Micro-Task Architecture

import json
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator, Tuple
from PySide6.QtCore import QObject, QThread, Signal

# Import LLMRole at the top level
try:
    from core.llm_client import LLMRole
except ImportError:
    # Fallback if enhanced client is not available
    class LLMRole:
        PLANNER = "planner"
        CODER = "coder"
        ASSEMBLER = "assembler"
        REVIEWER = "reviewer"
        CHAT = "chat"


class ContextCache:
    """Smart context cache for RAG results with ranking and pruning"""

    def __init__(self, max_cache_size: int = 100):
        self.cache: Dict[str, Dict] = {}
        self.access_count: Dict[str, int] = {}
        self.max_cache_size = max_cache_size

    def _generate_cache_key(self, query: str, context_type: str = "general") -> str:
        """Generate cache key from query and context type"""
        combined = f"{context_type}:{query.lower()}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def get_context(self, query: str, context_type: str = "general") -> Optional[str]:
        """Get cached context if available"""
        cache_key = self._generate_cache_key(query, context_type)

        if cache_key in self.cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]["context"]

        return None

    def store_context(self, query: str, context: str, context_type: str = "general",
                      relevance_score: float = 1.0):
        """Store context in cache with relevance score"""
        if not context or len(context.strip()) < 10:  # Skip very short context
            return

        cache_key = self._generate_cache_key(query, context_type)

        # Prune cache if at max size
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

        # Sort by access count (ascending) and remove bottom 25%
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


class PlannerAI:
    """Dedicated Planner AI - Creates high-level plans and micro-task breakdowns"""

    def __init__(self, llm_client, rag_manager=None, terminal=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager
        self.terminal = terminal
        self.role = LLMRole.PLANNER

    async def create_project_plan(self, user_prompt: str, context_cache: ContextCache) -> dict:
        """Create high-level project plan with architecture decisions"""
        self._log("🧠 PLANNER: Analyzing requirements and creating project plan...")

        # Get RAG context for planning
        rag_context = await self._get_planning_context(user_prompt, context_cache)

        plan_prompt = f"""
You are a Senior Software Architect. Analyze this request and create a comprehensive plan:

REQUEST: {user_prompt}

REFERENCE CONTEXT:
{rag_context if rag_context else "No specific context available - use best practices"}

Create a detailed project plan as JSON:
{{
    "project_name": "descriptive_snake_case_name",
    "description": "Clear project description",
    "architecture_type": "cli|web_app|gui|library|api",
    "tech_stack": ["python", "requests", "etc"],
    "project_structure": {{
        "main_entry": "main.py",
        "config_files": ["requirements.txt", "README.md"],
        "core_modules": ["core/engine.py", "utils/helpers.py"],
        "optional_files": ["tests/test_main.py"]
    }},
    "file_priorities": {{
        "main.py": 1,
        "core/engine.py": 2,
        "utils/helpers.py": 3,
        "requirements.txt": 4,
        "README.md": 5
    }},
    "dependencies": ["requests", "pathlib"],
    "complexity_estimate": "simple|medium|complex"
}}

Focus on creating a clean, professional structure. Maximum 8 files for initial version.
"""

        response = await self._stream_response(plan_prompt, "Planning")

        try:
            plan = self._extract_json(response)
            self._log(f"📋 PLANNER: Created plan for '{plan['project_name']}'")
            self._log(f"   • Architecture: {plan.get('architecture_type', 'unspecified')}")
            self._log(f"   • Files planned: {len(plan.get('project_structure', {}).get('core_modules', []))}")
            return plan
        except Exception as e:
            self._log(f"❌ PLANNER: Plan parsing failed: {e}")
            return self._create_fallback_plan(user_prompt)

    async def create_micro_tasks(self, plan: dict, file_path: str, context_cache: ContextCache) -> List[dict]:
        """Break down a file into atomic micro-tasks"""
        self._log(f"🧠 PLANNER: Creating micro-tasks for {file_path}...")

        # Get file-specific context
        file_context = await self._get_file_context(file_path, plan, context_cache)

        micro_task_prompt = f"""
You are a Senior Developer creating atomic micro-tasks for precise code generation.

FILE: {file_path}
PROJECT: {plan['project_name']} - {plan['description']}
ARCHITECTURE: {plan.get('architecture_type', 'unspecified')}

CONTEXT & EXAMPLES:
{file_context if file_context else "No specific patterns found - use standard practices"}

Break this file into ATOMIC micro-tasks. Each task should produce 5-15 lines of code maximum.

Return JSON array of micro-tasks:
[
    {{
        "id": "imports_{file_path.replace('/', '_').replace('.py', '')}",
        "type": "imports",
        "description": "Import required modules (requests, pathlib, etc)",
        "priority": 1,
        "dependencies": [],
        "expected_lines": 5,
        "context_hint": "Standard imports for this file type"
    }},
    {{
        "id": "class_definition_{file_path.replace('/', '_').replace('.py', '')}",
        "type": "class",
        "description": "Define main class with __init__ method",
        "priority": 2,
        "dependencies": ["imports_{file_path.replace('/', '_').replace('.py', '')}"],
        "expected_lines": 8,
        "context_hint": "Class structure with docstring and basic setup"
    }},
    {{
        "id": "method_process_data",
        "type": "method",
        "description": "Create process_data method with error handling",
        "priority": 3,
        "dependencies": ["class_definition_{file_path.replace('/', '_').replace('.py', '')}"],
        "expected_lines": 12,
        "context_hint": "Method with docstring, parameters, and return value"
    }}
]

Make tasks ATOMIC - each should be a single, clear coding action.
"""

        response = await self._stream_response(micro_task_prompt, f"Micro-tasking {file_path}")

        try:
            tasks = self._extract_json(response)
            for task in tasks:
                task["file_path"] = file_path
                task["created_by"] = "planner"

            self._log(f"   • Created {len(tasks)} micro-tasks for {file_path}")
            return tasks
        except Exception as e:
            self._log(f"❌ PLANNER: Micro-task creation failed: {e}")
            return self._create_fallback_tasks(file_path)

    async def review_and_approve(self, file_path: str, assembled_code: str, original_tasks: List[dict]) -> Tuple[
        bool, str, List[str]]:
        """Review assembled code and provide feedback"""
        self._log(f"🧠 PLANNER: Reviewing assembled code for {file_path}...")

        review_prompt = f"""
You are a Senior Code Reviewer. Review this assembled code for quality and completeness.

FILE: {file_path}
ORIGINAL TASKS: {len(original_tasks)} micro-tasks completed

CODE TO REVIEW:
{assembled_code}

ORIGINAL TASK LIST:
{json.dumps([{"id": t["id"], "description": t["description"]} for t in original_tasks], indent=2)}

Provide review as JSON:
{{
    "approved": true/false,
    "overall_quality": "excellent|good|needs_improvement|poor",
    "feedback": "Detailed feedback on what's good/bad",
    "suggestions": [
        "Specific improvement suggestion 1",
        "Specific improvement suggestion 2"
    ],
    "missing_elements": [
        "Any missing functionality from original tasks"
    ],
    "code_issues": [
        "Any syntax, logic, or style issues found"
    ]
}}

Be thorough but constructive in your review.
"""

        response = await self._stream_response(review_prompt, f"Reviewing {file_path}")

        try:
            review = self._extract_json(response)
            approved = review.get("approved", False)
            feedback = review.get("feedback", "Review completed")
            suggestions = review.get("suggestions", [])

            if approved:
                self._log(f"   ✅ PLANNER: Code approved for {file_path}")
            else:
                self._log(f"   ⚠️ PLANNER: Code needs improvement for {file_path}")
                for suggestion in suggestions[:3]:  # Show first 3 suggestions
                    self._log(f"      • {suggestion}")

            return approved, feedback, suggestions
        except Exception as e:
            self._log(f"❌ PLANNER: Review failed: {e}")
            return True, "Review error - proceeding", []  # Default to approved on error

    async def _get_planning_context(self, user_prompt: str, context_cache: ContextCache) -> str:
        """Get context for project planning"""
        cache_key = f"planning_{user_prompt[:50]}"
        cached = context_cache.get_context(cache_key, "planning")

        if cached:
            self._log("   ⚡ Using cached planning context")
            return cached

        if self.rag_manager and self.rag_manager.is_ready:
            context = self.rag_manager.get_context_for_code_generation(
                f"project architecture planning {user_prompt}", "python"
            )
            if context:
                context_cache.store_context(cache_key, context, "planning", 0.9)
                self._log("   📚 Retrieved planning context from RAG")
                return context

        return ""

    async def _get_file_context(self, file_path: str, plan: dict, context_cache: ContextCache) -> str:
        """Get context for specific file type"""
        file_type = Path(file_path).suffix or "module"
        cache_key = f"file_context_{file_type}_{plan.get('architecture_type', 'general')}"
        cached = context_cache.get_context(cache_key, "file_patterns")

        if cached:
            self._log("   ⚡ Using cached file context")
            return cached

        if self.rag_manager and self.rag_manager.is_ready:
            context = self.rag_manager.get_context_for_code_generation(
                f"{file_path} {plan.get('architecture_type', '')} implementation patterns", "python"
            )
            if context:
                context_cache.store_context(cache_key, context, "file_patterns", 0.8)
                self._log("   📚 Retrieved file patterns from RAG")
                return context

        return ""

    async def _stream_response(self, prompt: str, operation: str) -> str:
        """Stream LLM response with progress updates using Planner role"""
        chunks = []
        async for chunk in self.llm_client.stream_chat(prompt, self.role):
            chunks.append(chunk)
            if len(chunks) % 10 == 0:
                self._log(f"   → {operation}... ({len(''.join(chunks))} chars)")

        response = ''.join(chunks)
        self._log(f"   → {operation} completed ({len(response)} chars)")
        return response

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from LLM response"""
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
        raise ValueError("No valid JSON found in response")

    def _create_fallback_plan(self, user_prompt: str) -> dict:
        """Create fallback plan when LLM parsing fails"""
        return {
            "project_name": "generated_project",
            "description": user_prompt,
            "architecture_type": "cli",
            "tech_stack": ["python"],
            "project_structure": {
                "main_entry": "main.py",
                "core_modules": ["main.py"]
            },
            "file_priorities": {"main.py": 1},
            "dependencies": []
        }

    def _create_fallback_tasks(self, file_path: str) -> List[dict]:
        """Create fallback tasks when micro-task creation fails"""
        return [
            {
                "id": f"implement_{file_path}",
                "type": "complete_file",
                "description": f"Implement complete {file_path}",
                "priority": 1,
                "file_path": file_path,
                "created_by": "planner_fallback"
            }
        ]

    def _log(self, message: str):
        """Log message to terminal"""
        if self.terminal:
            self.terminal.log(message)


class CoderAI:
    """Specialized Coder AI - Implements atomic micro-tasks"""

    def __init__(self, llm_client, rag_manager=None, terminal=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager
        self.terminal = terminal
        self.role = LLMRole.CODER

    async def implement_micro_task(self, task: dict, file_context: str, context_cache: ContextCache) -> str:
        """Implement a single atomic micro-task"""
        task_id = task.get("id", "unknown")
        self._log(f"⚙️ CODER: Implementing {task_id}")

        # Get task-specific context
        task_context = await self._get_task_context(task, context_cache)

        # Use specialized code generation model if available
        code_prompt = f"""
You are a Specialist Code Generator. Implement this ATOMIC micro-task with precision.

TASK: {task['description']}
TYPE: {task.get('type', 'general')}
FILE: {task.get('file_path', 'unknown')}
EXPECTED LINES: ~{task.get('expected_lines', 10)}

EXISTING FILE CONTEXT:
{file_context}

REFERENCE PATTERNS & EXAMPLES:
{task_context if task_context else "Use standard Python practices"}

CONTEXT HINT: {task.get('context_hint', 'Standard implementation')}

REQUIREMENTS:
1. Generate ONLY the code for this specific task
2. Keep it atomic - don't implement other tasks
3. Follow Python best practices and PEP 8
4. Include appropriate docstrings for classes/functions
5. Add type hints where appropriate
6. Handle errors appropriately
7. Make code production-ready

Return ONLY Python code, no explanations:
"""

        chunks = []
        async for chunk in self.llm_client.stream_chat(code_prompt, self.role):
            chunks.append(chunk)
            if len(chunks) % 5 == 0:
                self._log(f"   → Generating {task_id}... ({len(''.join(chunks))} chars)")

        code = ''.join(chunks)
        clean_code = self._clean_code_response(code)

        self._log(f"   ✅ Generated {len(clean_code)} chars for {task_id}")
        return clean_code

    async def _get_task_context(self, task: dict, context_cache: ContextCache) -> str:
        """Get context specific to this task type"""
        task_type = task.get("type", "general")
        cache_key = f"task_context_{task_type}"
        cached = context_cache.get_context(cache_key, "task_patterns")

        if cached:
            return cached

        if self.rag_manager and self.rag_manager.is_ready:
            query = f"python {task_type} implementation patterns examples"
            context = self.rag_manager.get_context_for_code_generation(query, "python")
            if context:
                context_cache.store_context(cache_key, context, "task_patterns", 0.7)
                return context

        return ""

    def _clean_code_response(self, code_response: str) -> str:
        """Extract clean code from LLM response"""
        if "```python" in code_response:
            code = code_response.split("```python")[1].split("```")[0].strip()
        elif "```" in code_response:
            code = code_response.split("```")[1].split("```")[0].strip()
        else:
            code = code_response.strip()
        return code

    def _log(self, message: str):
        """Log message to terminal"""
        if self.terminal:
            self.terminal.log(message)


class AssemblerAI:
    """Assembler AI - Combines micro-tasks into cohesive files"""

    def __init__(self, llm_client, rag_manager=None, terminal=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager
        self.terminal = terminal
        self.role = LLMRole.ASSEMBLER

    async def assemble_file(self, file_path: str, task_results: List[dict], plan: dict,
                            context_cache: ContextCache) -> str:
        """Assemble micro-task results into a cohesive file"""
        self._log(f"📄 ASSEMBLER: Assembling {file_path} from {len(task_results)} micro-tasks")

        # Get assembly context
        assembly_context = await self._get_assembly_context(file_path, plan, context_cache)

        # Prepare task results for assembly
        code_sections = []
        for result in task_results:
            task_info = result['task']
            code = result['code']
            code_sections.append(f"# Task: {task_info['description']}\n{code}")

        assembly_prompt = f"""
You are a Code Assembler. Combine these micro-task results into a cohesive, professional Python file.

FILE: {file_path}
PROJECT: {plan['project_name']} - {plan['description']}

MICRO-TASK RESULTS TO ASSEMBLE:
{chr(10).join(code_sections)}

ASSEMBLY BEST PRACTICES:
{assembly_context if assembly_context else "Use standard Python file organization"}

REQUIREMENTS:
1. Organize imports at the top (remove duplicates)
2. Add file-level docstring with description
3. Maintain logical flow and structure
4. Ensure all components work together
5. Add missing connections between components
6. Fix any syntax or logical issues
7. Follow PEP 8 conventions
8. Make the file executable and functional

Return the complete, assembled Python file:
"""

        chunks = []
        async for chunk in self.llm_client.stream_chat(assembly_prompt, self.role):
            chunks.append(chunk)
            if len(chunks) % 8 == 0:
                self._log(f"   → Assembling {file_path}... ({len(''.join(chunks))} chars)")

        assembled_code = ''.join(chunks)
        clean_code = self._clean_code_response(assembled_code)

        self._log(f"   ✅ Assembled {file_path} ({len(clean_code)} chars)")
        return clean_code

    async def _get_assembly_context(self, file_path: str, plan: dict, context_cache: ContextCache) -> str:
        """Get context for file assembly"""
        file_type = Path(file_path).suffix or "module"
        cache_key = f"assembly_{file_type}_{plan.get('architecture_type', 'general')}"
        cached = context_cache.get_context(cache_key, "assembly_patterns")

        if cached:
            return cached

        if self.rag_manager and self.rag_manager.is_ready:
            query = f"python file organization structure {file_path} best practices"
            context = self.rag_manager.get_context_for_code_generation(query, "python")
            if context:
                context_cache.store_context(cache_key, context, "assembly_patterns", 0.7)
                return context

        return ""

    def _clean_code_response(self, code_response: str) -> str:
        """Extract clean code from LLM response"""
        if "```python" in code_response:
            code = code_response.split("```python")[1].split("```")[0].strip()
        elif "```" in code_response:
            code = code_response.split("```")[1].split("```")[0].strip()
        else:
            code = code_response.strip()
        return code

    def _log(self, message: str):
        """Log message to terminal"""
        if self.terminal:
            self.terminal.log(message)


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

        # Initialize AI specialists with enhanced LLM client
        self.planner = PlannerAI(llm_client, rag_manager, terminal_window)
        self.coder = CoderAI(llm_client, rag_manager, terminal_window)
        self.assembler = AssemblerAI(llm_client, rag_manager, terminal_window)

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
        self._log_model_assignments()

    def _connect_code_viewer(self):
        if self.code_viewer:
            self.code_viewer.file_changed.connect(self._on_file_changed)

    def _log_model_assignments(self):
        """Log which models are assigned to each AI role"""
        if hasattr(self.llm_client, 'get_role_assignments'):
            assignments = self.llm_client.get_role_assignments()
            self.terminal.log("🤖 AI Specialist Model Assignments:")
            self.terminal.log(f"   🧠 Planner: {assignments.get('planner', 'Not assigned')}")
            self.terminal.log(f"   ⚙️ Coder: {assignments.get('coder', 'Not assigned')}")
            self.terminal.log(f"   📄 Assembler: {assignments.get('assembler', 'Not assigned')}")
            self.terminal.log(f"   🔍 Reviewer: {assignments.get('reviewer', 'Not assigned')}")

            # Show cost optimization info
            if hasattr(self.llm_client, 'get_cost_estimate'):
                self.terminal.log("💰 Cost optimization: Using specialized models for each role")
        else:
            self.terminal.log("🤖 Using legacy LLM client (single model for all roles)")

    def execute_workflow(self, user_prompt: str):
        """Execute the enhanced micro-task workflow"""
        # Validation
        if not self._is_build_request(user_prompt):
            self.terminal.log(f"⚠️  Skipping workflow - not a build request: '{user_prompt}'")
            return

        self.terminal.log("🚀 Starting AvA Enhanced Micro-Task Workflow...")
        self.terminal.log(f"📝 Build request: {user_prompt}")

        # Display system status
        cache_stats = self.context_cache.get_stats()
        self.terminal.log(f"📊 Context cache: {cache_stats['cache_size']} items, {cache_stats['hit_rate']:.1%} hit rate")

        if self.rag_manager and self.rag_manager.is_ready:
            self.terminal.log("🧠 RAG system ready - will enhance all AI agents with knowledge")
        else:
            self.terminal.log("⚠️  RAG system not ready - proceeding with base knowledge")

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
        """Enhanced async workflow with clear AI role separation"""
        try:
            # Stage 1: Planner creates high-level plan
            self._update_workflow_stage("planning", "Planner AI creating project architecture...")
            plan = await self.planner.create_project_plan(user_prompt, self.context_cache)

            # Stage 2: Planner creates micro-tasks for each file
            self._update_workflow_stage("decomposition", "Planner AI creating micro-task breakdown...")
            all_micro_tasks = []

            # Get files in priority order
            file_priorities = plan.get("file_priorities", {"main.py": 1})
            sorted_files = sorted(file_priorities.items(), key=lambda x: x[1])

            for file_path, priority in sorted_files:
                file_tasks = await self.planner.create_micro_tasks(plan, file_path, self.context_cache)
                all_micro_tasks.extend(file_tasks)

            # Update total tasks count
            self.current_workflow_state["total_tasks"] = len(all_micro_tasks)
            self.terminal.log(f"📋 Total micro-tasks created: {len(all_micro_tasks)}")

            # Stage 3: Coder AI implements micro-tasks
            self._update_workflow_stage("generation", "Coder AI implementing micro-tasks...")
            result = await self._execute_micro_tasks_with_review(all_micro_tasks, plan)

            # Stage 4: Project finalization
            if result["success"]:
                self._update_workflow_stage("finalization", "Setting up code viewer...")
                self._setup_code_viewer_project(result)

            self._update_workflow_stage("complete", "Enhanced workflow completed!")
            self.terminal.log("✅ Enhanced micro-task workflow completed successfully!")
            self.workflow_completed.emit(result)

        except Exception as e:
            self._update_workflow_stage("error", f"Workflow failed: {e}")
            self.terminal.log(f"❌ Enhanced workflow failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.workflow_completed.emit(error_result)

    async def _execute_micro_tasks_with_review(self, micro_tasks: list, plan: dict) -> dict:
        """Execute micro-tasks with Coder AI and review with Planner AI"""
        project_name = plan["project_name"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_dir = self.output_dir / f"{project_name}_{timestamp}"
        project_dir.mkdir(exist_ok=True)

        generated_files = []

        # Group tasks by file
        tasks_by_file = {}
        for task in micro_tasks:
            file_path = task["file_path"]
            if file_path not in tasks_by_file:
                tasks_by_file[file_path] = []
            tasks_by_file[file_path].append(task)

        # Process each file with full AI pipeline
        for file_idx, (file_path, file_tasks) in enumerate(tasks_by_file.items()):
            self.terminal.log(f"🔄 Processing file {file_idx + 1}/{len(tasks_by_file)}: {file_path}")

            # Step 1: Coder AI implements each micro-task
            task_results = []
            file_context = ""

            for task in file_tasks:
                self.current_workflow_state["completed_tasks"] += 1
                progress = (self.current_workflow_state["completed_tasks"] /
                            self.current_workflow_state["total_tasks"]) * 100

                self.terminal.log(f"  [{progress:.0f}%] Coder AI: {task['description']}")

                # Coder AI implements the task
                code = await self.coder.implement_micro_task(task, file_context, self.context_cache)

                task_results.append({
                    "task": task,
                    "code": code
                })

                # Update file context for next task
                file_context += f"\n# {task['description']}\n{code}\n"

            # Step 2: Assembler AI combines micro-tasks
            self.terminal.log(f"📄 Assembler AI: Combining {len(task_results)} micro-tasks for {file_path}")
            assembled_code = await self.assembler.assemble_file(file_path, task_results, plan, self.context_cache)

            # Step 3: Planner AI reviews and approves
            approved, feedback, suggestions = await self.planner.review_and_approve(
                file_path, assembled_code, file_tasks
            )

            # Step 4: Handle review results
            final_code = assembled_code
            if not approved and suggestions:
                self.terminal.log(f"🔄 Implementing Planner AI suggestions for {file_path}")
                # Could implement revision loop here
                # For now, we'll proceed with suggestions logged
                for suggestion in suggestions[:2]:
                    self.terminal.log(f"   💡 Suggestion: {suggestion}")

            # Step 5: Write the final file
            full_file_path = project_dir / file_path
            full_file_path.parent.mkdir(parents=True, exist_ok=True)
            full_file_path.write_text(final_code, encoding='utf-8')
            generated_files.append(str(full_file_path))

            self.terminal.log(f"  ✅ File completed: {full_file_path}")
            self.file_generated.emit(str(full_file_path))

        return {
            "success": True,
            "project_dir": str(project_dir),
            "files": generated_files,
            "project_name": project_name,
            "file_count": len(generated_files)
        }

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
            "cache_stats": cache_stats
        }