# core/workflow_engine.py - Enhanced with Streaming & Caching

import json
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
from PySide6.QtCore import QObject, QThread, Signal


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


class StreamingWorkflowThread(QThread):
    """Async workflow thread for streaming execution"""

    def __init__(self, workflow_engine, user_prompt):
        super().__init__()
        self.workflow_engine = workflow_engine
        self.user_prompt = user_prompt

    def run(self):
        # Run async workflow in thread
        asyncio.run(self.workflow_engine._execute_workflow_async(self.user_prompt))


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

        # NEW: Context cache for RAG optimization
        self.context_cache = ContextCache(max_cache_size=150)

        # NEW: Workflow state tracking
        self.current_workflow_state = {
            "stage": "idle",
            "progress": 0,
            "total_tasks": 0,
            "completed_tasks": 0
        }

        self._connect_code_viewer()

    def _connect_code_viewer(self):
        if self.code_viewer:
            self.code_viewer.file_changed.connect(self._on_file_changed)

    def execute_workflow(self, user_prompt: str):
        # VALIDATION: Check if this is actually a build request
        if not self._is_build_request(user_prompt):
            self.terminal.log(f"⚠️  Skipping workflow - not a build request: '{user_prompt}'")
            return

        self.terminal.log("🚀 Starting AvA development workflow...")
        self.terminal.log(f"📝 Build request: {user_prompt}")

        # Display cache stats
        cache_stats = self.context_cache.get_stats()
        self.terminal.log(f"📊 Context cache: {cache_stats['cache_size']} items, {cache_stats['hit_rate']:.1%} hit rate")

        # NEW: Check RAG status with enhanced feedback
        if self.rag_manager and self.rag_manager.is_ready:
            self.terminal.log("🧠 RAG system ready - will enhance code generation with cached knowledge")
        else:
            self.terminal.log("⚠️  RAG system not ready - proceeding without context enhancement")

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

    def _is_build_request(self, prompt: str) -> bool:
        """Determine if this is actually a request to build something"""
        prompt_lower = prompt.lower().strip()

        # IGNORE casual chat
        casual_phrases = [
            'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay',
            'yes', 'no', 'sure', 'cool', 'nice', 'good', 'great'
        ]

        if prompt_lower in casual_phrases:
            return False

        # IGNORE questions without build intent
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(prompt_lower.startswith(word) for word in question_words):
            build_question_patterns = ['how to build', 'how to create', 'how to make', 'what should i build']
            if not any(pattern in prompt_lower for pattern in build_question_patterns):
                return False

        # REQUIRE explicit build keywords
        build_keywords = [
            'build', 'create', 'make', 'generate', 'develop', 'code',
            'implement', 'write', 'design', 'construct', 'program',
            'application', 'app', 'website', 'tool', 'script', 'project'
        ]

        has_build_keyword = any(keyword in prompt_lower for keyword in build_keywords)
        is_substantial = len(prompt.split()) >= 3

        return has_build_keyword and is_substantial

    async def _execute_workflow_async(self, user_prompt: str):
        """NEW: Async workflow execution with streaming"""
        try:
            # Stage 1: Planning
            self._update_workflow_stage("planning", "Creating high-level plan...")
            plan = await self._create_plan_streaming(user_prompt)

            # Stage 2: Micro-task creation
            self._update_workflow_stage("decomposition", "Breaking down into micro-tasks...")
            micro_tasks = await self._create_micro_tasks_streaming(plan, user_prompt)

            # Update total tasks count
            self.current_workflow_state["total_tasks"] = len(micro_tasks)

            # Stage 3: Code generation
            self._update_workflow_stage("generation", "Executing micro-tasks...")
            result = await self._execute_micro_tasks_streaming(micro_tasks, plan)

            # Stage 4: Finalization
            if result["success"]:
                self._update_workflow_stage("finalization", "Setting up code viewer...")
                self._setup_code_viewer_project(result)

            self._update_workflow_stage("complete", "Workflow completed!")
            self.terminal.log("✅ Workflow completed successfully!")
            self.workflow_completed.emit(result)

        except Exception as e:
            self._update_workflow_stage("error", f"Workflow failed: {e}")
            self.terminal.log(f"❌ Workflow failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.workflow_completed.emit(error_result)

    def _update_workflow_stage(self, stage: str, description: str):
        """Update workflow stage and emit progress signal"""
        self.current_workflow_state["stage"] = stage
        self.workflow_progress.emit(stage, description)
        self.terminal.log(f"📋 {description}")

    async def _create_plan_streaming(self, user_prompt: str) -> dict:
        """NEW: Streaming plan creation with context caching"""
        # Check cache for similar planning requests
        cache_key = f"planning_{user_prompt[:50]}"
        cached_context = self.context_cache.get_context(cache_key, "planning")

        # Get RAG context with caching
        rag_context = ""
        if self.rag_manager and self.rag_manager.is_ready:
            self.terminal.log("  -> Querying RAG for planning context (with caching)...")

            if cached_context:
                rag_context = cached_context
                self.terminal.log(f"  -> Using cached context ({len(rag_context)} chars) ⚡")
            else:
                rag_context = self.rag_manager.get_context_for_code_generation(
                    f"project planning architecture {user_prompt}", "python"
                )
                if rag_context:
                    # Store in cache with high relevance for planning
                    self.context_cache.store_context(cache_key, rag_context, "planning", 0.9)
                    self.terminal.log(f"  -> RAG context retrieved and cached ({len(rag_context)} chars)")
                else:
                    self.terminal.log("  -> No relevant RAG context found")

        plan_prompt = f"""
Act as a software architect. Create a detailed plan for: {user_prompt}

{f"Reference Context from Knowledge Base:\n{rag_context}\n" if rag_context else ""}

Return ONLY a JSON structure:
{{
    "project_name": "snake_case_name",
    "description": "Brief description",
    "architecture": "Brief architecture overview",
    "files_needed": [
        {{"path": "main.py", "purpose": "Main entry point", "priority": 1}},
        {{"path": "utils.py", "purpose": "Utility functions", "priority": 2}}
    ],
    "dependencies": ["requests", "pathlib"]
}}

Keep it simple but functional - maximum 5 files for a working prototype.
"""

        self.terminal.log("  -> Calling Planner LLM (streaming)...")

        # NEW: Stream the planning response
        response_chunks = []
        async for chunk in self.llm_client.stream_chat(plan_prompt):
            response_chunks.append(chunk)
            # Show streaming progress in terminal
            if len(response_chunks) % 10 == 0:  # Every 10 chunks
                self.terminal.log(f"    -> Planning... ({len(''.join(response_chunks))} chars)")

        response = ''.join(response_chunks)
        self.terminal.log(f"  -> Planner response completed ({len(response)} chars)")

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                plan = json.loads(response[start:end])
                self.terminal.log(f"  -> Plan created: {plan['project_name']}")

                self.terminal.log("  -> Plan Summary:")
                for file_info in plan["files_needed"]:
                    self.terminal.log(f"     📄 {file_info['path']}: {file_info['purpose']}")

                return plan
        except Exception as e:
            self.terminal.log(f"  -> JSON parsing failed: {e}")

        # Fallback plan
        plan = {
            "project_name": "generated_project",
            "description": user_prompt,
            "files_needed": [{"path": "main.py", "purpose": "Main implementation", "priority": 1}],
            "dependencies": []
        }
        self.terminal.log("  -> Using fallback plan")
        return plan

    async def _create_micro_tasks_streaming(self, plan: dict, user_prompt: str) -> list:
        """NEW: Streaming micro-task creation with smart caching"""
        micro_tasks = []
        files_sorted = sorted(plan["files_needed"], key=lambda x: x.get("priority", 999))

        for file_info in files_sorted:
            self.terminal.log(f"  -> Creating micro-tasks for {file_info['path']} (streaming)...")

            # Smart context caching by file purpose and type
            context_cache_key = f"{file_info['purpose']}_{file_info['path']}_microtasks"
            cached_context = self.context_cache.get_context(context_cache_key, "file_planning")

            rag_context = ""
            if self.rag_manager and self.rag_manager.is_ready:
                if cached_context:
                    rag_context = cached_context
                    self.terminal.log(f"    -> Using cached context for {file_info['path']} ⚡")
                else:
                    file_query = f"{file_info['purpose']} {file_info['path']} python implementation"
                    rag_context = self.rag_manager.get_context_for_code_generation(file_query, "python")

                    if rag_context:
                        # Cache with relevance based on context quality
                        relevance = min(1.0, len(rag_context) / 500)  # Higher relevance for longer context
                        self.context_cache.store_context(context_cache_key, rag_context, "file_planning", relevance)
                        self.terminal.log(f"    -> Context cached for future use")

            tasks_prompt = f"""
Break down this file into atomic micro-tasks for code generation:

File: {file_info['path']}
Purpose: {file_info['purpose']}
Project Context: {user_prompt}

{f"Reference Examples from Knowledge Base:\n{rag_context}\n" if rag_context else ""}

Return ONLY a JSON array of micro-tasks:
[
    {{"id": "imports_{file_info['path']}", "type": "imports", "description": "Import necessary modules"}},
    {{"id": "main_function_{file_info['path']}", "type": "function", "description": "Create main function"}}
]
"""

            # Stream micro-task creation
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(tasks_prompt):
                response_chunks.append(chunk)
                # Show progress every 8 chunks
                if len(response_chunks) % 8 == 0:
                    self.terminal.log(f"      -> Analyzing {file_info['path']}...")

            response = ''.join(response_chunks)

            try:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    file_tasks = json.loads(response[start:end])
                    for task in file_tasks:
                        task["file_path"] = file_info["path"]
                        task["file_purpose"] = file_info["purpose"]
                        task["rag_context"] = rag_context
                    micro_tasks.extend(file_tasks)
                    self.terminal.log(f"    -> Added {len(file_tasks)} micro-tasks")
                else:
                    raise ValueError("No JSON array found")
            except Exception as e:
                self.terminal.log(f"    -> Micro-task parsing failed: {e}")
                # Fallback task
                micro_tasks.append({
                    "id": f"implement_{file_info['path']}",
                    "file_path": file_info["path"],
                    "type": "complete_file",
                    "description": f"Implement complete {file_info['path']}",
                    "rag_context": rag_context
                })

        return micro_tasks

    async def _execute_micro_tasks_streaming(self, micro_tasks: list, plan: dict) -> dict:
        """NEW: Streaming micro-task execution with progress tracking"""
        project_name = plan["project_name"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_dir = self.output_dir / f"{project_name}_{timestamp}"
        project_dir.mkdir(exist_ok=True)

        generated_files = []
        code_snippets = {}

        # Group tasks by file
        tasks_by_file = {}
        for task in micro_tasks:
            file_path = task["file_path"]
            if file_path not in tasks_by_file:
                tasks_by_file[file_path] = []
            tasks_by_file[file_path].append(task)

        # Execute tasks file by file with streaming
        for file_idx, (file_path, file_tasks) in enumerate(tasks_by_file.items()):
            self.terminal.log(f"🔧 Processing file {file_idx + 1}/{len(tasks_by_file)}: {file_path}")
            code_snippets[file_path] = ""

            # Execute each micro-task for this file
            for task_idx, task in enumerate(file_tasks):
                # Update progress
                self.current_workflow_state["completed_tasks"] += 1
                progress = (self.current_workflow_state["completed_tasks"] /
                            self.current_workflow_state["total_tasks"]) * 100

                self.terminal.log(f"  -> [{progress:.0f}%] Executing: {task['description']}")

                # Smart context caching for code generation
                code_cache_key = f"{task['type']}_{task['file_path']}_code"
                cached_code_context = self.context_cache.get_context(code_cache_key, "code_generation")

                rag_context = task.get("rag_context", "")
                if not rag_context and cached_code_context:
                    rag_context = cached_code_context
                    self.terminal.log(f"    -> Using cached code context ⚡")

                code_prompt = f"""
Generate Python code for this micro-task:

Task: {task['description']}
File: {file_path}
Context: {code_snippets[file_path] if code_snippets[file_path] else 'First task for this file'}
Project: {plan['description']}

{f"Reference Examples and Best Practices:\n{rag_context}\n" if rag_context else ""}

Return ONLY Python code, no explanations:
"""

                self.terminal.log(f"    -> Calling Code LLM (streaming)...")

                # Stream code generation
                code_chunks = []
                async for chunk in self.llm_client.stream_chat(code_prompt):
                    code_chunks.append(chunk)
                    # Show streaming progress for code
                    if len(code_chunks) % 5 == 0:
                        self.terminal.log(f"      -> Generating code... ({len(''.join(code_chunks))} chars)")

                code_response = ''.join(code_chunks)
                code = self._clean_code_response(code_response)
                code_snippets[file_path] += f"\n# {task['description']}\n{code}\n"

                # Cache successful code context for similar tasks
                if len(code) > 50:  # Only cache substantial code
                    self.context_cache.store_context(code_cache_key, rag_context, "code_generation", 0.8)

                self.terminal.log(f"    -> Generated {len(code)} chars")

            # Assemble complete file with streaming
            self.terminal.log(f"📝 Assembling complete file: {file_path}")
            final_code = await self._assemble_final_file_streaming(file_path, code_snippets[file_path], plan)

            # Write file
            full_file_path = project_dir / file_path
            full_file_path.parent.mkdir(parents=True, exist_ok=True)
            full_file_path.write_text(final_code, encoding='utf-8')
            generated_files.append(str(full_file_path))

            self.terminal.log(f"  -> ✅ Written: {full_file_path}")
            self.file_generated.emit(str(full_file_path))

        return {
            "success": True,
            "project_dir": str(project_dir),
            "files": generated_files,
            "project_name": project_name,
            "file_count": len(generated_files)
        }

    async def _assemble_final_file_streaming(self, file_path: str, code_content: str, plan: dict) -> str:
        """NEW: Streaming file assembly with context caching"""
        # Check cache for assembly best practices
        assembly_cache_key = f"assembly_{file_path}_practices"
        cached_assembly_context = self.context_cache.get_context(assembly_cache_key, "assembly")

        rag_context = ""
        if self.rag_manager and self.rag_manager.is_ready:
            if cached_assembly_context:
                rag_context = cached_assembly_context
                self.terminal.log(f"  -> Using cached assembly practices ⚡")
            else:
                assembly_query = f"python file structure organization {file_path} best practices"
                rag_context = self.rag_manager.get_context_for_code_generation(assembly_query, "python")

                if rag_context:
                    self.context_cache.store_context(assembly_cache_key, rag_context, "assembly", 0.7)

        assemble_prompt = f"""
Assemble this code into a complete, working Python file:

File: {file_path}
Code snippets: {code_content}

{f"Best Practices Reference:\n{rag_context}\n" if rag_context else ""}

Requirements:
- Organize imports at the top
- Remove duplicates and conflicts
- Ensure proper indentation
- Add file-level docstring
- Make it runnable and functional

Return ONLY the complete Python file content:
"""

        self.terminal.log(f"  -> Final assembly pass (streaming)...")

        # Stream assembly
        assembly_chunks = []
        async for chunk in self.llm_client.stream_chat(assemble_prompt):
            assembly_chunks.append(chunk)
            if len(assembly_chunks) % 8 == 0:
                self.terminal.log(f"    -> Assembling... ({len(''.join(assembly_chunks))} chars)")

        final_code = ''.join(assembly_chunks)
        return self._clean_code_response(final_code)

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

    def _clean_code_response(self, code_response: str) -> str:
        if "```python" in code_response:
            code = code_response.split("```python")[1].split("```")[0].strip()
        elif "```" in code_response:
            code = code_response.split("```")[1].split("```")[0].strip()
        else:
            code = code_response.strip()
        return code

    def _on_file_changed(self, file_path: str, content: str):
        self.terminal.log(f"📝 File modified: {Path(file_path).name}")

    def get_workflow_stats(self) -> Dict:
        """Get current workflow statistics"""
        cache_stats = self.context_cache.get_stats()
        return {
            "workflow_state": self.current_workflow_state,
            "cache_stats": cache_stats
        }