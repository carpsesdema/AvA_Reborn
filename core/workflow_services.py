# core/workflow_services.py - FIXED ASYNC GENERATOR ISSUES & OPTIMIZED

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from core.llm_client import LLMRole
from core.assembler_service import AssemblerService


class PlannerService:
    """🧠 Pure Planning Service - Single Responsibility"""

    def __init__(self, llm_client, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager

    async def create_project_plan(self, user_prompt: str, context_cache) -> dict:
        """Create high-level project plan"""
        rag_context = await self._get_planning_context(user_prompt, context_cache)

        plan_prompt = f"""
You are a Senior Software Architect. Create a clean project plan:

REQUEST: {user_prompt}

CONTEXT: {rag_context if rag_context else "Use Python best practices"}

Return JSON:
{{
    "project_name": "descriptive_snake_case_name",
    "description": "Clear description",
    "architecture_type": "cli|web_app|gui|library",
    "files": {{
        "main.py": {{"priority": 1, "description": "Main entry point"}},
        "utils.py": {{"priority": 2, "description": "Helper functions"}}
    }},
    "dependencies": ["requests", "pathlib"]
}}

Maximum 6 files for clean workflow.
"""

        try:
            # FIXED: Properly consume the async generator
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(plan_prompt, LLMRole.PLANNER):
                response_chunks.append(chunk)
                # OPTIMIZATION: Allow UI updates during streaming
                if len(response_chunks) % 10 == 0:
                    await asyncio.sleep(0.01)

            response_text = ''.join(response_chunks)
            plan = self._extract_json(response_text)
            return plan

        except Exception as e:
            print(f"Planning failed: {e}")
            return self._create_fallback_plan(user_prompt)

    async def create_micro_tasks(self, file_path: str, plan: dict, context_cache) -> List[dict]:
        """Break file into atomic micro-tasks"""
        file_context = await self._get_file_context(file_path, plan, context_cache)

        task_prompt = f"""
Break {file_path} into ATOMIC micro-tasks (5-15 lines each):

PROJECT: {plan['project_name']}
FILE: {file_path}

CONTEXT: {file_context if file_context else "Standard Python practices"}

Return JSON array:
[
    {{
        "id": "imports_{file_path.replace('.py', '')}",
        "type": "imports", 
        "description": "Import required modules",
        "priority": 1,
        "expected_lines": 5
    }},
    {{
        "id": "main_function_{file_path.replace('.py', '')}",
        "type": "function",
        "description": "Create main function with logic", 
        "priority": 2,
        "expected_lines": 12
    }}
]

Keep tasks ATOMIC and independent.
"""

        try:
            # FIXED: Properly consume the async generator
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(task_prompt, LLMRole.PLANNER):
                response_chunks.append(chunk)
                # OPTIMIZATION: Allow UI updates during streaming
                if len(response_chunks) % 10 == 0:
                    await asyncio.sleep(0.01)

            response_text = ''.join(response_chunks)
            tasks = self._extract_json(response_text)

            for task in tasks:
                task["file_path"] = file_path
            return tasks

        except Exception as e:
            print(f"Task creation failed for {file_path}: {e}")
            return [{"id": f"implement_{file_path}", "type": "complete",
                     "description": f"Implement {file_path}", "file_path": file_path}]

    async def review_code(self, file_path: str, code: str, tasks: List[dict]) -> Tuple[bool, str]:
        """Review assembled code - MOVED TO ASSEMBLER SERVICE"""
        # This method is now deprecated - AssemblerService handles review
        return True, "Review handled by AssemblerService"

    async def _get_planning_context(self, prompt: str, cache):
        if self.rag_manager and self.rag_manager.is_ready:
            return self.rag_manager.get_context_for_code_generation(f"project planning {prompt}", "python")
        return ""

    async def _get_file_context(self, file_path: str, plan: dict, cache):
        if self.rag_manager and self.rag_manager.is_ready:
            return self.rag_manager.get_context_for_code_generation(f"{file_path} implementation patterns", "python")
        return ""

    def _extract_json(self, text: str) -> dict:
        """Extract JSON with better error handling"""
        try:
            start, end = text.find('{'), text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON extraction failed: {e}")

        # Try to find JSON array
        try:
            start, end = text.find('['), text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            pass

        raise ValueError("No valid JSON found in response")

    def _create_fallback_plan(self, prompt: str) -> dict:
        """Create fallback plan when LLM planning fails"""
        # Determine project type from prompt
        prompt_lower = prompt.lower()

        if "gui" in prompt_lower or "pyside" in prompt_lower or "tkinter" in prompt_lower:
            return {
                "project_name": "gui_application",
                "description": prompt,
                "architecture_type": "gui",
                "files": {
                    "main.py": {"priority": 1, "description": "Main GUI application"},
                    "ui_components.py": {"priority": 2, "description": "UI components and widgets"}
                },
                "dependencies": ["PySide6"]
            }
        elif "web" in prompt_lower or "flask" in prompt_lower or "fastapi" in prompt_lower:
            return {
                "project_name": "web_application",
                "description": prompt,
                "architecture_type": "web_app",
                "files": {
                    "main.py": {"priority": 1, "description": "Main web application"},
                    "routes.py": {"priority": 2, "description": "Web routes"}
                },
                "dependencies": ["flask"]
            }
        else:
            return {
                "project_name": "generated_project",
                "description": prompt,
                "architecture_type": "cli",
                "files": {"main.py": {"priority": 1, "description": "Main file"}},
                "dependencies": []
            }


class CoderService:
    """⚙️ Pure Coding Service - Parallel Task Execution"""

    def __init__(self, llm_client, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager

    async def execute_micro_task(self, task: dict, file_context: str, context_cache) -> str:
        """Execute single atomic micro-task"""
        task_context = await self._get_task_context(task, context_cache)

        code_prompt = f"""
Generate ONLY the code for this atomic task:

TASK: {task['description']}
TYPE: {task.get('type', 'general')}
EXPECTED LINES: ~{task.get('expected_lines', 10)}

CONTEXT: {task_context if task_context else "Standard Python"}
FILE CONTEXT: {file_context}

Requirements:
- Generate ONLY this specific task's code
- Follow PEP 8 standards
- Include docstrings for functions/classes
- Add type hints where appropriate
- Keep it atomic and focused

Return ONLY Python code:
"""

        try:
            # FIXED: Properly consume the async generator with optimization
            response_chunks = []
            async for chunk in self.llm_client.stream_chat(code_prompt, LLMRole.CODER):
                response_chunks.append(chunk)
                # OPTIMIZATION: Allow UI updates during streaming
                if len(response_chunks) % 5 == 0:
                    await asyncio.sleep(0.01)

            code = ''.join(response_chunks)
            return self._clean_code(code)

        except Exception as e:
            print(f"Micro-task execution failed: {e}")
            # Return a basic fallback implementation
            return f"# TODO: Implement {task['description']}\npass"

    async def execute_tasks_parallel(self, tasks: List[dict], file_context: str,
                                     context_cache, progress_callback=None) -> List[dict]:
        """Execute multiple tasks in parallel where safe"""
        # Group by dependencies for safe parallelism
        independent_tasks = [t for t in tasks if not t.get('dependencies')]
        dependent_tasks = [t for t in tasks if t.get('dependencies')]

        results = []

        # Execute independent tasks in parallel
        if independent_tasks:
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent tasks

            async def execute_with_semaphore(task):
                async with semaphore:
                    try:
                        if progress_callback:
                            progress_callback(f"[{task['file_path']}] ⚙️ {task['description']}")

                        code = await self.execute_micro_task(task, file_context, context_cache)
                        return {"task": task, "code": code}
                    except Exception as e:
                        print(f"Task execution failed: {e}")
                        return {"task": task, "code": f"# ERROR: {e}\npass"}

            # FIXED: Use asyncio.gather with proper exception handling
            try:
                parallel_results = await asyncio.gather(
                    *[execute_with_semaphore(task) for task in independent_tasks],
                    return_exceptions=True
                )

                # Filter out exceptions and add successful results
                for result in parallel_results:
                    if not isinstance(result, Exception):
                        results.append(result)
                    else:
                        print(f"Parallel task failed: {result}")

            except Exception as e:
                print(f"Parallel execution failed: {e}")

        # Execute dependent tasks sequentially
        for task in dependent_tasks:
            try:
                if progress_callback:
                    progress_callback(f"[{task['file_path']}] ⚙️ {task['description']}")

                code = await self.execute_micro_task(task, file_context, context_cache)
                results.append({"task": task, "code": code})

            except Exception as e:
                print(f"Dependent task execution failed: {e}")
                results.append({"task": task, "code": f"# ERROR: {e}\npass"})

        return results

    async def _get_task_context(self, task: dict, cache):
        if self.rag_manager and self.rag_manager.is_ready:
            task_type = task.get("type", "general")
            return self.rag_manager.get_context_for_code_generation(f"python {task_type} examples", "python")
        return ""

    def _clean_code(self, code: str) -> str:
        """Extract clean code from LLM response"""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 3:
                code = parts[1]
        return code.strip()


class WorkflowOrchestrator:
    """🎼 Orchestrates All Services - Parallel File Processing with Robust Assembly & Review"""

    def __init__(self, planner: PlannerService, coder: CoderService,
                 assembler: AssemblerService, terminal=None):
        self.planner = planner
        self.coder = coder
        self.assembler = assembler
        self.terminal = terminal

        # Context cache for RAG optimization
        from core.workflow_engine import ContextCache
        self.context_cache = ContextCache()

    async def execute_workflow(self, user_prompt: str, output_dir: Path) -> dict:
        """Execute complete workflow with parallel file processing and mandatory review"""
        try:
            self._log("🚀 Starting Enhanced Micro-Task Workflow with Robust Assembly...")

            # Stage 1: Planning
            self._log("🧠 PLANNER: Creating project architecture...")
            plan = await self.planner.create_project_plan(user_prompt, self.context_cache)

            project_name = plan["project_name"]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_dir = output_dir / f"{project_name}_{timestamp}"
            project_dir.mkdir(exist_ok=True)

            self._log(f"📋 Project planned: {project_name} ({len(plan.get('files', {}))} files)")

            # Stage 2: Parallel File Processing with Review
            self._log("⚡ Starting parallel file processing with mandatory review...")

            files = plan.get("files", {"main.py": {"priority": 1}})

            # OPTIMIZATION: Process files in smaller batches to maintain responsiveness
            file_tasks = []
            for file_path, file_info in files.items():
                task = self._process_single_file_with_review(file_path, file_info, plan, project_dir)
                file_tasks.append(task)

            # FIXED: Execute all files with proper exception handling
            results = await asyncio.gather(*file_tasks, return_exceptions=True)

            # Process results
            generated_files = []
            review_failures = []

            for result in results:
                if isinstance(result, Exception):
                    self._log(f"❌ File processing failed: {result}")
                elif result.get("review_approved", False):
                    generated_files.append(result["file_path"])
                    self._log(f"✅ {result['file_name']} - Review APPROVED")
                else:
                    review_failures.append(result)
                    self._log(f"⚠️ {result['file_name']} - Review FAILED: {result.get('review_feedback', 'Unknown')}")

            # Summary
            self._log(f"✅ Workflow completed! Generated {len(generated_files)} files")
            if review_failures:
                self._log(f"⚠️ {len(review_failures)} files failed review and need attention")

            return {
                "success": True,
                "project_dir": str(project_dir),
                "files": generated_files,
                "project_name": project_name,
                "file_count": len(generated_files),
                "review_failures": review_failures
            }

        except Exception as e:
            self._log(f"❌ Workflow failed: {e}")
            import traceback
            self._log(f"📝 Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    async def _process_single_file_with_review(self, file_path: str, file_info: dict,
                                               plan: dict, project_dir: Path) -> dict:
        """Process a single file with micro-tasks and mandatory review"""
        self._log(f"[{file_path}] 🧠 Creating micro-tasks...")

        try:
            # Step 1: Plan micro-tasks
            tasks = await self.planner.create_micro_tasks(file_path, plan, self.context_cache)
            self._log(f"[{file_path}] 📋 Created {len(tasks)} micro-tasks")

            # Step 2: Execute micro-tasks (parallel where safe)
            self._log(f"[{file_path}] ⚙️ Executing micro-tasks...")

            def progress_callback(message):
                self._log(message)

            task_results = await self.coder.execute_tasks_parallel(
                tasks, "", self.context_cache, progress_callback
            )

            # Step 3: ROBUST ASSEMBLY WITH MANDATORY REVIEW
            self._log(f"[{file_path}] 📄 Smart assembly with review...")

            assembled_code, review_approved, review_feedback = await self.assembler.assemble_file(
                file_path, task_results, plan, self.context_cache
            )

            # Step 4: Handle review results
            if review_approved:
                self._log(f"[{file_path}] ✅ Assembly review APPROVED - writing file")

                # Write file
                full_path = project_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(assembled_code, encoding='utf-8')

                return {
                    "success": True,
                    "file_path": str(full_path),
                    "file_name": file_path,
                    "review_approved": True,
                    "review_feedback": review_feedback
                }
            else:
                self._log(f"[{file_path}] ❌ Assembly review FAILED")
                self._log(f"[{file_path}] 📝 Review feedback: {review_feedback}")

                # Still write file but mark as needing attention
                full_path = project_dir / f"{file_path}.NEEDS_REVIEW"
                full_path.parent.mkdir(parents=True, exist_ok=True)

                # Add review feedback as comment at top of file
                reviewed_code = f"""# REVIEW FAILED - NEEDS ATTENTION
# Review Feedback: {review_feedback}
# Original file: {file_path}

{assembled_code}
"""
                full_path.write_text(reviewed_code, encoding='utf-8')

                return {
                    "success": False,
                    "file_path": str(full_path),
                    "file_name": file_path,
                    "review_approved": False,
                    "review_feedback": review_feedback,
                    "needs_attention": True
                }

        except Exception as e:
            self._log(f"[{file_path}] ❌ Processing failed: {e}")
            import traceback
            self._log(f"[{file_path}] 📝 Error traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "file_path": None,
                "file_name": file_path,
                "review_approved": False,
                "review_feedback": f"Processing error: {e}",
                "error": str(e)
            }

    def _log(self, message: str):
        """Log to terminal if available"""
        if self.terminal and hasattr(self.terminal, 'log'):
            self.terminal.log(message)
        else:
            print(message)