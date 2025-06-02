# core/workflow_services.py - Modular AI Services

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from core.llm_client import LLMRole


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

        response = await self.llm_client.stream_chat(plan_prompt, LLMRole.PLANNER)
        response_text = ''.join([chunk async for chunk in response])

        try:
            plan = self._extract_json(response_text)
            return plan
        except:
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

        response = await self.llm_client.stream_chat(task_prompt, LLMRole.PLANNER)
        response_text = ''.join([chunk async for chunk in response])

        try:
            tasks = self._extract_json(response_text)
            for task in tasks:
                task["file_path"] = file_path
            return tasks
        except:
            return [{"id": f"implement_{file_path}", "type": "complete",
                     "description": f"Implement {file_path}", "file_path": file_path}]

    async def review_code(self, file_path: str, code: str, tasks: List[dict]) -> Tuple[bool, str]:
        """Review assembled code"""
        review_prompt = f"""
Review this assembled code for {file_path}:

CODE:
{code}

ORIGINAL TASKS: {len(tasks)} completed

Return JSON:
{{
    "approved": true/false,
    "feedback": "Brief feedback",
    "issues": ["Any specific issues found"]
}}

Be constructive but thorough.
"""

        response = await self.llm_client.stream_chat(review_prompt, LLMRole.REVIEWER)
        response_text = ''.join([chunk async for chunk in response])

        try:
            review = self._extract_json(response_text)
            return review.get("approved", True), review.get("feedback", "Review completed")
        except:
            return True, "Review completed"

    async def _get_planning_context(self, prompt: str, cache):
        if self.rag_manager and self.rag_manager.is_ready:
            return self.rag_manager.get_context_for_code_generation(f"project planning {prompt}", "python")
        return ""

    async def _get_file_context(self, file_path: str, plan: dict, cache):
        if self.rag_manager and self.rag_manager.is_ready:
            return self.rag_manager.get_context_for_code_generation(f"{file_path} implementation patterns", "python")
        return ""

    def _extract_json(self, text: str) -> dict:
        start, end = text.find('{'), text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise ValueError("No JSON found")

    def _create_fallback_plan(self, prompt: str) -> dict:
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

        response = await self.llm_client.stream_chat(code_prompt, LLMRole.CODER)
        code_chunks = [chunk async for chunk in response]
        code = ''.join(code_chunks)

        return self._clean_code(code)

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
                    if progress_callback:
                        progress_callback(f"[{task['file_path']}] ⚙️ {task['description']}")

                    code = await self.execute_micro_task(task, file_context, context_cache)
                    return {"task": task, "code": code}

            parallel_results = await asyncio.gather(
                *[execute_with_semaphore(task) for task in independent_tasks],
                return_exceptions=True
            )

            # Filter out exceptions
            results.extend([r for r in parallel_results if not isinstance(r, Exception)])

        # Execute dependent tasks sequentially
        for task in dependent_tasks:
            if progress_callback:
                progress_callback(f"[{task['file_path']}] ⚙️ {task['description']}")

            code = await self.execute_micro_task(task, file_context, context_cache)
            results.append({"task": task, "code": code})

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
            code = code.split("```")[1].split("```")[0]
        return code.strip()


class AssemblerService:
    """📄 Pure Assembly Service - Combine Micro-tasks"""

    def __init__(self, llm_client, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager

    async def assemble_file(self, file_path: str, task_results: List[dict],
                            plan: dict, context_cache) -> str:
        """Assemble micro-task results into cohesive file"""
        assembly_context = await self._get_assembly_context(file_path, plan, context_cache)

        # Prepare code sections
        code_sections = []
        for result in task_results:
            task = result['task']
            code = result['code']
            code_sections.append(f"# {task['description']}\n{code}")

        assembly_prompt = f"""
Assemble these micro-task results into a cohesive Python file:

FILE: {file_path}
PROJECT: {plan['project_name']}

MICRO-TASK RESULTS:
{chr(10).join(code_sections)}

ASSEMBLY CONTEXT: {assembly_context if assembly_context else "Standard Python structure"}

Requirements:
1. Organize imports at top (remove duplicates)
2. Add file-level docstring
3. Maintain logical flow
4. Ensure components work together
5. Fix any syntax issues
6. Follow PEP 8
7. Make file functional and executable

Return complete, assembled Python file:
"""

        response = await self.llm_client.stream_chat(assembly_prompt, LLMRole.ASSEMBLER)
        code_chunks = [chunk async for chunk in response]
        assembled = ''.join(code_chunks)

        return self._clean_code(assembled)

    async def _get_assembly_context(self, file_path: str, plan: dict, cache):
        if self.rag_manager and self.rag_manager.is_ready:
            return self.rag_manager.get_context_for_code_generation(f"python file organization {file_path}", "python")
        return ""

    def _clean_code(self, code: str) -> str:
        """Extract clean code from LLM response"""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        return code.strip()


class WorkflowOrchestrator:
    """🎼 Orchestrates All Services - Parallel File Processing"""

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
        """Execute complete workflow with parallel file processing"""
        try:
            self._log("🚀 Starting Enhanced Micro-Task Workflow...")

            # Stage 1: Planning
            self._log("🧠 PLANNER: Creating project architecture...")
            plan = await self.planner.create_project_plan(user_prompt, self.context_cache)

            project_name = plan["project_name"]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_dir = output_dir / f"{project_name}_{timestamp}"
            project_dir.mkdir(exist_ok=True)

            self._log(f"📋 Project planned: {project_name} ({len(plan.get('files', {}))} files)")

            # Stage 2: Parallel File Processing
            self._log("⚡ Starting parallel file processing...")

            files = plan.get("files", {"main.py": {"priority": 1}})
            file_tasks = []

            # Create tasks for each file (can run in parallel)
            for file_path, file_info in files.items():
                task = asyncio.create_task(
                    self._process_single_file(file_path, file_info, plan, project_dir)
                )
                file_tasks.append(task)

            # Execute all files in parallel
            results = await asyncio.gather(*file_tasks, return_exceptions=True)

            # Process results
            generated_files = []
            for result in results:
                if isinstance(result, Exception):
                    self._log(f"❌ File processing failed: {result}")
                else:
                    generated_files.append(result)

            self._log(f"✅ Workflow completed! Generated {len(generated_files)} files")

            return {
                "success": True,
                "project_dir": str(project_dir),
                "files": generated_files,
                "project_name": project_name,
                "file_count": len(generated_files)
            }

        except Exception as e:
            self._log(f"❌ Workflow failed: {e}")
            return {"success": False, "error": str(e)}

    async def _process_single_file(self, file_path: str, file_info: dict,
                                   plan: dict, project_dir: Path) -> str:
        """Process a single file with micro-tasks"""
        self._log(f"[{file_path}] 🧠 Creating micro-tasks...")

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

        # Step 3: Assemble code
        self._log(f"[{file_path}] 📄 Assembling code...")
        assembled_code = await self.assembler.assemble_file(
            file_path, task_results, plan, self.context_cache
        )

        # Step 4: Review code
        self._log(f"[{file_path}] 🔍 Reviewing code...")
        approved, feedback = await self.planner.review_code(file_path, assembled_code, tasks)

        if approved:
            self._log(f"[{file_path}] ✅ Code approved")
        else:
            self._log(f"[{file_path}] ⚠️ Review feedback: {feedback}")

        # Step 5: Write file
        full_path = project_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(assembled_code, encoding='utf-8')

        self._log(f"[{file_path}] 💾 File written successfully")
        return str(full_path)

    def _log(self, message: str):
        """Log to terminal if available"""
        if self.terminal:
            self.terminal.log(message)