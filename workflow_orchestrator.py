# workflow_orchestrator.py

import asyncio
import json
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
from PySide6.QtCore import QObject, Signal
from utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MicroTask:
    """Represents a single atomic coding task"""
    id: str
    file_path: str
    task_type: str  # "function", "class", "import", "docstring", etc.
    description: str
    context: str  # Previous code in file or related context
    dependencies: List[str]  # IDs of tasks this depends on
    status: TaskStatus = TaskStatus.PENDING
    generated_code: str = ""
    error_message: str = ""


@dataclass
class FileTask:
    """Represents all tasks needed to generate a complete file"""
    file_path: str
    micro_tasks: List[MicroTask]
    assembled_content: str = ""
    status: TaskStatus = TaskStatus.PENDING


class WorkflowOrchestrator(QObject):
    """
    Coordinates the entire workflow from user input to final code generation.
    Emits signals to update UI components in real-time.
    """

    # Signals for UI updates
    planning_started = Signal(str)  # user_prompt
    planning_completed = Signal(str)  # plan_json
    task_started = Signal(str, str)  # task_id, description
    task_completed = Signal(str, str)  # task_id, generated_code
    file_assembled = Signal(str, str)  # file_path, content
    workflow_completed = Signal(dict)  # final_results
    error_occurred = Signal(str, str)  # component, error_message

    def __init__(self, planner, assembler, rag, file_manager):
        super().__init__()
        self.planner = planner
        self.assembler = assembler
        self.rag = rag
        self.file_manager = file_manager
        self.current_workflow = None
        self.file_tasks: Dict[str, FileTask] = {}

    async def execute_workflow(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main workflow execution method.
        Returns final results and emits signals for UI updates.
        """
        try:
            logger.info(f"Starting workflow for prompt: {user_prompt}")
            self.planning_started.emit(user_prompt)

            # Step 1: Generate plan with micro-tasks
            plan = await self._generate_plan(user_prompt)
            self.planning_completed.emit(json.dumps(plan, indent=2))

            # Step 2: Execute micro-tasks for each file
            await self._execute_file_tasks(plan)

            # Step 3: Final assembly and review
            results = await self._finalize_workflow()

            self.workflow_completed.emit(results)
            return results

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.error_occurred.emit("orchestrator", str(e))
            raise

    async def _generate_plan(self, user_prompt: str) -> Dict[str, Any]:
        """Generate structured plan with micro-tasks for each file"""

        # Enhanced prompt for better micro-task generation
        planning_prompt = f"""
        Create a detailed development plan for: {user_prompt}

        Return a JSON structure with this format:
        {{
            "project_overview": "Brief description",
            "files": [
                {{
                    "path": "relative/path/to/file.py",
                    "purpose": "What this file does",
                    "micro_tasks": [
                        {{
                            "id": "unique_task_id",
                            "type": "function|class|import|docstring|test",
                            "description": "Detailed task description",
                            "dependencies": ["other_task_ids"],
                            "context": "Any relevant context or requirements"
                        }}
                    ]
                }}
            ],
            "dependencies": {{
                "external_packages": ["package1", "package2"],
                "internal_modules": ["module1", "module2"]
            }}
        }}

        Make micro-tasks ATOMIC - each should produce a single, complete code unit.
        """

        plan_response = await self.planner.generate_plan(planning_prompt)
        plan = json.loads(plan_response)

        # Convert plan to internal task structure
        self._create_file_tasks(plan)

        return plan

    def _create_file_tasks(self, plan: Dict[str, Any]):
        """Convert plan JSON to internal FileTask and MicroTask objects"""
        self.file_tasks.clear()

        for file_info in plan["files"]:
            micro_tasks = []
            for task_info in file_info["micro_tasks"]:
                micro_task = MicroTask(
                    id=task_info["id"],
                    file_path=file_info["path"],
                    task_type=task_info["type"],
                    description=task_info["description"],
                    context=task_info.get("context", ""),
                    dependencies=task_info.get("dependencies", [])
                )
                micro_tasks.append(micro_task)

            file_task = FileTask(
                file_path=file_info["path"],
                micro_tasks=micro_tasks
            )
            self.file_tasks[file_info["path"]] = file_task

    async def _execute_file_tasks(self, plan: Dict[str, Any]):
        """Execute micro-tasks for each file, respecting dependencies"""

        # Create dependency graph and execution order
        execution_order = self._calculate_execution_order()

        for file_path in execution_order:
            file_task = self.file_tasks[file_path]
            logger.info(f"Processing file: {file_path}")

            # Execute micro-tasks in dependency order
            await self._execute_micro_tasks(file_task)

            # Assemble complete file
            await self._assemble_file(file_task)

    async def _execute_micro_tasks(self, file_task: FileTask):
        """Execute individual micro-tasks for a file"""

        # Sort tasks by dependencies
        sorted_tasks = self._sort_tasks_by_dependencies(file_task.micro_tasks)

        for task in sorted_tasks:
            try:
                self.task_started.emit(task.id, task.description)
                task.status = TaskStatus.IN_PROGRESS

                # Get RAG context if available
                rag_context = self.rag.query(task.description) if self.rag else ""

                # Build context from previously completed tasks
                context = self._build_task_context(task, file_task)

                # Generate code for this micro-task
                generated_code = await self.assembler.generate_micro_task_code(
                    task, context, rag_context
                )

                task.generated_code = generated_code
                task.status = TaskStatus.COMPLETED

                self.task_completed.emit(task.id, generated_code)

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                self.error_occurred.emit(f"task_{task.id}", str(e))
                raise

    async def _assemble_file(self, file_task: FileTask):
        """Assemble all micro-task outputs into a complete file"""

        # Collect all generated code snippets
        code_snippets = [task.generated_code for task in file_task.micro_tasks
                         if task.status == TaskStatus.COMPLETED]

        # Use assembler to create cohesive file
        assembled_content = await self.assembler.assemble_file(
            file_task.file_path, code_snippets
        )

        file_task.assembled_content = assembled_content
        file_task.status = TaskStatus.COMPLETED

        # Write to disk
        self.file_manager.write_file(file_task.file_path, assembled_content)

        # Emit signal for UI update
        self.file_assembled.emit(file_task.file_path, assembled_content)

    def _calculate_execution_order(self) -> List[str]:
        """Calculate optimal order for file execution based on dependencies"""
        # Simple implementation - can be enhanced with proper topological sort
        return list(self.file_tasks.keys())

    def _sort_tasks_by_dependencies(self, tasks: List[MicroTask]) -> List[MicroTask]:
        """Sort tasks to respect dependencies"""
        # Simple implementation - enhance with proper dependency resolution
        return sorted(tasks, key=lambda t: len(t.dependencies))

    def _build_task_context(self, current_task: MicroTask, file_task: FileTask) -> str:
        """Build context string from previously completed tasks in the same file"""
        context_parts = [current_task.context]

        for task in file_task.micro_tasks:
            if (task.status == TaskStatus.COMPLETED and
                    task.id in current_task.dependencies):
                context_parts.append(f"# From {task.id}:\n{task.generated_code}")

        return "\n\n".join(context_parts)

    async def _finalize_workflow(self) -> Dict[str, Any]:
        """Final review and cleanup"""
        results = {
            "files_generated": len([f for f in self.file_tasks.values()
                                    if f.status == TaskStatus.COMPLETED]),
            "files_failed": len([f for f in self.file_tasks.values()
                                 if f.status == TaskStatus.FAILED]),
            "file_paths": list(self.file_tasks.keys()),
            "total_tasks": sum(len(f.micro_tasks) for f in self.file_tasks.values())
        }

        logger.info(f"Workflow completed: {results}")
        return results