# core/workflow_engine.py - Workflow logic ONLY

import json
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import QObject, QThread, pyqtSignal


class WorkflowThread(QThread):
    """Thread for running workflow without blocking UI"""

    def __init__(self, workflow_engine, user_prompt):
        super().__init__()
        self.workflow_engine = workflow_engine
        self.user_prompt = user_prompt

    def run(self):
        """Execute workflow in separate thread"""
        self.workflow_engine._execute_workflow_internal(self.user_prompt)


class WorkflowEngine(QObject):
    """
    SINGLE RESPONSIBILITY: Coordinate the AI workflow process
    Plan -> Micro-tasks -> Code Generation -> Assembly
    Does NOT handle UI, LLM calls directly, or file display
    """

    def __init__(self, llm_client, terminal_window, code_viewer):
        super().__init__()
        self.llm_client = llm_client
        self.terminal = terminal_window
        self.code_viewer = code_viewer
        self.output_dir = Path("./generated_projects")
        self.output_dir.mkdir(exist_ok=True)

    def execute_workflow(self, user_prompt: str):
        """Start workflow execution in separate thread"""
        self.terminal.log("🚀 Starting AvA workflow...")
        self.terminal.log(f"📝 User request: {user_prompt}")

        # Run in separate thread to avoid blocking UI
        self.workflow_thread = WorkflowThread(self, user_prompt)
        self.workflow_thread.start()

    def _execute_workflow_internal(self, user_prompt: str):
        """Internal workflow execution (runs in thread)"""

        try:
            # Step 1: High-level planning
            self.terminal.log("🧠 PLANNER: Creating high-level plan...")
            plan = self._create_plan(user_prompt)

            # Step 2: Break into micro-tasks
            self.terminal.log("📋 PLANNER: Breaking down into micro-tasks...")
            micro_tasks = self._create_micro_tasks(plan, user_prompt)

            # Step 3: Execute micro-tasks
            self.terminal.log("⚙️ ASSEMBLER: Executing micro-tasks...")
            files = self._execute_micro_tasks(micro_tasks, plan)

            # Step 4: Show in code viewer
            self.terminal.log("📄 CODE VIEWER: Loading generated files...")
            self.code_viewer.load_project(files["project_dir"])

            self.terminal.log("✅ Workflow completed successfully!")

        except Exception as e:
            self.terminal.log(f"❌ Workflow failed: {e}")

    def _create_plan(self, user_prompt: str) -> dict:
        """Create high-level plan using Planner LLM"""

        plan_prompt = f"""
Act as a software architect. Create a high-level plan for: {user_prompt}

Return ONLY a JSON structure:
{{
    "project_name": "snake_case_name",
    "description": "Brief description",
    "architecture": "Brief architecture overview",
    "files_needed": [
        {{"path": "main.py", "purpose": "Main entry point"}},
        {{"path": "utils.py", "purpose": "Utility functions"}}
    ]
}}

Keep it simple - maximum 3-5 files for a working prototype.
"""

        self.terminal.log("  -> Calling Planner LLM...")
        response = self.llm_client.chat(plan_prompt)
        self.terminal.log(f"  -> Planner response received ({len(response)} chars)")

        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                plan = json.loads(response[start:end])
                self.terminal.log(f"  -> Plan created: {plan['project_name']}")
                return plan
        except Exception as e:
            self.terminal.log(f"  -> JSON parsing failed: {e}")

        # Fallback plan
        plan = {
            "project_name": "generated_project",
            "description": user_prompt,
            "files_needed": [{"path": "main.py", "purpose": "Main implementation"}]
        }
        self.terminal.log("  -> Using fallback plan")
        return plan

    def _create_micro_tasks(self, plan: dict, user_prompt: str) -> list:
        """Break plan into atomic micro-tasks"""

        micro_tasks = []

        for file_info in plan["files_needed"]:
            tasks_prompt = f"""
Break down this file into atomic micro-tasks for code generation:

File: {file_info['path']}
Purpose: {file_info['purpose']}
Project: {user_prompt}

Return ONLY a JSON array of micro-tasks:
[
    {{"id": "imports", "type": "imports", "description": "Import necessary modules"}},
    {{"id": "main_function", "type": "function", "description": "Create main function"}},
    {{"id": "helper_class", "type": "class", "description": "Create helper class"}}
]

Make each task atomic - one specific code element.
"""

            self.terminal.log(f"  -> Creating micro-tasks for {file_info['path']}")
            response = self.llm_client.chat(tasks_prompt)

            try:
                # Extract JSON array
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    file_tasks = json.loads(response[start:end])
                    for task in file_tasks:
                        task["file_path"] = file_info["path"]
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
                    "description": f"Implement complete {file_info['path']}"
                })
                self.terminal.log(f"    -> Added fallback task")

        return micro_tasks

    def _execute_micro_tasks(self, micro_tasks: list, plan: dict) -> dict:
        """Execute each micro-task with specialized Code LLM"""

        # Create project directory
        project_name = plan["project_name"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_dir = self.output_dir / f"{project_name}_{timestamp}"
        project_dir.mkdir(exist_ok=True)

        generated_files = []
        code_snippets = {}

        # Execute micro-tasks
        for task in micro_tasks:
            self.terminal.log(f"  -> Executing: {task['description']}")

            code_prompt = f"""
You are a specialized Python code generator. Generate ONLY the code for this micro-task:

Task: {task['description']}
Type: {task['type']}
File: {task['file_path']}

Context: Previous code snippets for this file:
{code_snippets.get(task['file_path'], 'None yet')}

Requirements:
- Write clean, working Python code
- Include proper imports if needed
- Add docstrings for functions/classes
- Follow Python best practices

Return ONLY Python code, no explanations or markdown:
"""

            self.terminal.log(f"    -> Calling Code LLM...")
            code_response = self.llm_client.chat(code_prompt)

            # Clean up code response
            code = self._clean_code_response(code_response)

            # Accumulate code for this file
            if task["file_path"] not in code_snippets:
                code_snippets[task["file_path"]] = ""
            code_snippets[task["file_path"]] += f"\n# {task['description']}\n{code}\n"

            self.terminal.log(f"    -> Generated {len(code)} chars")

        # Assemble complete files
        self.terminal.log("🔧 ASSEMBLER: Assembling complete files...")

        for file_path, code_content in code_snippets.items():
            full_file_path = project_dir / file_path

            # Final assembly pass
            assemble_prompt = f"""
Assemble this code into a complete, working Python file:

File: {file_path}
Code snippets:
{code_content}

Requirements:
- Organize imports at top
- Remove duplicates
- Ensure proper indentation
- Make it runnable
- Add if __name__ == "__main__": block if appropriate

Return ONLY the complete Python file content:
"""

            self.terminal.log(f"  -> Assembling {file_path}...")
            final_code = self.llm_client.chat(assemble_prompt)

            # Clean final code
            final_code = self._clean_code_response(final_code)

            # Write file
            full_file_path.write_text(final_code, encoding='utf-8')
            generated_files.append(str(full_file_path))

            self.terminal.log(f"  -> Wrote {full_file_path}")

            # Show snippet in terminal
            preview = final_code[:300] + "..." if len(final_code) > 300 else final_code
            self.terminal.log(f"\n📄 {file_path}:")
            self.terminal.log("─" * 50)
            self.terminal.log(preview)
            self.terminal.log("─" * 50)

        return {
            "success": True,
            "project_dir": str(project_dir),
            "files": generated_files
        }

    def _clean_code_response(self, code_response: str) -> str:
        """Clean up LLM code response"""

        # Remove markdown code blocks
        if "```python" in code_response:
            code = code_response.split("```python")[1].split("```")[0].strip()
        elif "```" in code_response:
            code = code_response.split("```")[1].split("```")[0].strip()
        else:
            code = code_response.strip()

        return code