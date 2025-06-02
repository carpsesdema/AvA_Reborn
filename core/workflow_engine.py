# core/workflow_engine.py - Enhanced Workflow logic with CodeViewer integration

import json
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import QObject, QThread, Signal


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
    Enhanced Workflow Engine with CodeViewer integration
    SINGLE RESPONSIBILITY: Coordinate the AI workflow process
    Plan -> Micro-tasks -> Code Generation -> Assembly -> Display
    """

    # Signals for UI updates
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    file_generated = Signal(str)  # NEW: Signal for when a file is generated
    project_loaded = Signal(str)  # NEW: Signal for when project is ready to view

    def __init__(self, llm_client, terminal_window, code_viewer):
        super().__init__()
        self.llm_client = llm_client
        self.terminal = terminal_window
        self.code_viewer = code_viewer
        self.output_dir = Path("./generated_projects")
        self.output_dir.mkdir(exist_ok=True)

        # Connect code viewer signals
        self._connect_code_viewer()

    def _connect_code_viewer(self):
        """Connect code viewer signals"""
        if self.code_viewer:
            # Connect file change notifications
            self.code_viewer.file_changed.connect(self._on_file_changed)

    def execute_workflow(self, user_prompt: str):
        """Start workflow execution in separate thread"""
        self.terminal.log("🚀 Starting AvA workflow...")
        self.terminal.log(f"📝 User request: {user_prompt}")

        # Emit workflow started signal
        self.workflow_started.emit(user_prompt)

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
            result = self._execute_micro_tasks(micro_tasks, plan)

            # Step 4: Setup code viewer project
            if result["success"]:
                self.terminal.log("📂 CODE VIEWER: Loading project...")
                self._setup_code_viewer_project(result)

            self.terminal.log("✅ Workflow completed successfully!")

            # Emit workflow completed signal
            self.workflow_completed.emit(result)

        except Exception as e:
            self.terminal.log(f"❌ Workflow failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.workflow_completed.emit(error_result)

    def _create_plan(self, user_prompt: str) -> dict:
        """Create high-level plan using Planner LLM"""

        plan_prompt = f"""
Act as a software architect. Create a detailed plan for: {user_prompt}

Return ONLY a JSON structure:
{{
    "project_name": "snake_case_name",
    "description": "Brief description",
    "architecture": "Brief architecture overview",
    "files_needed": [
        {{"path": "main.py", "purpose": "Main entry point", "priority": 1}},
        {{"path": "utils.py", "purpose": "Utility functions", "priority": 2}},
        {{"path": "config.py", "purpose": "Configuration", "priority": 3}}
    ],
    "dependencies": [
        "requests", "pathlib"
    ]
}}

Keep it simple but functional - maximum 5 files for a working prototype.
Order files by priority (1 = most important).
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

                # Show plan summary in terminal
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

    def _create_micro_tasks(self, plan: dict, user_prompt: str) -> list:
        """Break plan into atomic micro-tasks"""

        micro_tasks = []

        # Sort files by priority
        files_sorted = sorted(plan["files_needed"], key=lambda x: x.get("priority", 999))

        for file_info in files_sorted:
            tasks_prompt = f"""
Break down this file into atomic micro-tasks for code generation:

File: {file_info['path']}
Purpose: {file_info['purpose']}
Project Context: {user_prompt}
Dependencies: {plan.get('dependencies', [])}

Return ONLY a JSON array of micro-tasks:
[
    {{"id": "imports_{file_info['path']}", "type": "imports", "description": "Import necessary modules for {file_info['path']}"}},
    {{"id": "main_function_{file_info['path']}", "type": "function", "description": "Create main function for {file_info['purpose']}"}},
    {{"id": "helper_class_{file_info['path']}", "type": "class", "description": "Create helper class if needed"}}
]

Make each task atomic - one specific code element (function, class, import block).
Include docstrings as separate tasks for major functions/classes.
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
                        task["file_purpose"] = file_info["purpose"]
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
                    "description": f"Implement complete {file_info['path']} for {file_info['purpose']}"
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

        # Group tasks by file
        tasks_by_file = {}
        for task in micro_tasks:
            file_path = task["file_path"]
            if file_path not in tasks_by_file:
                tasks_by_file[file_path] = []
            tasks_by_file[file_path].append(task)

        # Execute tasks file by file
        for file_path, file_tasks in tasks_by_file.items():
            self.terminal.log(f"🔧 Processing file: {file_path}")

            code_snippets[file_path] = ""

            # Execute each micro-task for this file
            for task in file_tasks:
                self.terminal.log(f"  -> Executing: {task['description']}")

                code_prompt = f"""
You are a specialized Python code generator. Generate ONLY the code for this micro-task:

Task: {task['description']}
Task Type: {task['type']}
File: {file_path}
File Purpose: {task.get('file_purpose', 'General implementation')}

Context - Previous code snippets for this file:
{code_snippets[file_path] if code_snippets[file_path] else 'None yet - this is the first task for this file'}

Project Context: {plan['description']}
Dependencies: {plan.get('dependencies', [])}

Requirements:
- Write clean, working Python code
- Include proper imports if this is an import task
- Add comprehensive docstrings for functions/classes
- Follow Python best practices and PEP 8
- Make sure code works with the overall file context

Return ONLY Python code, no explanations or markdown:
"""

                self.terminal.log(f"    -> Calling Code LLM...")
                code_response = self.llm_client.chat(code_prompt)

                # Clean up code response
                code = self._clean_code_response(code_response)

                # Accumulate code for this file
                code_snippets[file_path] += f"\n# {task['description']}\n{code}\n"

                self.terminal.log(f"    -> Generated {len(code)} chars")

            # Assemble complete file
            self.terminal.log(f"📝 Assembling complete file: {file_path}")
            final_code = self._assemble_final_file(file_path, code_snippets[file_path], plan)

            # Write file
            full_file_path = project_dir / file_path
            full_file_path.parent.mkdir(parents=True, exist_ok=True)
            full_file_path.write_text(final_code, encoding='utf-8')
            generated_files.append(str(full_file_path))

            self.terminal.log(f"  -> ✅ Written: {full_file_path}")

            # Emit signal that file was generated
            self.file_generated.emit(str(full_file_path))

            # Show snippet in terminal
            preview = final_code[:200] + "..." if len(final_code) > 200 else final_code
            self.terminal.log(f"\n📄 {file_path} preview:")
            self.terminal.log("─" * 50)
            for line in preview.split('\n')[:10]:  # Show first 10 lines
                self.terminal.log(f"  {line}")
            if len(final_code) > 200:
                self.terminal.log("  ...")
            self.terminal.log("─" * 50)

        # Create README if not exists
        readme_path = project_dir / "README.md"
        if not readme_path.exists():
            readme_content = self._generate_readme(plan, generated_files)
            readme_path.write_text(readme_content, encoding='utf-8')
            generated_files.append(str(readme_path))
            self.file_generated.emit(str(readme_path))

        return {
            "success": True,
            "project_dir": str(project_dir),
            "files": generated_files,
            "project_name": project_name,
            "file_count": len(generated_files)
        }

    def _assemble_final_file(self, file_path: str, code_content: str, plan: dict) -> str:
        """Assemble and polish the final file"""

        assemble_prompt = f"""
Assemble this code into a complete, working Python file:

File: {file_path}
Purpose: {next((f['purpose'] for f in plan['files_needed'] if f['path'] == file_path), 'Implementation file')}
Code snippets:
{code_content}

Requirements:
- Organize imports at the top (standard library, third-party, local)
- Remove duplicates and conflicts
- Ensure proper indentation and formatting
- Add file-level docstring at the top
- Make it runnable and functional
- Add if __name__ == "__main__": block if this is a main file
- Ensure all functions and classes are properly defined

Return ONLY the complete Python file content:
"""

        self.terminal.log(f"  -> Final assembly pass...")
        final_code = self.llm_client.chat(assemble_prompt)

        # Clean final code
        final_code = self._clean_code_response(final_code)

        return final_code

    def _generate_readme(self, plan: dict, generated_files: list) -> str:
        """Generate a README.md for the project"""

        file_list = "\n".join(
            [f"- `{Path(f).name}`: {Path(f).name} implementation" for f in generated_files if f.endswith('.py')])

        readme_content = f"""# {plan['project_name'].replace('_', ' ').title()}

{plan['description']}

## Architecture

{plan.get('architecture', 'Simple Python implementation')}

## Files

{file_list}

## Dependencies

{chr(10).join(f"- {dep}" for dep in plan.get('dependencies', [])) if plan.get('dependencies') else "No external dependencies"}

## Usage

```bash
python main.py
```

## Generated by AvA

This project was generated by AvA (AI Virtual Assistant) on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
"""

        return readme_content

    def _setup_code_viewer_project(self, result: dict):
        """Setup the code viewer with the generated project"""
        if result["success"] and self.code_viewer:
            # Load the project directory in the code viewer
            self.code_viewer.load_project(result["project_dir"])

            # Auto-open the main file if it exists
            main_files = ["main.py", "app.py", "__init__.py"]
            project_path = Path(result["project_dir"])

            for main_file in main_files:
                main_path = project_path / main_file
                if main_path.exists():
                    self.code_viewer.auto_open_file(str(main_path))
                    break

            # Emit project loaded signal
            self.project_loaded.emit(result["project_dir"])

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

    def _on_file_changed(self, file_path: str, content: str):
        """Handle file changes from code viewer"""
        self.terminal.log(f"📝 File modified: {Path(file_path).name}")
        # Could implement auto-save or change tracking here