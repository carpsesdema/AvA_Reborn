# core/workflow_engine.py - Enhanced with RAG Integration

import json
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import QObject, QThread, Signal


class WorkflowThread(QThread):
    def __init__(self, workflow_engine, user_prompt):
        super().__init__()
        self.workflow_engine = workflow_engine
        self.user_prompt = user_prompt

    def run(self):
        self.workflow_engine._execute_workflow_internal(self.user_prompt)


class WorkflowEngine(QObject):
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    file_generated = Signal(str)
    project_loaded = Signal(str)

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.terminal = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager  # NEW: RAG integration
        self.output_dir = Path("./generated_projects")
        self.output_dir.mkdir(exist_ok=True)

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

        # NEW: Check RAG status
        if self.rag_manager and self.rag_manager.is_ready:
            self.terminal.log("🧠 RAG system ready - will enhance code generation with knowledge base")
        else:
            self.terminal.log("⚠️  RAG system not ready - proceeding without context enhancement")

        self.workflow_started.emit(user_prompt)

        self.workflow_thread = WorkflowThread(self, user_prompt)
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

    def _execute_workflow_internal(self, user_prompt: str):
        try:
            self.terminal.log("🧠 PLANNER: Creating high-level plan...")
            plan = self._create_plan(user_prompt)

            self.terminal.log("📋 PLANNER: Breaking down into micro-tasks...")
            micro_tasks = self._create_micro_tasks(plan, user_prompt)

            self.terminal.log("⚙️ ASSEMBLER: Executing micro-tasks...")
            result = self._execute_micro_tasks(micro_tasks, plan)

            if result["success"]:
                self.terminal.log("📂 CODE VIEWER: Loading project...")
                self._setup_code_viewer_project(result)

            self.terminal.log("✅ Workflow completed successfully!")
            self.workflow_completed.emit(result)

        except Exception as e:
            self.terminal.log(f"❌ Workflow failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.workflow_completed.emit(error_result)

    def _create_plan(self, user_prompt: str) -> dict:
        # NEW: Get RAG context for planning
        rag_context = ""
        if self.rag_manager and self.rag_manager.is_ready:
            self.terminal.log("  -> Querying RAG for planning context...")
            rag_context = self.rag_manager.get_context_for_code_generation(
                f"project planning architecture {user_prompt}", "python"
            )
            if rag_context:
                self.terminal.log(f"  -> RAG context retrieved ({len(rag_context)} chars)")
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

        self.terminal.log("  -> Calling Planner LLM...")
        response = self.llm_client.chat(plan_prompt)
        self.terminal.log(f"  -> Planner response received ({len(response)} chars)")

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

    def _create_micro_tasks(self, plan: dict, user_prompt: str) -> list:
        micro_tasks = []
        files_sorted = sorted(plan["files_needed"], key=lambda x: x.get("priority", 999))

        for file_info in files_sorted:
            # NEW: Get file-specific RAG context
            rag_context = ""
            if self.rag_manager and self.rag_manager.is_ready:
                file_query = f"{file_info['purpose']} {file_info['path']} python implementation"
                rag_context = self.rag_manager.get_context_for_code_generation(file_query, "python")

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

            self.terminal.log(f"  -> Creating micro-tasks for {file_info['path']}")
            if rag_context:
                self.terminal.log(f"    -> Using RAG context ({len(rag_context)} chars)")

            response = self.llm_client.chat(tasks_prompt)

            try:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    file_tasks = json.loads(response[start:end])
                    for task in file_tasks:
                        task["file_path"] = file_info["path"]
                        task["file_purpose"] = file_info["purpose"]
                        task["rag_context"] = rag_context  # Store for later use
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

    def _execute_micro_tasks(self, micro_tasks: list, plan: dict) -> dict:
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

                # NEW: Include RAG context in code generation
                rag_context = task.get("rag_context", "")

                code_prompt = f"""
Generate Python code for this micro-task:

Task: {task['description']}
File: {file_path}
Context: {code_snippets[file_path] if code_snippets[file_path] else 'First task for this file'}
Project: {plan['description']}

{f"Reference Examples and Best Practices:\n{rag_context}\n" if rag_context else ""}

Return ONLY Python code, no explanations:
"""

                self.terminal.log(f"    -> Calling Code LLM...")
                if rag_context:
                    self.terminal.log(f"    -> Enhanced with RAG context")

                code_response = self.llm_client.chat(code_prompt)
                code = self._clean_code_response(code_response)
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
            self.file_generated.emit(str(full_file_path))

        return {
            "success": True,
            "project_dir": str(project_dir),
            "files": generated_files,
            "project_name": project_name,
            "file_count": len(generated_files)
        }

    def _assemble_final_file(self, file_path: str, code_content: str, plan: dict) -> str:
        # NEW: Get RAG context for assembly
        rag_context = ""
        if self.rag_manager and self.rag_manager.is_ready:
            assembly_query = f"python file structure organization {file_path} best practices"
            rag_context = self.rag_manager.get_context_for_code_generation(assembly_query, "python")

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

        self.terminal.log(f"  -> Final assembly pass...")
        if rag_context:
            self.terminal.log(f"  -> Using RAG best practices")

        final_code = self.llm_client.chat(assemble_prompt)
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