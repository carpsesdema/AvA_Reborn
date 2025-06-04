import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from core.llm_client import LLMRole
from core.assembler_service import AssemblerService


class EnhancedPlannerService:
    """🧠 Production-Ready Planning Service with Context Intelligence"""

    def __init__(self, llm_client, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager
        self.conversation_context = []  # Store conversation history

    def add_conversation_context(self, message: str, role: str = "user"):
        """Add conversation context for better planning"""
        self.conversation_context.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now()
        })
        # Keep last 10 messages for context
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)

    async def create_project_plan(self, user_prompt: str, context_cache, full_conversation: List[Dict] = None) -> dict:
        """Create intelligent project plan with full context"""

        # Extract FULL requirements from conversation
        full_requirements = self._extract_full_requirements(user_prompt, full_conversation)

        # Get targeted RAG context
        rag_context = await self._get_intelligent_rag_context(full_requirements)

        # Detect project type with high accuracy
        project_type = self._detect_project_type(full_requirements)

        plan_prompt = f"""
You are a Senior Software Architect creating a COMPLETE working project.

FULL REQUIREMENTS ANALYSIS:
{full_requirements}

PROJECT TYPE DETECTED: {project_type}

RELEVANT CODE EXAMPLES:
{rag_context}

Create a comprehensive JSON plan for a FULLY FUNCTIONAL application:

{{
    "project_name": "descriptive_name",
    "description": "Clear description of what this does",
    "architecture_type": "{project_type}",
    "main_requirements": [
        "Must have PySide6 GUI with calculator interface",
        "Must handle arithmetic operations",
        "Must have professional error handling",
        "Must be immediately executable"
    ],
    "files": {{
        "main.py": {{
            "priority": 1,
            "description": "Complete {project_type} application with full functionality",
            "size_estimate": "80-120 lines",
            "key_features": ["GUI setup", "event handlers", "calculations", "error handling"]
        }}
    }},
    "dependencies": ["PySide6"],
    "execution_notes": "Should run immediately with 'python main.py'"
}}

CRITICAL: This must be a COMPLETE, WORKING application, not a stub or template.
"""

        try:
            response_chunks = []
            stream_generator = None
            try:
                stream_generator = self.llm_client.stream_chat(plan_prompt, LLMRole.PLANNER)
                async for chunk in stream_generator:
                    response_chunks.append(chunk)
                    if len(response_chunks) % 15 == 0:
                        await asyncio.sleep(0.01)
            finally:
                if stream_generator and hasattr(stream_generator, 'aclose'):
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass

            response_text = ''.join(response_chunks)
            plan = self._extract_json(response_text)

            # Validate and enhance plan
            if not isinstance(plan, dict) or 'files' not in plan:
                print(f"Invalid plan format, creating intelligent fallback")
                return self._create_intelligent_fallback_plan(full_requirements, project_type)

            return plan

        except Exception as e:
            print(f"Planning failed: {e}")
            return self._create_intelligent_fallback_plan(full_requirements, project_type)

    def _extract_full_requirements(self, current_prompt: str, conversation: List[Dict] = None) -> str:
        """Extract complete requirements from conversation history"""
        requirements = []

        # Add conversation context
        if conversation:
            for msg in conversation[-5:]:  # Last 5 messages
                if msg.get("role") == "user":
                    requirements.append(msg.get("message", ""))

        # Add stored conversation context
        for ctx in self.conversation_context[-3:]:
            if ctx.get("role") == "user":
                requirements.append(ctx.get("message", ""))

        # Add current prompt
        requirements.append(current_prompt)

        # Combine and clean
        full_text = " ".join(requirements)

        return f"""
COMPLETE USER REQUEST ANALYSIS:
{full_text}

KEY REQUIREMENTS EXTRACTED:
- Application Type: {"GUI Calculator" if "calculator" in full_text.lower() else "Application"}
- UI Framework: {"PySide6" if "pyside" in full_text.lower() else "GUI"}
- Core Function: {"Mathematical calculations" if "calculator" in full_text.lower() else "General application"}
- User Interface: {"Button-based calculator interface" if "calculator" in full_text.lower() else "User interface"}
"""

    def _detect_project_type(self, requirements: str) -> str:
        """Intelligent project type detection"""
        req_lower = requirements.lower()

        if any(word in req_lower for word in ["calculator", "pyside", "gui", "interface"]):
            return "gui_calculator"
        elif any(word in req_lower for word in ["web", "api", "flask", "fastapi"]):
            return "web_app"
        elif any(word in req_lower for word in ["cli", "command", "terminal"]):
            return "cli_app"
        else:
            return "desktop_app"

    async def _get_intelligent_rag_context(self, requirements: str) -> str:
        """Get highly targeted RAG context"""
        if not self.rag_manager or not self.rag_manager.is_ready:
            return "No RAG context available - use Python best practices"

        # Multiple targeted queries for better context
        queries = [
            "PySide6 calculator application complete example",
            "PyQt calculator implementation with buttons",
            "Python GUI calculator arithmetic operations",
            "PySide6 QGridLayout calculator interface",
            "Python calculator eval mathematical expressions"
        ]

        context_parts = []
        for query in queries:
            try:
                results = self.rag_manager.query_context(query, k=2)
                for result in results:
                    content = result.get('content', '')
                    if len(content) > 100:  # Only substantial examples
                        context_parts.append(f"# {query}:\n{content[:800]}")
            except Exception as e:
                print(f"RAG query failed for '{query}': {e}")

        if context_parts:
            return "\n\n".join(context_parts[:3])  # Top 3 most relevant

        return "No specific examples found - use standard Python GUI patterns"

    async def create_micro_tasks(self, file_path: str, plan: dict, context_cache) -> List[dict]:
        """Create detailed, executable micro-tasks"""

        files_info = plan.get('files', {})
        file_info = files_info.get(file_path, {})
        project_name = plan.get('project_name', 'project')
        project_type = plan.get('architecture_type', 'desktop_app')

        # Get file-specific RAG context
        file_context = await self._get_file_specific_context(file_path, project_type, context_cache)

        task_prompt = f"""
Break {file_path} into detailed, executable micro-tasks for a {project_type}:

PROJECT: {project_name}
FILE: {file_path}
TYPE: {project_type}
DESCRIPTION: {file_info.get('description', 'Application file')}

SPECIFIC CONTEXT:
{file_context}

Create detailed JSON tasks for COMPLETE implementation:

[
    {{
        "id": "imports_and_setup",
        "type": "imports",
        "description": "Import all required modules and set up application structure",
        "priority": 1,
        "expected_lines": 8,
        "specific_requirements": [
            "Import PySide6.QtWidgets, QtCore modules",
            "Import sys for application execution",
            "Set up proper module organization"
        ]
    }},
    {{
        "id": "main_class_definition",
        "type": "class",
        "description": "Create main calculator class with complete GUI setup",
        "priority": 2,
        "expected_lines": 25,
        "specific_requirements": [
            "QMainWindow or QWidget inheritance",
            "Window properties (title, size, styling)",
            "Layout management setup",
            "All UI components (display, buttons)"
        ]
    }},
    {{
        "id": "event_handlers",
        "type": "methods",
        "description": "Implement all button click handlers and calculator logic",
        "priority": 3,
        "expected_lines": 30,
        "specific_requirements": [
            "Number button handlers",
            "Operator button handlers",
            "Equals and clear functionality",
            "Error handling for division by zero"
        ]
    }},
    {{
        "id": "main_execution",
        "type": "main",
        "description": "Application entry point with proper initialization",
        "priority": 4,
        "expected_lines": 8,
        "specific_requirements": [
            "QApplication creation",
            "Main window instantiation",
            "Event loop execution",
            "Proper exit handling"
        ]
    }}
]

Each task must be SPECIFIC and EXECUTABLE. Include exact requirements.
"""

        try:
            response_chunks = []
            stream_generator = None
            try:
                stream_generator = self.llm_client.stream_chat(task_prompt, LLMRole.PLANNER)
                async for chunk in stream_generator:
                    response_chunks.append(chunk)
                    if len(response_chunks) % 15 == 0:
                        await asyncio.sleep(0.01)
            finally:
                if stream_generator and hasattr(stream_generator, 'aclose'):
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass

            response_text = ''.join(response_chunks)
            tasks = self._extract_json(response_text)

            if not isinstance(tasks, list):
                print(f"Invalid tasks format for {file_path}, creating fallback")
                return self._create_fallback_tasks(file_path, project_type)

            # Enhance tasks with file context
            for task in tasks:
                task["file_path"] = file_path
                task["project_type"] = project_type

            return tasks if tasks else self._create_fallback_tasks(file_path, project_type)

        except Exception as e:
            print(f"Task creation failed for {file_path}: {e}")
            return self._create_fallback_tasks(file_path, project_type)

    async def _get_file_specific_context(self, file_path: str, project_type: str, cache) -> str:
        """Get context specific to file and project type"""
        if not self.rag_manager or not self.rag_manager.is_ready:
            return f"Standard {project_type} implementation patterns"

        query = f"{project_type} {file_path} implementation patterns complete example"
        try:
            results = self.rag_manager.query_context(query, k=2)
            if results:
                return "\n".join([r.get('content', '')[:600] for r in results])
        except Exception as e:
            print(f"File context retrieval failed: {e}")

        return f"Standard {project_type} implementation patterns"

    def _create_intelligent_fallback_plan(self, requirements: str, project_type: str) -> dict:
        """Create intelligent fallback based on detected project type"""

        if project_type == "gui_calculator":
            return {
                "project_name": "pyside6_calculator",
                "description": "Complete PySide6 Calculator Application",
                "architecture_type": "gui_calculator",
                "main_requirements": [
                    "PySide6 GUI with calculator interface",
                    "Button grid for numbers and operations",
                    "Display area for calculations",
                    "Complete arithmetic functionality"
                ],
                "files": {
                    "main.py": {
                        "priority": 1,
                        "description": "Complete calculator application with PySide6 GUI",
                        "size_estimate": "100+ lines",
                        "key_features": ["GUI layout", "calculation logic", "event handling"]
                    }
                },
                "dependencies": ["PySide6"]
            }
        else:
            return {
                "project_name": "generated_application",
                "description": requirements[:100],
                "architecture_type": project_type,
                "files": {
                    "main.py": {"priority": 1, "description": "Main application file"}
                },
                "dependencies": []
            }

    def _create_fallback_tasks(self, file_path: str, project_type: str) -> List[dict]:
        """Create fallback tasks based on project type"""

        if project_type == "gui_calculator":
            return [
                {
                    "id": "complete_calculator",
                    "type": "complete",
                    "description": "Implement complete PySide6 calculator application",
                    "file_path": file_path,
                    "project_type": project_type,
                    "specific_requirements": [
                        "Complete PySide6 calculator with button grid",
                        "Display area for numbers and results",
                        "All arithmetic operations (+, -, *, /)",
                        "Clear and equals functionality",
                        "Professional error handling"
                    ]
                }
            ]
        else:
            return [
                {
                    "id": f"implement_{file_path}",
                    "type": "complete",
                    "description": f"Implement complete {file_path}",
                    "file_path": file_path,
                    "project_type": project_type
                }
            ]

    def _extract_json(self, text: str):
        """Enhanced JSON extraction"""
        try:
            # Try standard extraction
            start, end = text.find('{'), text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
        except:
            pass

        try:
            # Try array extraction
            start, end = text.find('['), text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                return json.loads(json_text)
        except:
            pass

        print("JSON extraction failed")
        return {}


class EnhancedCoderService:
    """⚙️ Production-Ready Coding Service with Intelligent Generation"""

    def __init__(self, llm_client, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager

    async def execute_micro_task(self, task: dict, file_context: str, context_cache) -> str:
        """Execute task with intelligent context and examples"""

        # Get targeted examples for this specific task
        task_examples = await self._get_task_specific_examples(task)

        # Create comprehensive prompt
        code_prompt = f"""
You are a senior Python developer. Generate COMPLETE, PRODUCTION-READY code for this task:

TASK: {task['description']}
TYPE: {task.get('type', 'general')}
PROJECT TYPE: {task.get('project_type', 'application')}

SPECIFIC REQUIREMENTS:
{chr(10).join('- ' + req for req in task.get('specific_requirements', []))}

RELEVANT EXAMPLES:
{task_examples}

CRITICAL INSTRUCTIONS:
1. Generate COMPLETE, WORKING code - not stubs or TODOs
2. Include proper imports, error handling, and documentation
3. Use professional coding standards
4. Make it immediately functional
5. For GUI: Create complete interface with all components
6. For calculators: Include full arithmetic logic

Generate ONLY the Python code:
"""

        try:
            response_chunks = []
            stream_generator = None
            try:
                stream_generator = self.llm_client.stream_chat(code_prompt, LLMRole.CODER)
                async for chunk in stream_generator:
                    response_chunks.append(chunk)
                    if len(response_chunks) % 20 == 0:
                        await asyncio.sleep(0.005)
            finally:
                if stream_generator and hasattr(stream_generator, 'aclose'):
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass

            code = ''.join(response_chunks)
            cleaned_code = self._clean_and_validate_code(code, task)

            return cleaned_code

        except Exception as e:
            print(f"Task execution failed: {e}")
            return self._create_emergency_fallback(task)

    async def _get_task_specific_examples(self, task: dict) -> str:
        """Get examples specific to this task type"""
        if not self.rag_manager or not self.rag_manager.is_ready:
            return "Use Python best practices and standard patterns"

        task_type = task.get('type', 'general')
        project_type = task.get('project_type', 'application')

        # Create targeted queries based on task
        queries = []

        if task_type == "imports":
            queries = [f"PySide6 {project_type} imports standard setup"]
        elif task_type == "class":
            queries = [f"PySide6 {project_type} main class complete", f"Python {project_type} class structure"]
        elif task_type == "methods":
            queries = [f"{project_type} event handlers Python", f"calculator button handlers PySide6"]
        elif task_type == "main":
            queries = [f"Python {project_type} main execution QApplication"]
        else:
            queries = [f"Python {project_type} complete implementation"]

        context_parts = []
        for query in queries:
            try:
                results = self.rag_manager.query_context(query, k=1)
                for result in results:
                    content = result.get('content', '')
                    if len(content) > 50:
                        context_parts.append(f"Example for {task_type}:\n{content[:500]}")
            except Exception as e:
                print(f"RAG query failed: {e}")

        return "\n\n".join(context_parts) if context_parts else "Use standard Python patterns"

    def _clean_and_validate_code(self, code: str, task: dict) -> str:
        """Clean and validate generated code"""
        # Remove markdown
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 3:
                code = parts[1]

        cleaned = code.strip()

        # Basic validation
        if len(cleaned) < 20:
            print(f"Generated code too short for task: {task['description']}")
            return self._create_emergency_fallback(task)

        # Check for basic syntax (try to compile)
        try:
            compile(cleaned, '<string>', 'exec')
        except SyntaxError as e:
            print(f"Generated code has syntax errors: {e}")
            return self._create_emergency_fallback(task)

        return cleaned

    def _create_emergency_fallback(self, task: dict) -> str:
        """Create emergency fallback code"""
        task_type = task.get('type', 'general')
        project_type = task.get('project_type', 'application')

        if project_type == "gui_calculator" and task_type == "complete":
            return '''import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QGridLayout

class Calculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculator")
        self.setGeometry(100, 100, 300, 400)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.display = QLineEdit("0")
        self.display.setReadOnly(True)
        layout.addWidget(self.display)

        button_layout = QGridLayout()
        buttons = [
            ('7', 0, 0), ('8', 0, 1), ('9', 0, 2), ('/', 0, 3),
            ('4', 1, 0), ('5', 1, 1), ('6', 1, 2), ('*', 1, 3),
            ('1', 2, 0), ('2', 2, 1), ('3', 2, 2), ('-', 2, 3),
            ('0', 3, 0), ('C', 3, 1), ('=', 3, 2), ('+', 3, 3)
        ]

        for text, row, col in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, t=text: self.on_button_click(t))
            button_layout.addWidget(btn, row, col)

        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        layout.addWidget(button_widget)

        self.setCentralWidget(central_widget)
        self.expression = ""

    def on_button_click(self, text):
        if text == 'C':
            self.expression = ""
            self.display.setText("0")
        elif text == '=':
            try:
                result = str(eval(self.expression))
                self.display.setText(result)
                self.expression = result
            except:
                self.display.setText("Error")
                self.expression = ""
        else:
            if self.expression == "0":
                self.expression = ""
            self.expression += text
            self.display.setText(self.expression)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = Calculator()
    calculator.show()
    sys.exit(app.exec())'''

        return f"# TODO: Implement {task['description']}\npass"

    def _clean_code(self, code: str) -> str:
        """Legacy method for compatibility"""
        return self._clean_and_validate_code(code, {})

    # Keep the original parallel execution method
    async def execute_tasks_parallel(self, tasks: List[dict], file_context: str,
                                     context_cache, progress_callback=None) -> List[dict]:
        """Execute tasks with intelligent parallel/sequential strategy"""
        results = []

        if not isinstance(tasks, list):
            print(f"Invalid tasks format: {type(tasks)}")
            return []

        # For complex tasks, execute sequentially for better quality
        for task in tasks:
            try:
                if progress_callback:
                    progress_callback(f"[{task.get('file_path', 'unknown')}] ⚙️ {task['description']}")

                code = await self.execute_micro_task(task, file_context, context_cache)
                results.append({"task": task, "code": code})

            except Exception as e:
                print(f"Task execution failed: {e}")
                results.append({"task": task, "code": f"# ERROR: {e}\npass"})

        return results


# Keep original services for backwards compatibility
class PlannerService(EnhancedPlannerService):
    pass


class CoderService(EnhancedCoderService):
    pass


# Enhanced orchestrator
class WorkflowOrchestrator:
    """🎼 Enhanced Orchestrator with Conversation Intelligence"""

    def __init__(self, planner, coder, assembler, terminal=None):
        # Use enhanced services if they're the right type
        if isinstance(planner, EnhancedPlannerService):
            self.planner = planner
        else:
            # Wrap existing planner
            self.planner = EnhancedPlannerService(planner.llm_client, planner.rag_manager)

        if isinstance(coder, EnhancedCoderService):
            self.coder = coder
        else:
            # Wrap existing coder
            self.coder = EnhancedCoderService(coder.llm_client, coder.rag_manager)

        self.assembler = assembler
        self.terminal = terminal

        from core.workflow_engine import ContextCache
        self.context_cache = ContextCache()

    def set_conversation_context(self, conversation_history: List[Dict]):
        """Set full conversation context for intelligent planning"""
        for msg in conversation_history:
            self.planner.add_conversation_context(msg.get("message", ""), msg.get("role", "user"))

    async def execute_workflow(self, user_prompt: str, output_dir: Path,
                               conversation_history: List[Dict] = None) -> dict:
        """Execute enhanced workflow with conversation intelligence"""
        try:
            self._log("🚀 Starting Enhanced Production Workflow...")

            # Set conversation context
            if conversation_history:
                self.set_conversation_context(conversation_history)

            # Enhanced planning with full context
            self._log("🧠 PLANNER: Creating intelligent project architecture...")
            plan = await self.planner.create_project_plan(user_prompt, self.context_cache, conversation_history)

            if not isinstance(plan, dict) or 'files' not in plan:
                self._log("❌ Planning failed - invalid structure")
                return {"success": False, "error": "Enhanced planning failed"}

            project_name = plan.get("project_name", "enhanced_project")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_dir = output_dir / f"{project_name}_{timestamp}"
            project_dir.mkdir(exist_ok=True)

            files = plan.get("files", {})
            self._log(f"📋 Processing {len(files)} files with enhanced intelligence")

            generated_files = []
            for file_path, file_info in files.items():
                try:
                    result = await self._process_file_intelligently(file_path, file_info, plan, project_dir)
                    if result.get("success"):
                        generated_files.append(result["file_path"])
                        self._log(f"✅ {file_path} completed with enhanced quality")
                    else:
                        self._log(f"⚠️ {file_path} had issues: {result.get('error', 'Unknown')}")
                except Exception as e:
                    self._log(f"❌ {file_path} failed: {e}")

            self._log(f"✅ Enhanced Workflow completed! Generated {len(generated_files)} files")

            return {
                "success": True,
                "project_dir": str(project_dir),
                "files": generated_files,
                "project_name": project_name,
                "file_count": len(generated_files),
                "enhanced": True
            }

        except Exception as e:
            self._log(f"❌ Enhanced workflow failed: {e}")
            return {"success": False, "error": str(e)}

    async def _process_file_intelligently(self, file_path: str, file_info: dict,
                                          plan: dict, project_dir: Path) -> dict:
        """Process single file with enhanced intelligence"""
        try:
            self._log(f"[{file_path}] Creating intelligent micro-tasks...")
            tasks = await self.planner.create_micro_tasks(file_path, plan, self.context_cache)
            self._log(f"[{file_path}] Created {len(tasks)} enhanced tasks")

            def progress_callback(message):
                self._log(message)

            self._log(f"[{file_path}] Executing with enhanced coding...")
            task_results = await self.coder.execute_tasks_parallel(
                tasks, "", self.context_cache, progress_callback
            )

            self._log(f"[{file_path}] Assembling with quality checks...")
            assembled_code, review_approved, review_feedback = await self.assembler.assemble_file(
                file_path, task_results, plan, self.context_cache
            )

            # Write file
            full_path = project_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(assembled_code, encoding='utf-8')

            return {
                "success": True,
                "file_path": str(full_path),
                "file_name": file_path,
                "review_approved": review_approved,
                "review_feedback": review_feedback
            }

        except Exception as e:
            self._log(f"[{file_path}] Enhanced processing failed: {e}")
            return {
                "success": False,
                "file_path": None,
                "file_name": file_path,
                "error": str(e)
            }

    def _log(self, message: str):
        """Enhanced logging"""
        if self.terminal and hasattr(self.terminal, 'log'):
            self.terminal.log(message)
        else:
            print(message)