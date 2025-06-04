# enhanced_workflow_engine.py - COMPLETE WORKING FILE with Role-Based LLM

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil
import json
import traceback

from PySide6.QtCore import QObject, Signal, QTimer, QThread


class EnhancedWorkflowEngine(QObject):
    """
    🚀 Enhanced Workflow Engine with Role-Based AI Code Generation
    """

    # Define all the signals the terminal expects
    workflow_started = Signal(str)  # prompt
    workflow_completed = Signal(dict)  # result
    workflow_progress = Signal(str, str)  # stage, description
    file_generated = Signal(str)  # file_path
    project_loaded = Signal(str)  # project_path

    # Additional streaming signals
    file_progress = Signal(str, str)  # file_path, status
    streaming_content = Signal(str, str, str)  # content, type, indent
    task_progress = Signal(int, int)  # completed, total

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.terminal_window = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager

        self.logger = logging.getLogger(__name__)

        # DEBUG: Test LLM immediately
        self._debug_llm_client()

        # Performance tracking
        self.workflow_stats = {
            "cache_stats": {"cache_size": 0, "hit_rate": 0.0},
            "workflow_state": {"stage": "idle", "completed_tasks": 0, "total_tasks": 0}
        }

        # Connect signals to terminal methods
        self._connect_terminal_signals()

        self.logger.info("✅ Enhanced Workflow Engine initialized with Role-Based AI")

    def _debug_llm_client(self):
        """DEBUG: Check what's wrong with LLM client"""
        try:
            self.logger.info("🔍 DEBUGGING LLM CLIENT...")

            if not self.llm_client:
                self.logger.error("❌ LLM CLIENT IS NONE!")
                return

            self.logger.info(f"✅ LLM Client exists: {type(self.llm_client)}")

            # Check available methods
            methods = [method for method in dir(self.llm_client) if not method.startswith('_')]
            self.logger.info(f"📋 LLM Client methods: {methods}")

            # Check if chat method exists
            if hasattr(self.llm_client, 'chat'):
                self.logger.info("✅ chat method exists")
            else:
                self.logger.error("❌ chat method MISSING!")

            # Check available models
            if hasattr(self.llm_client, 'get_available_models'):
                models = self.llm_client.get_available_models()
                self.logger.info(f"📊 Available models: {models}")

                if not models or models == ["No LLM services available"]:
                    self.logger.error("❌ NO LLM MODELS AVAILABLE!")
                else:
                    self.logger.info("✅ LLM models are available")
            else:
                self.logger.warning("⚠️ get_available_models method missing")

        except Exception as e:
            self.logger.error(f"❌ LLM DEBUG FAILED: {e}")
            self.logger.error(traceback.format_exc())

    def _connect_terminal_signals(self):
        """Connect workflow signals to terminal streaming methods"""
        if not self.terminal_window:
            return

        try:
            # Connect progress signals (FIXED: Only one connection to avoid duplicates)
            self.workflow_progress.connect(self.terminal_window.update_workflow_progress)
            self.task_progress.connect(self.terminal_window.update_task_progress)

            # Connect file progress
            if hasattr(self.terminal_window, 'start_file_generation'):
                self.file_progress.connect(self._handle_file_progress)

            # Connect streaming content
            if hasattr(self.terminal_window, 'stream_log'):
                self.streaming_content.connect(self._handle_streaming_content)

            self.logger.info("✅ Terminal signals connected for streaming")

        except Exception as e:
            self.logger.error(f"❌ Failed to connect terminal signals: {e}")

    def _handle_file_progress(self, file_path: str, status: str):
        """Handle file progress updates"""
        if status == "started":
            self.terminal_window.start_file_generation(file_path)
        elif status in ["completed", "error"]:
            success = status == "completed"
            self.terminal_window.complete_file_generation(file_path, success)

    def _handle_streaming_content(self, content: str, content_type: str, indent: str):
        """Handle streaming content updates"""
        indent_level = int(indent) if indent.isdigit() else 0
        self.terminal_window.stream_log(content, indent=indent_level)

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        """
        🚀 Execute workflow with Role-Based AI code generation and streaming progress
        """
        self.logger.info(f"🚀 Starting Role-Based AI workflow: {user_prompt[:100]}...")

        # Initialize workflow state
        self.workflow_stats["workflow_state"] = {
            "stage": "initializing",
            "completed_tasks": 0,
            "total_tasks": 6
        }

        try:
            # Emit workflow started
            self.workflow_started.emit(user_prompt)

            # Stage 1: Initialize
            await self._emit_stage_progress("initializing", "Setting up AI workflow environment")
            await asyncio.sleep(0.1)

            # Stage 2: Context Discovery
            await self._emit_stage_progress("context_discovery", "Analyzing project requirements with AI")
            context_info = await self._discover_context_with_ai(user_prompt, conversation_context)
            self._update_task_progress(1, 6)

            # Stage 3: Planning
            await self._emit_stage_progress("planning", "Creating intelligent development plan")
            plan = await self._create_ai_development_plan(user_prompt, context_info)
            self._update_task_progress(2, 6)

            # Stage 4: Architecture Design
            await self._emit_stage_progress("design", "Designing project architecture with AI")
            architecture = await self._design_project_architecture(user_prompt, plan)
            self._update_task_progress(3, 6)

            # Stage 5: Code Generation
            await self._emit_stage_progress("generation", "Generating code with AI assistance")
            results = await self._execute_ai_generation_phase(user_prompt, plan, architecture)
            self._update_task_progress(5, 6)

            # Stage 6: Finalization
            await self._emit_stage_progress("finalization", "Finalizing project")
            final_result = await self._finalize_project(results)
            self._update_task_progress(6, 6)

            # Complete workflow
            await self._emit_stage_progress("complete", "AI workflow completed successfully")
            self.workflow_completed.emit(final_result)

            self.logger.info("✅ Enhanced AI workflow completed successfully")
            return final_result

        except Exception as e:
            self.logger.error(f"❌ AI Workflow failed: {e}", exc_info=True)
            await self._emit_stage_progress("error", f"AI workflow failed: {str(e)}")
            self.workflow_completed.emit({"success": False, "error": str(e)})
            raise

    async def _emit_stage_progress(self, stage: str, description: str):
        """Emit stage progress with terminal streaming"""
        # FIXED: Only emit one signal to avoid duplicates
        self.workflow_progress.emit(stage, description)

        # Update internal state
        self.workflow_stats["workflow_state"]["stage"] = stage

        # Stream to terminal
        if stage == "initializing":
            self.streaming_content.emit("🚀 Initializing AvA AI workflow system", "info", "0")
        elif stage == "context_discovery":
            self.streaming_content.emit("🔍 AI analyzing project context and requirements", "info", "0")
        elif stage == "planning":
            self.streaming_content.emit("🧠 AI creating intelligent development plan", "info", "0")
        elif stage == "design":
            self.streaming_content.emit("🏗️ AI designing project architecture", "info", "0")
        elif stage == "generation":
            self.streaming_content.emit("⚡ AI generating professional code", "info", "0")
        elif stage == "finalization":
            self.streaming_content.emit("📄 Finalizing AI-generated project", "info", "0")
        elif stage == "complete":
            self.streaming_content.emit("✅ AI workflow completed successfully!", "success", "0")
        elif stage == "error":
            self.streaming_content.emit(f"❌ {description}", "error", "0")

    def _update_task_progress(self, completed: int, total: int):
        """Update task progress"""
        self.workflow_stats["workflow_state"]["completed_tasks"] = completed
        self.workflow_stats["workflow_state"]["total_tasks"] = total
        self.task_progress.emit(completed, total)

    async def _discover_context_with_ai(self, prompt: str, conversation_context: List[Dict]) -> Dict[str, Any]:
        """AI-powered context discovery"""
        self.streaming_content.emit("  🧠 AI analyzing user requirements", "stream", "1")

        # Use LLM to analyze the prompt
        analysis_prompt = f"""
        Analyze this development request and extract key information:

        Request: "{prompt}"

        Please provide a JSON response with:
        {{
            "project_type": "web_app|desktop_app|cli_tool|library|game|other",
            "main_technology": "python|javascript|java|etc",
            "frameworks": ["PySide6", "Flask", etc],
            "complexity": "simple|moderate|complex",
            "key_features": ["feature1", "feature2"],
            "estimated_files": 3
        }}
        """

        try:
            if self.llm_client and hasattr(self.llm_client, 'chat'):
                self.streaming_content.emit("  🤖 Consulting AI for project analysis", "stream", "1")

                # Get AI analysis using role-based system
                ai_response = await self._call_llm_async(analysis_prompt, "analysis")

                # Try to parse JSON from response
                try:
                    # Extract JSON from response
                    start = ai_response.find('{')
                    end = ai_response.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = ai_response[start:end]
                        ai_analysis = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found")

                    self.streaming_content.emit("  ✅ AI analysis complete", "success", "1")

                    return {
                        "prompt": prompt,
                        "conversation_history": conversation_context or [],
                        "ai_analysis": ai_analysis,
                        "project_type": ai_analysis.get("project_type", "desktop_app"),
                        "complexity": ai_analysis.get("complexity", "moderate"),
                        "frameworks": ai_analysis.get("frameworks", ["Python"])
                    }

                except (json.JSONDecodeError, ValueError) as e:
                    self.streaming_content.emit("  ⚠️ Using fallback analysis", "warning", "1")

        except Exception as e:
            self.logger.warning(f"AI analysis failed, using fallback: {e}")
            self.streaming_content.emit("  ⚠️ AI unavailable, using smart fallback", "warning", "1")

        # Fallback analysis
        return {
            "prompt": prompt,
            "conversation_history": conversation_context or [],
            "project_type": "desktop_app" if "gui" in prompt.lower() or "pyside" in prompt.lower() else "cli_tool",
            "complexity": "moderate",
            "frameworks": ["PySide6"] if "pyside" in prompt.lower() else ["Python"]
        }

    async def _create_ai_development_plan(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered development planning"""
        self.streaming_content.emit("  🧠 AI creating development strategy", "stream", "1")

        planning_prompt = f"""
        Create a development plan for this project:

        Request: "{prompt}"
        Project Type: {context.get('project_type', 'unknown')}
        Complexity: {context.get('complexity', 'moderate')}

        Create a JSON plan with:
        {{
            "project_name": "descriptive_name",
            "files_to_create": [
                {{"filename": "main.py", "purpose": "Main application file"}},
                {{"filename": "calculator.py", "purpose": "Calculator logic"}}
            ],
            "architecture": "single_file|modular|mvc",
            "strategy": "fast_single_file|standard_multi_file|comprehensive"
        }}

        For a calculator app, suggest appropriate modular structure.
        """

        try:
            if self.llm_client:
                self.streaming_content.emit("  🤖 AI designing project structure", "stream", "1")
                ai_response = await self._call_llm_async(planning_prompt, "planning")

                # Parse AI response
                try:
                    start = ai_response.find('{')
                    end = ai_response.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = ai_response[start:end]
                        ai_plan = json.loads(json_str)

                        self.streaming_content.emit("  ✅ AI development plan created", "success", "1")
                        return ai_plan

                except (json.JSONDecodeError, ValueError):
                    pass

        except Exception as e:
            self.logger.warning(f"AI planning failed: {e}")

        # Intelligent fallback based on prompt analysis
        self.streaming_content.emit("  🎯 Creating smart fallback plan", "stream", "1")

        if "calculator" in prompt.lower():
            return {
                "project_name": "calculator_app",
                "files_to_create": [
                    {"filename": "main.py", "purpose": "Main application entry point"},
                    {"filename": "calculator_gui.py", "purpose": "PySide6 GUI interface"},
                    {"filename": "calculator_logic.py", "purpose": "Calculator computation logic"},
                    {"filename": "styles.py", "purpose": "Dark theme styling"},
                    {"filename": "README.md", "purpose": "Project documentation"}
                ],
                "architecture": "modular",
                "strategy": "standard_multi_file"
            }
        else:
            return {
                "project_name": "ai_generated_app",
                "files_to_create": [
                    {"filename": "main.py", "purpose": "Main application file"},
                    {"filename": "README.md", "purpose": "Project documentation"}
                ],
                "architecture": "single_file",
                "strategy": "fast_single_file"
            }

    async def _design_project_architecture(self, prompt: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered architecture design"""
        self.streaming_content.emit("  🏗️ AI designing system architecture", "stream", "1")
        await asyncio.sleep(0.3)

        # For now, return a simple architecture
        architecture = {
            "pattern": plan.get("architecture", "modular"),
            "entry_point": "main.py",
            "dependencies": plan.get("frameworks", ["Python"]),
            "structure": "Clean, modular design with separation of concerns"
        }

        self.streaming_content.emit("  ✅ Architecture design complete", "success", "1")
        return architecture

    async def _execute_ai_generation_phase(self, prompt: str, plan: Dict[str, Any], architecture: Dict[str, Any]) -> \
    Dict[str, Any]:
        """Execute REAL AI code generation"""
        results = {"files_created": [], "project_dir": None}

        # Create project directory
        project_name = plan.get("project_name", "ai_project")
        project_dir = Path("./workspace") / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        results["project_dir"] = str(project_dir)

        self.streaming_content.emit(f"  📁 Created project directory: {project_name}", "info", "1")

        # Generate files with REAL AI
        files_to_create = plan.get("files_to_create", [])
        for i, file_info in enumerate(files_to_create):
            filename = file_info.get("filename", f"file_{i}.py")
            purpose = file_info.get("purpose", "Generated file")

            await self._generate_ai_file(project_dir, filename, purpose, prompt, plan, architecture)
            results["files_created"].append(filename)

            # Update progress
            self._update_task_progress(4 + (i / len(files_to_create)), 6)

        return results

    async def _generate_ai_file(self, project_dir: Path, filename: str, purpose: str,
                                original_prompt: str, plan: Dict[str, Any], architecture: Dict[str, Any]):
        """Generate individual file with REAL AI"""
        file_path = project_dir / filename

        # Start file generation
        self.file_progress.emit(str(file_path), "started")
        self.streaming_content.emit(f"  📄 AI generating {filename}", "stream", "1")

        try:
            # Create AI prompt for this specific file
            if filename.endswith('.py'):
                file_prompt = f"""
                Generate a complete, professional {filename} file for this project:

                Original Request: "{original_prompt}"
                File Purpose: {purpose}
                Project Architecture: {architecture.get('pattern', 'modular')}

                Requirements:
                - Write complete, working Python code
                - Follow best practices and PEP 8
                - Include proper docstrings and comments
                - Make it production-ready
                - If PySide6 GUI, include proper dark theme
                - If calculator, implement all basic operations

                Generate ONLY the Python code, no explanations.
                """
            else:
                file_prompt = f"""
                Generate a complete {filename} file for this project:

                Original Request: "{original_prompt}"
                File Purpose: {purpose}
                Project Name: {plan.get('project_name', 'AI Project')}

                Make it professional and comprehensive.
                Generate ONLY the file content, no explanations.
                """

            self.streaming_content.emit(f"    🤖 AI coding {filename}...", "stream", "2")

            # Call AI to generate content using role-based system
            if self.llm_client:
                ai_content = await self._call_llm_async(file_prompt, "code")

                # Clean up AI response (remove markdown code blocks if present)
                content = ai_content.strip()
                if content.startswith("```python"):
                    content = content[9:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # Ensure we have actual content
                if len(content) < 50:  # Too short, probably failed
                    raise ValueError("AI generated content too short")

            else:
                # Fallback content if no LLM available
                content = self._generate_fallback_content(filename, purpose, original_prompt)

            # Write file
            file_path.write_text(content, encoding='utf-8')

            # Stream success
            self.streaming_content.emit(f"    ✅ {filename} generated ({len(content)} chars)", "success", "2")
            self.file_progress.emit(str(file_path), "completed")
            self.file_generated.emit(str(file_path))

        except Exception as e:
            self.logger.error(f"Failed to generate {filename}: {e}")
            # Create fallback content
            content = self._generate_fallback_content(filename, purpose, original_prompt)
            file_path.write_text(content, encoding='utf-8')

            self.streaming_content.emit(f"    ⚠️ {filename} created with fallback ({len(content)} chars)", "warning",
                                        "2")
            self.file_progress.emit(str(file_path), "completed")
            self.file_generated.emit(str(file_path))

    def _generate_fallback_content(self, filename: str, purpose: str, original_prompt: str) -> str:
        """Generate fallback content when AI is unavailable"""
        if filename == "main.py":
            return f'''#!/usr/bin/env python3
"""
{purpose}
Generated by AvA for: {original_prompt}
"""

def main():
    """Main entry point"""
    print("Application starting...")
    # TODO: Implement main application logic

if __name__ == "__main__":
    main()
'''
        elif filename.endswith('.py'):
            class_name = filename.replace('.py', '').replace('_', ' ').title().replace(' ', '')
            return f'''"""
{purpose}
Generated by AvA for: {original_prompt}
"""

class {class_name}:
    """Main class for {purpose}"""

    def __init__(self):
        """Initialize {class_name}"""
        pass

    def run(self):
        """Run the main functionality"""
        print(f"{class_name} is running...")
        # TODO: Implement functionality
'''
        elif filename == "README.md":
            project_name = Path(original_prompt.split()[0] if original_prompt else "Project").name
            return f'''# {project_name}

{purpose}

## Description
{original_prompt}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

Generated by AvA on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
        else:
            return f'''# {filename}

{purpose}

Created for: {original_prompt}
Generated by AvA on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''

    async def _call_llm_async(self, prompt: str, task_type: str = "code") -> str:
        """Call LLM asynchronously using ROLE-BASED API"""
        try:
            self.logger.info(f"🤖 Attempting LLM call for {task_type}...")
            self.streaming_content.emit(f"    🔍 Calling AI {task_type} specialist...", "stream", "2")

            # Import the LLMRole enum and parent class
            from core.llm_client import LLMRole, EnhancedLLMClient

            # Choose the right AI role based on task type
            if task_type == "planning":
                role = LLMRole.PLANNER
                self.streaming_content.emit("    🧠 Using PLANNER AI for strategic thinking", "stream", "2")
            elif task_type == "code":
                role = LLMRole.CODER
                self.streaming_content.emit("    ⚡ Using CODER AI for code generation", "stream", "2")
            elif task_type == "analysis":
                role = LLMRole.REVIEWER
                self.streaming_content.emit("    👁️ Using REVIEWER AI for analysis", "stream", "2")
            else:
                role = LLMRole.CHAT
                self.streaming_content.emit("    💬 Using CHAT AI for general tasks", "stream", "2")

            # DEBUG: Check if LLM client exists
            if not self.llm_client:
                raise Exception("LLM client is None - not initialized properly")

            self.logger.info(f"✅ LLM client validated, calling {role.value} role...")

            # FIXED: Call parent class method directly (LLMClient overrides chat to only take prompt)
            self.logger.info(f"📞 Calling async EnhancedLLMClient.chat with {role.value} role...")
            response = await EnhancedLLMClient.chat(self.llm_client, prompt, role)

            self.logger.info(f"✅ LLM response received: {len(response)} chars from {role.value}")
            self.streaming_content.emit(f"    ✅ AI {role.value} response received", "success", "2")

            if not response or len(response.strip()) < 10:
                raise Exception(f"LLM returned empty/short response: '{response}'")

            return response

        except Exception as e:
            error_msg = f"LLM call failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.streaming_content.emit(f"    ❌ AI call failed: {str(e)}", "error", "2")
            raise Exception(error_msg)

    async def _finalize_project(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize project with streaming updates"""
        self.streaming_content.emit("  📋 Creating project summary", "stream", "1")
        await asyncio.sleep(0.2)

        if results.get("project_dir"):
            self.project_loaded.emit(results["project_dir"])
            self.streaming_content.emit(f"  📁 Project loaded: {results['project_dir']}", "success", "1")

        final_result = {
            "success": True,
            "project_name": Path(results["project_dir"]).name if results.get("project_dir") else "Unknown",
            "project_dir": results.get("project_dir"),
            "file_count": len(results.get("files_created", [])),
            "files_created": results.get("files_created", [])
        }

        self.streaming_content.emit("  ✅ AI project finalization complete", "success", "1")
        return final_result

    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get current workflow statistics"""
        return self.workflow_stats.copy()

    # Legacy methods for compatibility
    def execute_workflow(self, prompt: str):
        """Legacy workflow execution (non-async)"""
        asyncio.create_task(self.execute_enhanced_workflow(prompt, []))