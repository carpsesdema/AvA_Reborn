import asyncio  # Ensure asyncio is imported
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication

from core.llm_client import LLMClient
from core.workflow_engine import WorkflowEngine

# NEW: Import enhanced workflow components
try:
    from core.enhanced_workflow_engine import EnhancedWorkflowEngine
    from core.project_state_manager import ProjectStateManager
    from core.ai_feedback_system import AIFeedbackSystem

    ENHANCED_WORKFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced workflow components not available: {e}")
    ENHANCED_WORKFLOW_AVAILABLE = False

# Import the components
from gui.main_window import AvAMainWindow
from windows.code_viewer import CodeViewerWindow
# FIXED: Import the enhanced terminal instead of the simple one
from gui.terminals import TerminalWindow

# Try to import RAG manager - gracefully handle if not available
try:
    from core.rag_manager import RAGManager

    RAG_MANAGER_AVAILABLE = True
except ImportError as e:
    RAG_MANAGER_AVAILABLE = False
    print(f"RAG Manager not available: {e}")


class AvAApplication(QObject):
    """
    AvA Application - Enhanced with Project Awareness & AI Collaboration
    """
    fully_initialized_signal = Signal()  # Signal to indicate all async init is done

    # Signals for status updates
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    error_occurred = Signal(str, str)
    rag_status_changed = Signal(str, str)  # For RAGManager to emit to this app instance
    project_loaded = Signal(str)

    # NEW: Enhanced workflow signals
    ai_collaboration_started = Signal(str)  # session_id
    ai_feedback_received = Signal(str, str, str)  # from_ai, to_ai, content
    iteration_completed = Signal(str, int)  # file_path, iteration_number
    quality_check_completed = Signal(str, bool, str)  # file_path, approved, feedback

    def __init__(self):
        super().__init__()

        # Initialize logging (Windows-compatible)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.llm_client = None
        self.main_window = None
        self.terminal_window = None
        self.code_viewer = None
        self.workflow_engine = None
        self.rag_manager = None

        # NEW: Enhanced workflow components
        self.enhanced_workflow_engine = None
        self.project_state_manager = None
        self.ai_feedback_system = None
        self.use_enhanced_workflow = ENHANCED_WORKFLOW_AVAILABLE

        # Application state
        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)

        self.current_project = "Default Project"
        self.current_session = "Main Chat"
        self.active_workflows = {}

        # Configuration
        self.current_config = {
            "chat_model": "Gemini: gemini-2.5-pro-preview-05-06",
            "code_model": "qwen2.5-coder:14b",
            "temperature": 0.7,
            # NEW: Enhanced workflow settings
            "enable_iterations": True,
            "max_iterations": 3,
            "pause_for_feedback": False
        }

        # NEW: Performance monitoring timer
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance_stats)
        self.performance_timer.start(5000)  # Update every 5 seconds

    def _setup_logging(self):
        """Setup Windows-compatible logging without Unicode emojis"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                emoji_replacements = {
                    '🚀': '[LAUNCH]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARNING]',
                    '🧠': '[BRAIN]', '📊': '[STATS]', '🔧': '[TOOL]', '📄': '[FILE]',
                    '📝': '[EDIT]', '📂': '[FOLDER]', '🤖': '[AI]', '🤝': '[COLLAB]'
                }
                for emoji, replacement in emoji_replacements.items():
                    msg = msg.replace(emoji, replacement)
                return msg

        formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_dir / "ava.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    async def initialize(self):
        self.logger.info("[LAUNCH] Initializing AvA Application with Enhanced AI Collaboration...")

        self.main_window = AvAMainWindow(ava_app=self)
        self.logger.info("[OK] Main window initialized")

        # FIXED: Use the enhanced terminal with progress tracking
        self.terminal_window = TerminalWindow()
        self.logger.info("[OK] Enhanced terminal window initialized")

        self.code_viewer = CodeViewerWindow()
        self.logger.info("[OK] Code viewer initialized")

        self._initialize_core_services()

        if self.main_window and not self.main_window.isVisible():
            self.main_window.show()
            self.logger.info("Main window shown.")
            await asyncio.sleep(0.01)

        self.logger.info("Starting async component initialization...")
        await self.async_initialize_components()
        self.logger.info("Async component initialization completed.")

    async def async_initialize_components(self):
        self.logger.info("[LAUNCH] Initializing async components with enhanced workflow...")
        try:
            await self._initialize_rag_manager_async()
            self.logger.info(
                f"RAG Manager initialization complete. Ready: {self.rag_manager.is_ready if self.rag_manager and hasattr(self.rag_manager, 'is_ready') else 'N/A'}")

            # NEW: Initialize enhanced workflow components
            if self.use_enhanced_workflow:
                await self._initialize_enhanced_workflow()
            else:
                self._initialize_workflow_engine()

            self._connect_components()
            self._setup_window_behaviors()

            status = self.get_status()
            self.logger.info(f"System status after initialization: {status}")
            self.logger.info("[OK] AvA Application with enhanced workflow initialized successfully.")

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize AvA components: {e}", exc_info=True)
            self.error_occurred.emit("async_initialization", str(e))
        finally:
            self.fully_initialized_signal.emit()

    def _initialize_core_services(self):
        self.logger.info("Initializing core services...")
        self.llm_client = LLMClient()
        available_models = self.llm_client.get_available_models()
        self.logger.info(f"Available LLM models: {available_models}")
        if not available_models or available_models == ["No LLM services available"]:
            self.logger.warning("[WARNING] No LLM services available!")

    async def _initialize_enhanced_workflow(self):
        """NEW: Initialize enhanced workflow components"""
        try:
            self.logger.info("[AI] Initializing project-aware AI collaboration system...")

            # Initialize project state manager
            self.project_state_manager = ProjectStateManager(self.workspace_dir)
            self.logger.info("[OK] Project state manager initialized")

            # Initialize AI feedback system
            self.ai_feedback_system = AIFeedbackSystem(
                self.llm_client,
                self.project_state_manager,
                self.terminal_window
            )
            self.logger.info("[OK] AI feedback system initialized")

            # Initialize enhanced workflow engine
            self.enhanced_workflow_engine = EnhancedWorkflowEngine(
                llm_client=self.llm_client,
                terminal_window=self.terminal_window,
                code_viewer=self.code_viewer,
                rag_manager=self.rag_manager,
                project_root=str(self.workspace_dir)
            )

            # Set as primary workflow engine
            self.workflow_engine = self.enhanced_workflow_engine
            self.logger.info("[OK] Enhanced workflow engine initialized")

        except Exception as e:
            self.logger.error(f"[ERROR] Enhanced workflow initialization failed: {e}")
            self.logger.info("Falling back to standard workflow engine...")
            self.use_enhanced_workflow = False
            self._initialize_workflow_engine()

    async def _initialize_rag_manager_async(self):
        self.logger.info("Attempting to initialize RAG manager (async)...")
        if RAG_MANAGER_AVAILABLE:
            try:
                self.rag_manager = RAGManager()
                self.rag_manager.status_changed.connect(self._on_rag_status_changed)
                self.rag_manager.upload_completed.connect(self._on_rag_upload_completed)
                await self.rag_manager.async_initialize()
                self.logger.info(
                    f"[OK] RAG manager initialized. Ready: {self.rag_manager.is_ready}")
            except Exception as e:
                self.logger.error(f"[ERROR] RAG manager initialization failed: {e}", exc_info=True)
                self.rag_manager = None
                self._on_rag_status_changed(f"RAG Init Exception: {e}", "error")
        else:
            self.logger.info("RAG services not available - running without RAG functionality.")
            self.rag_manager = None
            self._on_rag_status_changed("RAG: Not Available (Import Fail)", "grey")

    def _initialize_workflow_engine(self):
        """Fallback: Initialize standard workflow engine"""
        self.logger.info("Initializing standard workflow engine...")
        self.workflow_engine = WorkflowEngine(
            self.llm_client,
            self.terminal_window,
            self.code_viewer,
            self.rag_manager
        )
        self.logger.info("[OK] Standard workflow engine initialized.")

    def _connect_components(self):
        self.logger.info("Connecting components...")
        if self.main_window:
            self.main_window.workflow_requested.connect(self._handle_workflow_request)
            if hasattr(self.main_window, 'new_project_requested'):
                self.main_window.new_project_requested.connect(self.create_new_project_dialog)

        # Connect workflow signals
        self.workflow_started.connect(self._on_workflow_started)
        self.workflow_completed.connect(self._on_workflow_completed)
        self.error_occurred.connect(self._on_error_occurred)

        if self.workflow_engine:
            if self.main_window:
                self.workflow_engine.workflow_started.connect(self.main_window.on_workflow_started)
                self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)

            # NEW: Connect enhanced workflow signals if available
            if self.use_enhanced_workflow and hasattr(self.workflow_engine, 'ai_collaboration_started'):
                self.workflow_engine.ai_collaboration_started.connect(self.ai_collaboration_started)
                self.workflow_engine.ai_feedback_received.connect(self.ai_feedback_received)
                self.workflow_engine.iteration_completed.connect(self.iteration_completed)
                self.workflow_engine.quality_check_completed.connect(self.quality_check_completed)

                # Connect to main window feedback panel if available
                if hasattr(self.main_window, 'feedback_panel'):
                    self.workflow_engine.ai_collaboration_started.connect(
                        lambda session_id: self.main_window.feedback_panel.add_ai_collaboration_message(
                            "system", "", f"Collaboration session started: {session_id}"
                        )
                    )
                    self.workflow_engine.ai_feedback_received.connect(
                        self.main_window.feedback_panel.add_ai_collaboration_message
                    )
                    self.workflow_engine.quality_check_completed.connect(
                        lambda file_path, approved, feedback:
                        self.main_window.feedback_panel.show_file_completed(file_path, approved, 1)
                    )

            # Connect progress updates to terminal
            if self.terminal_window and hasattr(self.terminal_window, 'update_workflow_progress'):
                self.workflow_engine.workflow_progress.connect(self.terminal_window.update_workflow_progress)
                self.logger.info("[OK] Workflow progress connected to terminal")

            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)

        self.logger.info("[OK] Components connected.")

    def _setup_window_behaviors(self):
        if self.main_window:
            title = "AvA - Enhanced AI Development Assistant" if self.use_enhanced_workflow else "AvA - AI Development Assistant"
            self.main_window.setWindowTitle(title)
        if self.terminal_window:
            self.terminal_window.setWindowTitle("AvA - Enhanced Workflow Terminal")
        if self.code_viewer:
            self.code_viewer.setWindowTitle("AvA - Code Viewer")
        self._position_windows()
        self.logger.info("[OK] Window behaviors set up.")

    def _position_windows(self):
        screen_geo = QApplication.primaryScreen().geometry()
        if self.main_window:
            self.main_window.setGeometry(100, 50, 1400, 900)
        if self.terminal_window:
            self.terminal_window.setGeometry(screen_geo.width() - 850, screen_geo.height() - 400, 900, 700)
        if self.code_viewer:
            self.code_viewer.setGeometry(screen_geo.width() - 1000, 50, 950, 700)

    def _update_performance_stats(self):
        """Update performance statistics in terminal"""
        if not (self.terminal_window and self.workflow_engine):
            return

        try:
            stats = self.workflow_engine.get_workflow_stats()
            cache_stats = stats.get("cache_stats", {})

            cache_size = cache_stats.get("cache_size", 0)
            hit_rate = cache_stats.get("hit_rate", 0.0)

            if hasattr(self.terminal_window, 'update_cache_status'):
                self.terminal_window.update_cache_status(cache_size, hit_rate)

            workflow_state = stats.get("workflow_state", {})
            if workflow_state.get("stage") not in ["idle", "complete", "error"]:
                completed = workflow_state.get("completed_tasks", 0)
                total = workflow_state.get("total_tasks", 0)
                if hasattr(self.terminal_window, 'update_task_progress'):
                    self.terminal_window.update_task_progress(completed, total)

        except Exception as e:
            self.logger.debug(f"Performance stats update failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        llm_models_list = self.llm_client.get_available_models() if self.llm_client else ["LLM Client not init"]
        rag_info = {"ready": False, "status_text": "RAG: Not Initialized", "available": RAG_MANAGER_AVAILABLE,
                    "collections": {}}
        if RAG_MANAGER_AVAILABLE:
            if self.rag_manager and hasattr(self.rag_manager, 'is_ready') and hasattr(self.rag_manager,
                                                                                      'current_status'):
                rag_info["ready"] = self.rag_manager.is_ready
                rag_info["status_text"] = self.rag_manager.current_status
                if hasattr(self.rag_manager, 'get_collection_info') and callable(
                        getattr(self.rag_manager, 'get_collection_info')):
                    rag_info["collections"] = self.rag_manager.get_collection_info()
            elif self.rag_manager:
                rag_info["status_text"] = "RAG: Manager exists, status uncertain"
            else:
                rag_info["status_text"] = "RAG: Initialization Failed (Manager None)"
        else:
            rag_info["status_text"] = "RAG: Dependencies Missing / Not Available"

        # Include performance stats and enhanced workflow info
        performance_stats = {}
        enhanced_info = {}

        if self.workflow_engine:
            try:
                workflow_stats = self.workflow_engine.get_workflow_stats()
                performance_stats = {
                    "cache_stats": workflow_stats.get("cache_stats", {}),
                    "workflow_state": workflow_stats.get("workflow_state", {})
                }

                # NEW: Enhanced workflow specific info
                if self.use_enhanced_workflow:
                    enhanced_info = {
                        "project_awareness": self.project_state_manager is not None,
                        "ai_collaboration": self.ai_feedback_system is not None,
                        "project_files": len(self.project_state_manager.files) if self.project_state_manager else 0,
                        "ai_decisions": len(
                            self.project_state_manager.ai_decisions) if self.project_state_manager else 0
                    }
            except Exception:
                pass

        return {
            "ready": self.workflow_engine is not None,
            "enhanced_workflow": self.use_enhanced_workflow,  # NEW
            "llm_models": llm_models_list,
            "workspace": str(self.workspace_dir),
            "current_project": self.current_project,
            "current_session": self.current_session,
            "configuration": self.current_config,
            "rag": rag_info,
            "performance": performance_stats,
            "enhanced": enhanced_info,  # NEW
            "windows": {
                "main": self.main_window.isVisible() if self.main_window else False,
                "terminal": self.terminal_window.isVisible() if self.terminal_window else False,
                "code_viewer": self.code_viewer.isVisible() if self.code_viewer else False
            },
            "active_workflows": len(self.active_workflows)
        }

    def _handle_workflow_request(self, user_prompt: str):
        self.logger.info(f"Workflow request received: {user_prompt[:100]}...")
        current_status = self.get_status()

        # Check for LLM availability
        if not current_status["llm_models"] or current_status["llm_models"] == ["LLM Client not init"] or \
                current_status["llm_models"] == ["No LLM services available"]:
            error_msg = "No LLM services available. Please configure API keys."
            if self.terminal_window:
                self.terminal_window.log(f"[ERROR] Workflow Halted: {error_msg}")
            self.error_occurred.emit("llm_unavailable", error_msg)
            return

        if not self.workflow_engine:
            error_msg = "Workflow engine not initialized. Cannot process request."
            if self.terminal_window:
                self.terminal_window.log(f"[ERROR] Workflow Halted: {error_msg}")
            self.error_occurred.emit("workflow_engine_unavailable", error_msg)
            return

        self._open_terminal()
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {
            "prompt": user_prompt,
            "start_time": datetime.now(),
            "status": "running",
            "enhanced": self.use_enhanced_workflow
        }

        self.workflow_started.emit(user_prompt)

        try:
            # NEW: Use enhanced workflow if available
            if self.use_enhanced_workflow and hasattr(self.workflow_engine, 'execute_enhanced_workflow'):
                self.logger.info("[AI] Starting enhanced workflow with AI collaboration...")
                settings = self.current_config
                asyncio.create_task(self.workflow_engine.execute_enhanced_workflow(
                    user_prompt,
                    enable_iterations=settings.get("enable_iterations", True),
                    max_iterations=settings.get("max_iterations", 3)
                ))
            else:
                # Fallback to standard workflow
                self.logger.info("Starting standard workflow...")
                self.workflow_engine.execute_workflow(user_prompt)

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            self.error_occurred.emit("workflow_execution_error", str(e))
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)

    # NEW: Enhanced workflow control methods
    def update_workflow_settings(self, settings: Dict[str, Any]):
        """Update enhanced workflow settings"""
        self.current_config.update(settings)
        self.logger.info(f"[AI] Workflow settings updated: {settings}")

        if self.use_enhanced_workflow and hasattr(self.workflow_engine, 'set_pause_for_feedback'):
            self.workflow_engine.set_pause_for_feedback(settings.get("pause_for_feedback", False))

    def add_user_feedback(self, feedback_type: str, content: str, rating: int = 5, file_path: str = None):
        """Add user feedback to the enhanced workflow"""
        if self.use_enhanced_workflow and hasattr(self.workflow_engine, 'add_user_feedback'):
            self.workflow_engine.add_user_feedback(feedback_type, content, rating, file_path)
            self.logger.info(f"[AI] User feedback added: {feedback_type}")

    def request_file_iteration(self, file_path: str, feedback: str) -> bool:
        """Request iteration for a specific file"""
        if self.use_enhanced_workflow and hasattr(self.workflow_engine, 'request_file_iteration'):
            return asyncio.create_task(self.workflow_engine.request_file_iteration(file_path, feedback))
        return False

    def _open_terminal(self):
        if self.terminal_window:
            self.terminal_window.show()
            self.terminal_window.raise_()
            self.terminal_window.activateWindow()

    def _open_code_viewer(self):
        if self.code_viewer:
            self.code_viewer.show()
            self.code_viewer.raise_()
            self.code_viewer.activateWindow()

    def _on_file_generated(self, file_path: str):
        self.logger.info(f"File generated: {file_path}")
        if self.code_viewer:
            self.code_viewer.auto_open_file(file_path)

    def _on_project_loaded(self, project_path: str):
        self.logger.info(f"Project loaded: {project_path}")
        self.current_project = Path(project_path).name
        if self.main_window and hasattr(self.main_window, 'update_project_display'):
            self.main_window.update_project_display(self.current_project)
        self._open_code_viewer()

    def _on_workflow_started(self, prompt: str):
        self.logger.info(f"[OK] Workflow started: {prompt}")
        if self.main_window and hasattr(self.main_window, 'on_workflow_started'):
            self.main_window.on_workflow_started(prompt)

    def _on_workflow_completed(self, result: dict):
        self.logger.info(f"[OK] Workflow completed: {result.get('project_name', 'N/A')}")
        for wf_id, wf_data in self.active_workflows.items():
            if wf_data["status"] == "running":
                wf_data["status"] = "completed" if result.get("success") else "failed"
                wf_data["end_time"] = datetime.now()
                break
        if self.main_window and hasattr(self.main_window, 'on_workflow_completed'):
            self.main_window.on_workflow_completed(result)

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"[ERROR] App Error in {component}: {error_message}")
        if self.main_window and hasattr(self.main_window, 'on_app_error_occurred'):
            self.main_window.on_app_error_occurred(component, error_message)

    def _on_rag_status_changed(self, status_text: str, color: str):
        self.logger.info(f"RAG Status Update: {status_text} (Color: {color})")
        self.rag_status_changed.emit(status_text, color)

    def _on_rag_upload_completed(self, collection_id: str, files_processed: int):
        msg = f"RAG: {files_processed} files added to {collection_id}"
        self.logger.info(f"RAG Upload Completed: {msg}")
        self.rag_status_changed.emit(msg, "#4ade80")

    def update_configuration(self, config_updates: dict):
        self.current_config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")
        if self.main_window and hasattr(self.main_window, '_update_initial_ui_status'):
            self.main_window._update_initial_ui_status()

    def create_new_project_dialog(self):
        """Enhanced: Allow creating new directories or selecting existing ones"""
        self.logger.info("Request to create a new project received.")
        if self.terminal_window:
            self.terminal_window.log("✨ New Project creation requested.")

        from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

        if not self.main_window:
            return

        # First, ask user if they want to create new or select existing
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel

        class ProjectDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("New Project")
                self.setModal(True)
                self.result_action = None

                layout = QVBoxLayout()

                # Title
                title = QLabel("Create or Select Project")
                title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d7ff; margin: 10px;")
                layout.addWidget(title)

                # Description
                desc = QLabel("Choose how to set up your project:")
                desc.setStyleSheet("color: #cccccc; margin: 10px;")
                layout.addWidget(desc)

                # Buttons
                button_layout = QHBoxLayout()

                self.create_btn = QPushButton("📁 Create New Directory")
                self.create_btn.setStyleSheet("""
                    QPushButton {
                        background: #0078d4;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 10px 15px;
                        font-weight: bold;
                    }
                    QPushButton:hover { background: #106ebe; }
                """)
                self.create_btn.clicked.connect(lambda: self.set_result("create"))

                self.select_btn = QPushButton("📂 Select Existing Directory")
                self.select_btn.setStyleSheet("""
                    QPushButton {
                        background: #2d2d30;
                        color: #cccccc;
                        border: 1px solid #404040;
                        border-radius: 6px;
                        padding: 10px 15px;
                    }
                    QPushButton:hover { background: #3e3e42; }
                """)
                self.select_btn.clicked.connect(lambda: self.set_result("select"))

                self.cancel_btn = QPushButton("Cancel")
                self.cancel_btn.setStyleSheet(self.select_btn.styleSheet())
                self.cancel_btn.clicked.connect(self.reject)

                button_layout.addWidget(self.create_btn)
                button_layout.addWidget(self.select_btn)
                button_layout.addWidget(self.cancel_btn)

                layout.addLayout(button_layout)
                self.setLayout(layout)

                # Style the dialog
                self.setStyleSheet("""
                    QDialog {
                        background: #1e1e1e;
                        border: 2px solid #00d7ff;
                        border-radius: 8px;
                    }
                """)

            def set_result(self, action):
                self.result_action = action
                self.accept()

        dialog = ProjectDialog(self.main_window)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.logger.info("New project creation cancelled by user.")
            return

        action = dialog.result_action
        project_path = None

        if action == "create":
            # Create new directory workflow
            parent_dir = QFileDialog.getExistingDirectory(
                self.main_window,
                "Select Parent Directory for New Project",
                str(self.workspace_dir)
            )

            if not parent_dir:
                return

            # Ask for project name
            project_name, ok = QInputDialog.getText(
                self.main_window,
                "New Project Name",
                "Enter project name:",
                text="my_new_project"
            )

            if not ok or not project_name.strip():
                return

            # Clean project name (make it filesystem-safe)
            import re
            clean_name = re.sub(r'[<>:"/\\|?*]', '_', project_name.strip())
            clean_name = clean_name.replace(' ', '_')

            project_path = Path(parent_dir) / clean_name

            # Create the directory
            try:
                project_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created new project directory: {project_path}")
                if self.terminal_window:
                    self.terminal_window.log(f"📁 Created new project directory: {project_path.name}")

                # Create a basic README file
                readme_path = project_path / "README.md"
                readme_content = f"""# {clean_name}

Created with AvA - AI Development Assistant

## Project Description
This project was created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

## Getting Started
1. Open this project in AvA's Code Viewer
2. Use the chat interface to describe what you want to build
3. Let AvA generate the code for you!

## Files
- This README.md file
- Additional files will be generated by AvA based on your requirements
"""
                readme_path.write_text(readme_content, encoding='utf-8')
                self.logger.info(f"Created README.md in new project")

            except Exception as e:
                QMessageBox.warning(
                    self.main_window,
                    "Error",
                    f"Failed to create project directory:\n{e}"
                )
                self.logger.error(f"Failed to create project directory: {e}")
                return

        elif action == "select":
            # Select existing directory workflow
            project_dir = QFileDialog.getExistingDirectory(
                self.main_window,
                "Select Existing Project Directory",
                str(self.workspace_dir)
            )

            if project_dir:
                project_path = Path(project_dir)
            else:
                return

        if project_path and project_path.exists():
            self.current_project = project_path.name
            self.logger.info(f"Project set to: {project_path}")

            # NEW: Update project state manager if available
            if self.use_enhanced_workflow and self.project_state_manager:
                self.project_state_manager = ProjectStateManager(project_path)
                self.logger.info("[AI] Project state manager updated for new project")

            if hasattr(self.main_window, 'update_project_display'):
                self.main_window.update_project_display(self.current_project)

            self.project_loaded.emit(str(project_path))

            if self.terminal_window:
                self.terminal_window.log(f"🎯 Project set: {self.current_project}")
                self.terminal_window.log(f"📍 Location: {project_path}")

            # Automatically open code viewer with the project
            if self.code_viewer:
                self.code_viewer.load_project(str(project_path))
                self._open_code_viewer()
        else:
            self.logger.error(f"Project path does not exist: {project_path}")

    def shutdown(self):
        self.logger.info("Shutting down AvA Application...")

        # Stop performance monitoring
        if self.performance_timer:
            self.performance_timer.stop()

        # NEW: Save project state if enhanced workflow is active
        if self.use_enhanced_workflow and self.project_state_manager:
            try:
                self.project_state_manager.save_state()
                self.logger.info("[AI] Project state saved")
            except Exception as e:
                self.logger.error(f"Failed to save project state: {e}")

        # Close windows
        if self.main_window:
            self.main_window.close()
        if self.terminal_window:
            self.terminal_window.close()
        if self.code_viewer:
            self.code_viewer.close()

        self.logger.info("AvA Application shutdown complete.")