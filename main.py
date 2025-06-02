# main.py - REVISED APPROACH

import sys
import asyncio
from pathlib import Path
from PySide6.QtWidgets import QApplication
import qasync

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.application import AvAApplication


async def startup_ava(app: QApplication, app_instance_container: list):
    """
    Coroutine to initialize the AvA application.
    This will be scheduled onto the qasync event loop.
    """
    print("startup_ava: Coroutine started.")
    ava_app = AvAApplication()
    app_instance_container.append(ava_app)  # Store the instance

    # Connect the fully_initialized_signal for status printing
    def on_fully_initialized():
        print("AvAApplication fully initialized (async components complete).")
        # Ensure ava_app is accessible, it should be from the closure
        if app_instance_container:
            current_ava_app = app_instance_container[0]
            status = current_ava_app.get_status()
            print(f"Final Status - Models available: {status['llm_models']}")
            rag_status_info = status.get('rag', {})
            rag_ready = rag_status_info.get('ready', False)
            rag_available = rag_status_info.get('available', False)
            rag_text = rag_status_info.get('status_text', 'Status Unknown')

            if rag_available:
                print(f"Final Status - RAG available: {rag_ready} ({rag_text})")
            else:
                print(f"Final Status - RAG not available: {rag_text}")

            # Update main window title example (if needed)
            if current_ava_app.main_window:
                project_name = current_ava_app.current_project
                session_name = current_ava_app.current_session
                chat_model = current_ava_app.current_config.get("chat_model", "N/A")
                # Example of updating title, customize as needed
                # current_ava_app.main_window.setWindowTitle(f"AvA [{project_name}] - Session: {session_name} (LLM: {chat_model})")

    ava_app.fully_initialized_signal.connect(on_fully_initialized)

    # Call initialize. This method is now designed to:
    # 1. Perform synchronous UI setup.
    # 2. Use QTimer.singleShot(0, ...) to schedule its async_initialize_components.
    # This ensures that asyncio.create_task within async_initialize_components
    # (or further down the call chain like in RAGManager) has a running event loop.
    ava_app.initialize()
    print("startup_ava: AvAApplication.initialize() called. Async parts are scheduled via QTimer.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("AvA")
    app.setApplicationDisplayName("AvA - AI Development Assistant")
    app.setApplicationVersion("2.0")

    # Set up qasync event loop
    # This QEventLoop will be the one asyncio.get_running_loop() returns
    # once app.exec() starts processing events.
    qasync_loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(qasync_loop)

    ava_app_instance_container = []  # To hold the AvAApplication instance

    try:
        # Schedule the startup coroutine.
        # asyncio.ensure_future (or create_task) will add it to the asyncio loop.
        # qasync's QEventLoop will ensure it gets processed when the Qt loop runs.
        # This task will run, call ava_app.initialize(), which then uses QTimer
        # to schedule the next async part.
        startup_task = asyncio.ensure_future(startup_ava(app, ava_app_instance_container))

        print("main.py: QApplication.exec() is about to be called.")
        # Start the Qt event loop. This will also drive the asyncio event loop via qasync.
        exit_code = app.exec()
        print(f"main.py: QApplication.exec() finished with code {exit_code}.")

        # Optional: Graceful shutdown
        if ava_app_instance_container:
            ava_app_instance = ava_app_instance_container[0]
            if hasattr(ava_app_instance, 'shutdown'):
                print("main.py: Calling AvAApplication.shutdown().")
                ava_app_instance.shutdown()  # Assuming shutdown is synchronous for simplicity here

        sys.exit(exit_code)

    except Exception as e:
        print(f"Failed to launch AvA: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

# core/application_enhanced.py - Enhanced application class for new UI
# NOTE: This is just a comment placeholder for the context of combined.txt.
# The actual content of this file is expected to be the full version you have.
from core.application import AvAApplication as BaseAvAApplication
from PySide6.QtCore import Signal


class AvAApplicationEnhanced(BaseAvAApplication):
    """Enhanced AvA Application with better UI integration"""

    # Additional signals for enhanced UI
    model_status_changed = Signal(str, str, bool)  # (model, status, available)
    project_loaded = Signal(str)  # project_path
    session_changed = Signal(str)  # session_name

    def __init__(self):
        super().__init__()
        self.current_session = "Main Chat"
        self.current_project = "Default Project"

    def get_enhanced_status(self):
        """Get enhanced status for UI"""
        base_status = self.get_status()
        return {
            **base_status,
            "current_project": self.current_project,
            "current_session": self.current_session,
            "ui_ready": True
        }

    def create_new_session(self, name: str):
        """Create a new chat session"""
        self.current_session = name
        self.session_changed.emit(name)
        self.logger.info(f"Created new session: {name}")

    def load_project(self, project_path: str):
        """Load a project"""
        self.current_project = Path(project_path).name
        self.project_loaded.emit(project_path)
        self.logger.info(f"Loaded project: {self.current_project}")

    def get_available_models_formatted(self):
        """Get models formatted for UI display"""
        if not self.llm_client:
            return []

        models = self.llm_client.get_available_models()
        formatted = []

        for model in models:
            if "gemini" in model.lower():
                formatted.append(f"Google: {model}")
            elif "gpt" in model.lower():
                formatted.append(f"OpenAI: {model}")
            elif "claude" in model.lower():
                formatted.append(f"Anthropic: {model}")
            elif "ollama" in model.lower() or "qwen" in model.lower():
                formatted.append(f"Local: {model}")
            else:
                formatted.append(model)

        return formatted


# gui/components_enhanced.py - Enhanced components for new UI
# NOTE: This is just a comment placeholder for the context of combined.txt.
# The actual content of this file is expected to be the full version you have.
from PySide6.QtWidgets import QPushButton, QFrame, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPainter, QLinearGradient, QColor


class AnimatedButton(QPushButton):
    """Button with hover animations"""

    def __init__(self, text="", button_type="primary"):
        super().__init__(text)
        self.button_type = button_type
        self._animation = QPropertyAnimation(self, b"styleSheet")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._setup_styles()

    def _setup_styles(self):
        """Setup button styles"""
        if self.button_type == "accent":
            self.normal_style = """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #00d7ff, stop:1 #0078d4);
                    border: 2px solid #00d7ff;
                    border-radius: 8px;
                    color: #1e1e1e;
                    padding: 12px 20px;
                    font-weight: bold;
                }
            """
            self.hover_style = """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #40e0ff, stop:1 #00a0f0);
                    border: 2px solid #40e0ff;
                    border-radius: 8px;
                    color: #1e1e1e;
                    padding: 12px 20px;
                    font-weight: bold;
                }
            """
        else:
            # Default styles for other types
            self.normal_style = """
                QPushButton {
                    background: #2d2d30;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    color: #cccccc;
                    padding: 8px 16px;
                }
            """
            self.hover_style = """
                QPushButton {
                    background: #3e3e42;
                    border: 1px solid #00d7ff;
                    color: white;
                    padding: 8px 16px;
                }
            """

        self.setStyleSheet(self.normal_style)

    def enterEvent(self, event):
        """Handle mouse enter"""
        self.setStyleSheet(self.hover_style)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave"""
        self.setStyleSheet(self.normal_style)
        super().leaveEvent(event)


class GlowingProgressBar(QProgressBar):
    """Progress bar with glowing effect"""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QProgressBar {
                background: #2d2d30;
                border: 1px solid #404040;
                border-radius: 6px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:0.5 #40e0ff, stop:1 #00d7ff);
                border-radius: 6px;
                box-shadow: 0 0 10px #00d7ff;
            }
        """)