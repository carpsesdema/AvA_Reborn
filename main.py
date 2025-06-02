# core/application.py - Updated to use enhanced main window

# main.py - FIXED VERSION

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gui.main_window import AvAMainWindow
from core.application import AvAApplication


def main():
    """Main entry point"""

    app = QApplication(sys.argv)
    app.setApplicationName("AvA")
    app.setApplicationDisplayName("AvA - AI Development Assistant")
    app.setApplicationVersion("2.0")

    try:
        # Create AvA application (backend)
        ava_app = AvAApplication()

        # Initialize AvA backend
        ava_app.initialize()

        # Connect the UI to the AvA application
        _connect_ui_to_backend(ava_app.main_window, ava_app)

        print("AvA launched successfully!")
        print("Available models:", ava_app.llm_client.get_available_models() if ava_app.llm_client else "None")

        return app.exec()

    except Exception as e:
        print(f"Failed to launch AvA: {e}")
        import traceback
        traceback.print_exc()
        return 1


def _connect_ui_to_backend(main_window, ava_app):
    """Connect the UI to the AvA backend"""

    # Connect sidebar actions to backend windows
    def handle_sidebar_action(action):
        if action == "view_log" or action == "open_terminal":
            ava_app._open_terminal()
        elif action == "view_code" or action == "open_code_viewer":
            ava_app._open_code_viewer()
        elif action == "new_session":
            print("New session requested")
        elif action == "force_gen":
            print("Force code generation requested")
        elif action == "check_updates":
            print("Update check requested")

    main_window.sidebar.action_triggered.connect(handle_sidebar_action)


if __name__ == "__main__":
    sys.exit(main())

# core/application_enhanced.py - Enhanced application class for new UI
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


if __name__ == "__main__":
    # Run the enhanced version
    sys.exit(main())