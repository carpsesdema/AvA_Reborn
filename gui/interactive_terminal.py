# gui/interactive_terminal.py - A Beautiful, Snappy, and Venv-Aware Console

import sys
import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QProcess, QProcessEnvironment
from PySide6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QHBoxLayout, QLabel

from gui.components import Colors, Typography, ModernButton


class InteractiveTerminal(QWidget):
    """
    A beautiful and functional interactive terminal that runs commands
    asynchronously using QProcess and is aware of project virtual environments.
    """
    command_completed = Signal(int)  # Emits the exit code when a command finishes
    force_run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_directory = Path.cwd()

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._handle_output)
        self.process.finished.connect(self._on_command_finished)

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(0)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(Typography.code())
        self.output_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.output_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 8, 0, 0)
        input_layout.setSpacing(8)

        self.prompt_label = QLabel()
        self._update_prompt_label()

        self.input_line = QLineEdit()
        self.input_line.setFont(Typography.code())
        self.input_line.setFrame(False)
        self.input_line.returnPressed.connect(self.run_manual_command)

        self.run_project_btn = ModernButton("▶️ Run Project", button_type="primary")
        self.run_project_btn.setMinimumHeight(32)
        self.run_project_btn.setMaximumWidth(140)
        self.run_project_btn.clicked.connect(self.force_run_requested.emit)
        self.run_project_btn.setToolTip(
            "Run the current project's main file after ensuring venv and dependencies are set up.")

        input_layout.addWidget(self.prompt_label)
        input_layout.addWidget(self.input_line, 1)
        input_layout.addWidget(self.run_project_btn)

        main_layout.addWidget(self.output_area, 1)
        main_layout.addLayout(input_layout)

    def _apply_style(self):
        self.setStyleSheet(
            f"background: {Colors.PRIMARY_BG}; border: 1px solid {Colors.BORDER_DEFAULT}; border-radius: 8px;")
        self.output_area.setStyleSheet(f"""
            QTextEdit {{ background: {Colors.PRIMARY_BG}; color: {Colors.TEXT_PRIMARY}; border: none; padding: 8px; }}
            QScrollBar:vertical {{ background: {Colors.PRIMARY_BG}; width: 8px; border-radius: 4px; }}
            QScrollBar::handle:vertical {{ background: {Colors.BORDER_DEFAULT}; border-radius: 4px; }}
            QScrollBar::handle:vertical:hover {{ background: {Colors.ACCENT_BLUE}; }}
        """)
        self.prompt_label.setStyleSheet(
            f"background: transparent; color: {Colors.ACCENT_BLUE}; font-weight: bold; padding-top: 2px;")
        self.input_line.setStyleSheet(
            f"background: transparent; color: {Colors.TEXT_PRIMARY}; border: none; padding: 4px 0px;")
        self.run_project_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {Colors.ACCENT_GREEN}, stop:1 #2d863a);
                color: white; border: 2px solid #2d863a; border-radius: 6px; padding: 6px 12px; font-weight: bold;
            }}
            QPushButton:hover {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4dce5d, stop:1 #3fb950); border-color: #3fb950; }}
            QPushButton:pressed {{ background: #2d863a; }}
            QPushButton:disabled {{ background: {Colors.BORDER_DEFAULT}; color: {Colors.TEXT_MUTED}; border-color: {Colors.BORDER_MUTED}; }}
        """)

    def set_working_directory(self, path: str):
        self.working_directory = Path(path).resolve()
        if not self.working_directory.exists() or not self.working_directory.is_dir():
            self.append_error(f"Directory not found, reverting to default: {path}")
            self.working_directory = Path.cwd()
        self._update_prompt_label()
        self.append_system_message(f"Working directory set to: {self.working_directory}")

    def _update_prompt_label(self):
        prompt_path = str(self.working_directory.name)
        venv_path = self.working_directory / 'venv'
        if venv_path.exists():
            self.prompt_label.setText(f"({prompt_path}) >")
        else:
            self.prompt_label.setText(f"{prompt_path} >")

    def _get_venv_environment(self) -> QProcessEnvironment:
        """Creates a QProcessEnvironment with the venv's PATH prepended."""
        venv_path = self.working_directory / 'venv'
        if not venv_path.exists():
            return QProcessEnvironment.systemEnvironment()

        env = QProcessEnvironment.systemEnvironment()
        original_path = env.value("PATH")

        # Determine script path based on OS
        if sys.platform == "win32":
            scripts_path = str(venv_path / "Scripts")
        else:
            scripts_path = str(venv_path / "bin")

        # Prepend the venv path to the system PATH
        new_path = f"{scripts_path}{os.pathsep}{original_path}"
        env.insert("PATH", new_path)

        # Unset PYTHONHOME if it's set, as it can interfere with venvs
        env.remove("PYTHONHOME")

        self.append_system_message(f"Activated virtual environment: {venv_path.name}")
        return env

    def execute_command(self, program: str, arguments: list):
        """Executes a command using QProcess, activating venv if available."""
        if self.process.state() == QProcess.ProcessState.Running:
            self.append_error("A command is already running.")
            return

        self.process.setWorkingDirectory(str(self.working_directory))

        # Set the venv-aware environment for the process
        self.process.setProcessEnvironment(self._get_venv_environment())

        self.append_command(f"{program} {' '.join(arguments)}")
        self.process.start(program, arguments)
        if not self.process.waitForStarted(2000):  # Wait 2s for process to start
            self.append_error(f"Error starting process: {self.process.errorString()}")

    def run_manual_command(self):
        """Executes a command manually entered by the user."""
        command_text = self.input_line.text().strip()
        if not command_text: return
        self.input_line.clear()

        # Simple parsing for program and args
        parts = command_text.split()
        program = parts[0]
        args = parts[1:]
        self.execute_command(program, args)

    def _handle_output(self):
        """Handle new data available on stdout/stderr."""
        data = self.process.readAll()
        self.append_output(data.data().decode(errors='ignore'))

    def _on_command_finished(self, exit_code, exit_status):
        """Handle the completion of a command."""
        status_text = "successfully" if exit_code == 0 else f"with error code {exit_code}"
        self.append_system_message(f"Process finished {status_text}.\n")
        self.input_line.setFocus()
        self.command_completed.emit(exit_code)

    def clear_terminal(self):
        self.output_area.clear()

    def append_output(self, text: str):
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.output_area.ensureCursorVisible()

    def append_error(self, text: str):
        self._append_formatted_text(text, Colors.ACCENT_RED)

    def append_system_message(self, text: str):
        self._append_formatted_text(f"{text}\n", Colors.TEXT_MUTED, italic=True)

    def append_command(self, command: str):
        self.append_output("\n")
        # Update prompt label to include venv indicator if present
        current_prompt_text = self.prompt_label.text()
        self._append_formatted_text(f"{current_prompt_text} ", Colors.ACCENT_BLUE, bold=True)
        self._append_formatted_text(f"{command}\n", Colors.TEXT_PRIMARY)

    def _append_formatted_text(self, text: str, color: str, bold=False, italic=False):
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        char_format = QTextCharFormat()
        char_format.setForeground(QColor(color))
        char_format.setFontWeight(QFont.Weight.Bold if bold else QFont.Weight.Normal)
        char_format.setFontItalic(italic)
        cursor.mergeCharFormat(char_format)
        cursor.insertText(text)
        self.output_area.setCurrentCharFormat(QTextCharFormat())
        self.output_area.ensureCursorVisible()