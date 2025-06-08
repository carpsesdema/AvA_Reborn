# gui/interactive_terminal.py - A Beautiful, Interactive Command Console

import asyncio
import sys
from pathlib import Path
import subprocess
from PySide6.QtCore import Qt, Signal, QProcess
from PySide6.QtGui import QFont, QTextCursor, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QHBoxLayout, QLabel, QPushButton

from gui.components import Colors, Typography, ModernButton


class InteractiveTerminal(QWidget):
    """
    A beautiful and functional interactive terminal for running commands within AvA.
    """
    command_completed = Signal(int)
    force_run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_process = None
        self.working_directory = Path.cwd()

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(0)

        # Output Display Area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(Typography.code())
        self.output_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.output_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Input Area (with new Run button)
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 8, 0, 0)
        input_layout.setSpacing(8)

        self.prompt_label = QLabel()
        self._update_prompt_label()

        self.input_line = QLineEdit()
        self.input_line.setFont(Typography.code())
        self.input_line.setFrame(False)
        self.input_line.returnPressed.connect(self.run_command)

        # Use ModernButton for consistent styling, then customize it
        self.run_project_btn = ModernButton("▶️ Run Project", button_type="primary")
        self.run_project_btn.setMinimumHeight(32)
        self.run_project_btn.setMaximumWidth(140)
        self.run_project_btn.clicked.connect(self.force_run_requested.emit)
        self.run_project_btn.setToolTip("Run the current project and capture output/errors for Ava.")

        input_layout.addWidget(self.prompt_label)
        input_layout.addWidget(self.input_line, 1)
        input_layout.addWidget(self.run_project_btn)

        main_layout.addWidget(self.output_area, 1)
        main_layout.addLayout(input_layout)
        self.setLayout(main_layout)

    def _apply_style(self):
        """Apply the modern, integrated styling."""
        self.setStyleSheet(f"""
            InteractiveTerminal {{
                background: {Colors.PRIMARY_BG};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 8px;
            }}
        """)

        self.output_area.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.PRIMARY_BG};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                padding: 8px;
            }}
            QScrollBar:vertical {{
                background: {Colors.PRIMARY_BG};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER_DEFAULT};
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Colors.ACCENT_BLUE};
            }}
        """)

        self.prompt_label.setStyleSheet(f"""
            background: transparent;
            color: {Colors.ACCENT_BLUE};
            font-weight: bold;
            padding-top: 2px;
        """)

        self.input_line.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                color: {Colors.TEXT_PRIMARY};
                border: none;
                padding: 4px 0px;
            }}
        """)

        # Apply the green 'Run' button styling
        self.run_project_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 {Colors.ACCENT_GREEN}, 
                                          stop: 1 #2d863a);
                color: white;
                border: 2px solid #2d863a;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #4dce5d, 
                                          stop: 1 #3fb950);
                border-color: #3fb950;
            }}
            QPushButton:pressed {{
                background: #2d863a;
            }}
            QPushButton:disabled {{
                background: {Colors.BORDER_DEFAULT};
                color: {Colors.TEXT_MUTED};
                border-color: {Colors.BORDER_MUTED};
            }}
        """)

    def set_working_directory(self, path: str):
        """Sets the current working directory for commands."""
        self.working_directory = Path(path).resolve()
        if not self.working_directory.exists() or not self.working_directory.is_dir():
            self.append_error(f"Directory not found, reverting to default: {path}")
            self.working_directory = Path.cwd()
        self._update_prompt_label()
        self.append_system_message(f"Working directory set to: {self.working_directory}")

    def _update_prompt_label(self):
        """Updates the visual command prompt label."""
        prompt_path = str(self.working_directory)
        if len(prompt_path) > 30:
            prompt_path = f"...{prompt_path[-27:]}"
        self.prompt_label.setText(f"{prompt_path} >")

    def run_command(self):
        """Executes the command from the input line."""
        command = self.input_line.text().strip()
        if not command:
            return

        if self.current_process and self.current_process.state() == QProcess.ProcessState.Running:
            self.append_error("A command is already running.")
            return

        self.input_line.clear()
        self.append_command(command)

        self.current_process = QProcess()
        self.current_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.current_process.setWorkingDirectory(str(self.working_directory))

        self.current_process.readyReadStandardOutput.connect(self._handle_stdout)
        self.current_process.readyReadStandardError.connect(self._handle_stderr)
        self.current_process.finished.connect(self._on_command_finished)

        if "win" in str(sys.platform).lower():
            self.current_process.start("cmd.exe", ["/C", command])
        else:
            self.current_process.start("/bin/sh", ["-c", command])

    def _handle_stdout(self):
        """Handle standard output from the running process."""
        data = self.current_process.readAllStandardOutput()
        self.append_output(data.data().decode(errors='ignore'))

    def _handle_stderr(self):
        """Handle standard error from the running process."""
        data = self.current_process.readAllStandardError()
        self.append_error(data.data().decode(errors='ignore'))

    def _on_command_finished(self, exit_code, exit_status):
        """Handle the completion of a command."""
        self.append_system_message(f"Process finished with exit code {exit_code}\n")
        self.input_line.setFocus()
        self.command_completed.emit(exit_code)
        self.current_process = None

    def set_force_run_enabled(self, enabled: bool):
        """Enable/disable the run button."""
        self.run_project_btn.setEnabled(enabled)

    # --- Public methods for appending text ---

    def clear_terminal(self):
        """Clears the output area."""
        self.output_area.clear()

    def append_output(self, text: str):
        """Appends standard output text."""
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.output_area.ensureCursorVisible()

    def append_error(self, text: str):
        """Appends error text in a distinct color."""
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        char_format = cursor.charFormat()
        char_format.setForeground(QColor(Colors.ACCENT_RED))
        cursor.setCharFormat(char_format)
        cursor.insertText(text)

        char_format.setForeground(QColor(Colors.TEXT_PRIMARY))
        cursor.setCharFormat(char_format)
        self.output_area.ensureCursorVisible()

    def append_system_message(self, text: str):
        """Appends a system message in a muted color."""
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        char_format = cursor.charFormat()
        char_format.setForeground(QColor(Colors.TEXT_MUTED))
        char_format.setFontItalic(True)
        cursor.setCharFormat(char_format)
        cursor.insertText(text + "\n")

        char_format.setForeground(QColor(Colors.TEXT_PRIMARY))
        char_format.setFontItalic(False)
        cursor.setCharFormat(char_format)
        self.output_area.ensureCursorVisible()

    def append_command(self, command: str):
        """Appends the command that was just run."""
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        prompt_format = cursor.charFormat()
        prompt_format.setForeground(QColor(Colors.ACCENT_BLUE))
        prompt_format.setFontWeight(QFont.Weight.Bold)

        command_format = cursor.charFormat()
        command_format.setForeground(QColor(Colors.TEXT_PRIMARY))
        command_format.setFontWeight(QFont.Weight.Normal)

        cursor.setCharFormat(prompt_format)
        cursor.insertText(self.prompt_label.text() + " ")

        cursor.setCharFormat(command_format)
        cursor.insertText(command + "\n")

        self.output_area.ensureCursorVisible()