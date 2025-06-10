# gui/interactive_terminal.py - Beautiful Terminal with Two Essential Buttons

import sys
import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QProcess, QProcessEnvironment
from PySide6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QHBoxLayout, QLabel

from gui.components import Colors, Typography, ModernButton


class InteractiveTerminal(QWidget):
    """
    Beautiful terminal with essential run functionality
    """
    command_completed = Signal(int)
    force_run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_directory = Path.cwd()
        self.current_project_path = None

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._handle_output)
        self.process.finished.connect(self._on_command_finished)

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Simple button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.run_main_btn = ModernButton("▶️ Run main.py", button_type="primary")
        self.run_main_btn.setMinimumHeight(36)
        self.run_main_btn.clicked.connect(self._run_main_py)

        self.install_deps_btn = ModernButton("📦 Install Requirements", button_type="secondary")
        self.install_deps_btn.setMinimumHeight(36)
        self.install_deps_btn.clicked.connect(self._install_requirements)

        button_layout.addWidget(self.run_main_btn)
        button_layout.addWidget(self.install_deps_btn)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # Beautiful terminal output
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(QFont("Consolas", 11))
        self.output_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.output_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(self.output_area, 1)

        # Command input
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 8, 0, 0)
        input_layout.setSpacing(8)

        self.prompt_label = QLabel()
        self._update_prompt_label()

        self.input_line = QLineEdit()
        self.input_line.setFont(QFont("Consolas", 10))
        self.input_line.setFrame(False)
        self.input_line.returnPressed.connect(self.run_manual_command)

        input_layout.addWidget(self.prompt_label)
        input_layout.addWidget(self.input_line, 1)
        main_layout.addLayout(input_layout)

    def _apply_style(self):
        """Apply beautiful styling to the terminal"""
        self.setStyleSheet(f"""
            InteractiveTerminal {{
                background: {Colors.PRIMARY_BG};
                border-radius: 8px;
            }}
            QTextEdit {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a1a, stop:1 #0f0f0f);
                color: #e0e0e0;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 12px;
                font-family: 'Consolas', 'Courier New', monospace;
                selection-background-color: {Colors.ACCENT_BLUE};
            }}
            QLineEdit {{
                background: #2a2a2a;
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px 8px;
                font-family: 'Consolas', monospace;
            }}
            QLineEdit:focus {{
                border-color: {Colors.ACCENT_BLUE};
                background: #333;
            }}
            QScrollBar:vertical {{
                background: #2a2a2a;
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #555;
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #666;
            }}
        """)

    def _run_main_py(self):
        """Simple: just run python main.py"""
        if not self.current_project_path:
            self.append_error("No project loaded.")
            return

        main_file = self.current_project_path / 'main.py'
        if not main_file.exists():
            self.append_error("main.py not found in project.")
            return

        self.append_system_message(f"Running main.py...")
        self.execute_command(sys.executable, [str(main_file)])

    def _install_requirements(self):
        """Simple: install requirements.txt if it exists"""
        if not self.current_project_path:
            self.append_error("No project loaded.")
            return

        req_file = self.current_project_path / 'requirements.txt'
        if not req_file.exists():
            self.append_error("requirements.txt not found.")
            return

        self.append_system_message("Installing requirements...")
        self.execute_command(sys.executable, ['-m', 'pip', 'install', '-r', str(req_file)])

    def set_working_directory(self, directory: str):
        """Set the working directory"""
        self.working_directory = Path(directory)
        self.current_project_path = Path(directory)
        self._update_prompt_label()

    def _update_prompt_label(self):
        """Update the prompt label"""
        if hasattr(self, 'working_directory'):
            dir_name = self.working_directory.name
            self.prompt_label.setText(f"{dir_name}$ ")
            self.prompt_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            self.prompt_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN}; font-weight: bold;")
        else:
            self.prompt_label.setText("$ ")

    def execute_command(self, program: str, arguments: list):
        """Execute a command"""
        if self.process.state() == QProcess.ProcessState.Running:
            self.append_error("A command is already running.")
            return

        self.process.setWorkingDirectory(str(self.working_directory))
        self.process.setProcessEnvironment(QProcessEnvironment.systemEnvironment())

        command_str = f"{program} {' '.join(arguments)}"
        self.append_command(command_str)

        self.process.start(program, arguments)
        if not self.process.waitForStarted(3000):
            self.append_error(f"Error starting process: {self.process.errorString()}")

    def run_manual_command(self):
        """Execute manually typed commands"""
        command_text = self.input_line.text().strip()
        if not command_text:
            return

        self.input_line.clear()

        if command_text == "clear":
            self.clear_terminal()
            return

        parts = command_text.split()
        program = parts[0]
        args = parts[1:]
        self.execute_command(program, args)

    def _handle_output(self):
        """Handle process output"""
        data = self.process.readAll()
        output = data.data().decode(errors='ignore')
        self.append_output(output)

    def _on_command_finished(self, exit_code, exit_status):
        """Handle command completion"""
        if exit_code == 0:
            self.append_system_message("✅ Command completed")
        else:
            self.append_error(f"❌ Command failed (exit code {exit_code})")

        self.input_line.setFocus()
        self.command_completed.emit(exit_code)

    def clear_terminal(self):
        """Clear the terminal"""
        self.output_area.clear()

    def append_output(self, text: str):
        """Append regular output"""
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.output_area.ensureCursorVisible()

    def append_error(self, text: str):
        """Append error text"""
        self._append_formatted_text(f"❌ {text}\n", Colors.ACCENT_RED)

    def append_system_message(self, text: str):
        """Append system message"""
        self._append_formatted_text(f"ℹ️ {text}\n", Colors.ACCENT_BLUE)

    def append_command(self, command: str):
        """Append command with prompt"""
        self.append_output("\n")
        current_prompt = self.prompt_label.text()
        self._append_formatted_text(f"{current_prompt}", Colors.ACCENT_GREEN)
        self._append_formatted_text(f"{command}\n", Colors.TEXT_PRIMARY)

    def _append_formatted_text(self, text: str, color: str):
        """Append colored text"""
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        char_format = QTextCharFormat()
        char_format.setForeground(QColor(color))

        cursor.mergeCharFormat(char_format)
        cursor.insertText(text)

        self.output_area.setCurrentCharFormat(QTextCharFormat())
        self.output_area.ensureCursorVisible()