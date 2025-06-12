# gui/interactive_terminal.py - V3 with Command History AND Tab Completion!

import sys
import os
import logging
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QProcess, QProcessEnvironment
from PySide6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat, QKeyEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QHBoxLayout, QLabel

from gui.components import Colors, Typography, ModernButton


# --- NEW: Custom LineEdit to handle special key presses ---
class TerminalInput(QLineEdit):
    """A QLineEdit that handles up/down arrow keys for history and Tab for completion."""
    up_arrow_pressed = Signal()
    down_arrow_pressed = Signal()
    tab_pressed = Signal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Up:
            self.up_arrow_pressed.emit()
            return
        if event.key() == Qt.Key.Key_Down:
            self.down_arrow_pressed.emit()
            return
        if event.key() == Qt.Key.Key_Tab:
            self.tab_pressed.emit()
            return
        # For all other keys, use the default behavior
        super().keyPressEvent(event)


class InteractiveTerminal(QWidget):
    """
    Beautiful terminal with run functionality, command history, and tab completion.
    """
    command_completed = Signal(int)
    force_run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.working_directory = Path.cwd()
        self.current_project_path = None
        self.logger = logging.getLogger(__name__)

        # --- Command History attributes ---
        self.command_history: list[str] = []
        self.history_index = -1

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

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        self.run_main_btn = ModernButton("▶️ Run main.py", button_type="primary")
        self.run_main_btn.clicked.connect(self._run_main_py)
        self.install_deps_btn = ModernButton("📦 Install Requirements", button_type="secondary")
        self.install_deps_btn.clicked.connect(self._install_requirements)
        button_layout.addWidget(self.run_main_btn)
        button_layout.addWidget(self.install_deps_btn)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(QFont("Consolas", 11))
        main_layout.addWidget(self.output_area, 1)

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 8, 0, 0)
        input_layout.setSpacing(8)

        self.prompt_label = QLabel()
        self._update_prompt_label()

        self.input_line = TerminalInput()
        self.input_line.setFont(QFont("Consolas", 10))
        self.input_line.setFrame(False)
        self.input_line.returnPressed.connect(self.run_manual_command)
        self.input_line.up_arrow_pressed.connect(self._show_previous_history)
        self.input_line.down_arrow_pressed.connect(self._show_next_history)
        self.input_line.tab_pressed.connect(self._handle_tab_completion)

        input_layout.addWidget(self.prompt_label)
        input_layout.addWidget(self.input_line, 1)
        main_layout.addLayout(input_layout)

    def _apply_style(self):
        self.setStyleSheet(f"""
            InteractiveTerminal {{ background: {Colors.PRIMARY_BG}; border-radius: 8px; }}
            QTextEdit {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a1a1a, stop:1 #0f0f0f);
                color: #e0e0e0; border: 1px solid #333; border-radius: 6px; padding: 12px;
                font-family: 'Consolas', 'Courier New', monospace; selection-background-color: {Colors.ACCENT_BLUE};
            }}
            QLineEdit {{ background: #2a2a2a; color: {Colors.TEXT_PRIMARY}; border: 1px solid #404040;
                        border-radius: 4px; padding: 6px 8px; font-family: 'Consolas', monospace; }}
            QLineEdit:focus {{ border-color: {Colors.ACCENT_BLUE}; background: #333; }}
        """)

    def _run_main_py(self):
        if not self.current_project_path: return self.append_error("No project loaded.")
        main_file = self.current_project_path / 'main.py'
        if not main_file.exists(): return self.append_error("main.py not found.")
        venv_path = self.current_project_path / 'venv'
        if not venv_path.is_dir(): return self.append_error(
            "Project 'venv' not found. Please click 'Install Requirements'.")
        python_exe = venv_path / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not python_exe.exists(): return self.append_error(f"Python not found in venv: {python_exe}")

        command = f'"{python_exe}" "{main_file}"'
        self._add_to_history(command)
        self.execute_command(str(python_exe), [str(main_file)])

    def _install_requirements(self):
        if not self.current_project_path: return self.append_error("No project loaded.")
        req_file = self.current_project_path / 'requirements.txt'
        if not req_file.exists(): return self.append_error("requirements.txt not found.")
        venv_path = self.current_project_path / 'venv'
        if not venv_path.is_dir(): return self.append_error("Project 'venv' not found.")
        pip_exe = venv_path / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")
        if not pip_exe.exists(): return self.append_error(f"Pip not found in venv: {pip_exe}")

        command = f'"{pip_exe}" install -r "{req_file}"'
        self._add_to_history(command)
        self.execute_command(str(pip_exe), ['install', '-r', str(req_file)])

    def set_working_directory(self, directory: str):
        self.working_directory = Path(directory)
        self.current_project_path = Path(directory)
        self._update_prompt_label()

    def _update_prompt_label(self):
        dir_name = self.working_directory.name if self.working_directory else ""
        self.prompt_label.setText(f"{dir_name}$ ")
        self.prompt_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.prompt_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN};")

    def execute_command(self, program: str, arguments: list):
        if self.process.state() == QProcess.ProcessState.Running:
            return self.append_error("A command is already running.")

        self.process.setWorkingDirectory(str(self.working_directory))
        self.process.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        command_str = f"{Path(program).name} {' '.join(arguments)}"
        self.append_command(command_str)
        self.process.start(program, arguments)

    def run_manual_command(self):
        command_text = self.input_line.text().strip()
        if not command_text: return
        self._add_to_history(command_text)
        self.input_line.clear()
        if command_text == "clear": return self.clear_terminal()

        # Basic command parsing
        import shlex
        try:
            parts = shlex.split(command_text)
        except ValueError:
            self.append_error("Error: Mismatched quotes in command.")
            return

        if not parts: return
        self.execute_command(parts[0], parts[1:])

    def _add_to_history(self, command: str):
        if command and (not self.command_history or self.command_history[0] != command):
            self.command_history.insert(0, command)
        self.history_index = -1

    def _show_previous_history(self):
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.input_line.setText(self.command_history[self.history_index])
            self.input_line.end(False)  # Move cursor to end

    def _show_next_history(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.input_line.setText(self.command_history[self.history_index])
            self.input_line.end(False)
        else:
            self.history_index = -1
            self.input_line.clear()

    def _handle_tab_completion(self):
        text_under_cursor = self.input_line.text()
        parts = text_under_cursor.split()
        if not parts: return

        to_complete = parts[-1]
        try:
            path_obj = Path(to_complete)
            # Determine search directory and partial name
            if to_complete.endswith(os.sep):
                search_dir = self.working_directory / path_obj
                partial_name = ""
            else:
                search_dir = self.working_directory / path_obj.parent
                partial_name = path_obj.name

            if not search_dir.is_dir(): return

            matches = [p.name for p in search_dir.iterdir() if p.name.startswith(partial_name)]

            if not matches: return

            if len(matches) == 1:
                completed_name = matches[0]
                if (search_dir / completed_name).is_dir():
                    completed_name += os.sep

                # Reconstruct the full path for the part we're completing
                full_completion = (path_obj.parent / completed_name).as_posix()
                parts[-1] = full_completion
                self.input_line.setText(" ".join(parts))
            else:
                prefix = os.path.commonprefix(matches)
                if len(prefix) > len(partial_name):
                    full_completion = (path_obj.parent / prefix).as_posix()
                    parts[-1] = full_completion
                    self.input_line.setText(" ".join(parts))

                self.append_output("\n" + "    ".join(matches) + "\n")
                self.append_command(self.input_line.text())

        except Exception as e:
            self.logger.error(f"Tab completion failed: {e}")

    def _handle_output(self):
        output = self.process.readAll().data().decode(errors='ignore')
        self.append_output(output)

    def _on_command_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.append_system_message("✅ Command completed")
        else:
            self.append_error(f"❌ Command failed (exit code {exit_code})")
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
        self._append_formatted_text(f"❌ {text}\n", Colors.ACCENT_RED)

    def append_system_message(self, text: str):
        self._append_formatted_text(f"ℹ️ {text}\n", Colors.ACCENT_BLUE)

    def append_command(self, command: str):
        self.append_output("\n")
        self._append_formatted_text(f"{self.prompt_label.text()}", Colors.ACCENT_GREEN)
        self._append_formatted_text(f"{command}", Colors.TEXT_PRIMARY)
        self.output_area.ensureCursorVisible()

    def _append_formatted_text(self, text: str, color: str):
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        char_format = QTextCharFormat()
        char_format.setForeground(QColor(color))
        cursor.mergeCharFormat(char_format)
        cursor.insertText(text)
        self.output_area.setCurrentCharFormat(QTextCharFormat())