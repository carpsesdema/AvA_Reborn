from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QCheckBox, QProgressBar, QLabel
from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor, QColor  # Added QColor
from datetime import datetime
import queue  # Keep for now, though direct slot connection is primary
import threading
import re
import html  # For escaping HTML in code


class StreamingProgressDisplay:
    """Clean, hierarchical progress display like aider"""

    def __init__(self, terminal_instance):  # Changed to accept terminal_instance
        self.terminal = terminal_instance  # Store the terminal instance
        self.current_stage = None
        self.current_file = None
        # Indent level is now managed by the stream_log_rich's indent_level argument

    def start_workflow(self, prompt: str):
        self.terminal.add_separator("WORKFLOW START")
        self.terminal.stream_log_rich("WorkflowEngine", "status", "🚀 Starting AvA workflow", "0")
        self.terminal.stream_log_rich("WorkflowEngine", "info", f"📋 Request: {prompt[:80]}...", "0")
        self.terminal.stream_log_rich("WorkflowEngine", "info", "", "0")  # Spacer

    def start_stage(self, stage: str, description: str):
        self.current_stage = stage
        # The engine will emit this directly using detailed_log_event -> stream_log_rich
        # Example: self.terminal.stream_log_rich("WorkflowEngine", "stage_update", f"▶ {stage.upper()}: {description}", 0)

    def start_file(self, file_path: str):
        self.current_file = file_path
        # Example: self.terminal.stream_log_rich("WorkflowEngine", "file_op", f"📄 Processing: {file_path}", 1)

    def stream_progress(self, agent: str, content: str, progress_type: str = "thought", indent_level: int = 1):
        # This method is now largely superseded by direct calls to stream_log_rich from services/engine
        self.terminal.stream_log_rich(agent, progress_type, content, str(indent_level))

    def complete_file(self, file_path: str, status: str):  # status is "success", "warning", "error"
        status_icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
        # Example: self.terminal.stream_log_rich("WorkflowEngine", status, f"{status_icon} File '{file_path}' processed.", 1)

    def complete_workflow(self, result: dict):
        self.terminal.stream_log_rich("WorkflowEngine", "info", "", "0")  # Spacer
        success = result.get('success', True)
        status_key = "success" if success else "error"
        icon = "🎉" if success else "❌"
        self.terminal.stream_log_rich("WorkflowEngine", status_key,
                                      f"{icon} Workflow complete: {result.get('file_count', 0)} files.", "0")
        self.terminal.stream_log_rich("WorkflowEngine", "info", f"📁 Project: {result.get('project_dir', 'N/A')}", "0")


class WorkflowProgressWidget(QWidget):
    """Visual workflow progress indicator"""

    def __init__(self):
        super().__init__()
        self._init_ui()
        self.reset_progress()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.stage_label = QLabel("Workflow: Idle")
        self.stage_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(self.stage_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #2d2d30; border: none; border-radius: 3px; } QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d7ff, stop:1 #0078d4); border-radius: 3px; }")
        layout.addWidget(self.progress_bar)
        self.task_label = QLabel("Tasks: 0/0")
        self.task_label.setFont(QFont("Segoe UI", 9))
        self.task_label.setStyleSheet("color: #cccccc; background: transparent;")
        layout.addWidget(self.task_label)
        self.setLayout(layout)
        self.setMaximumHeight(60)

    def update_stage(self, stage: str, description: str):
        stage_colors = {"idle": "#888", "initializing": "#ffb900", "context_discovery": "#a5a5f0",
                        "planning": "#a5a5f0", "generation": "#00d7ff", "finalization": "#3fb950",
                        "complete": "#3fb950", "error": "#f85149"}
        stage_emojis = {"idle": "⏸️", "initializing": "🚀", "context_discovery": "🔍", "planning": "🧠", "generation": "⚡",
                        "finalization": "📄", "complete": "✅", "error": "❌"}
        color = stage_colors.get(stage, "#ccc")
        emoji = stage_emojis.get(stage, "ℹ️")
        self.stage_label.setText(f"{emoji} {description}")
        self.stage_label.setStyleSheet(f"color: {color}; background: transparent; font-weight: bold;")

    def update_progress(self, completed: int, total: int):
        if total > 0:
            progress = int((completed / total) * 100)
            self.progress_bar.setValue(progress)
            self.task_label.setText(
                f"Tasks: {completed}/{total} ({progress}%)")
        else:
            self.progress_bar.setValue(0)
            self.task_label.setText("Tasks: 0/0")

    def reset_progress(self):
        self.update_stage("idle", "Workflow: Idle")
        self.update_progress(0, 0)


class StreamingIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(120, 20)
        self.is_streaming = False
        self.animation_step = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.setInterval(300)
        self.label = QLabel("● Ready")
        self.label.setStyleSheet("color: #3fb950; font-size: 11px; font-weight: bold;")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def start_streaming(self, message: str = "Streaming"):
        self.is_streaming = True
        self.animation_step = 0
        self.base_message = message
        self.label.setStyleSheet("color: #00d7ff; font-size: 11px; font-weight: bold;")
        self.timer.start()

    def stop_streaming(self, message: str = "● Ready"):
        self.is_streaming = False
        self.timer.stop()
        self.label.setText(message)
        self.label.setStyleSheet("color: #3fb950; font-size: 11px; font-weight: bold;")

    def _animate(self):
        if not self.is_streaming: return
        dots = ["⚡", "⚡.", "⚡..", "⚡..."]
        dot = dots[self.animation_step % len(dots)]
        self.label.setText(f"{dot} {self.base_message}")
        self.animation_step += 1


class StreamingTerminal(QWidget):
    def __init__(self):
        super().__init__()
        self._last_stream_message_info = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.workflow_progress = WorkflowProgressWidget()
        layout.addWidget(self.workflow_progress)
        controls_layout = QHBoxLayout()
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet("color: #c6c6c6;")
        self.streaming_indicator = StreamingIndicator()
        self.cache_status = QLabel("Cache: 0")
        self.cache_status.setStyleSheet("color: #888; font-size: 11px;")
        self.perf_status = QLabel("Perf: Ready")
        self.perf_status.setStyleSheet("color: #888; font-size: 11px;")
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_terminal)
        btn_style = "QPushButton { background-color: #2d2d2d; color: #fff; border: 1px solid #3e3e3e; border-radius: 4px; padding: 4px 8px; font-size: 12px; } QPushButton:hover { background-color: #3e3e42; }"
        self.clear_button.setStyleSheet(btn_style)
        self.save_button = QPushButton("Save Log")
        self.save_button.clicked.connect(self.save_log)
        self.save_button.setStyleSheet(btn_style)
        controls_layout.addWidget(self.auto_scroll_checkbox)
        controls_layout.addWidget(self.streaming_indicator)
        controls_layout.addWidget(self.cache_status)
        controls_layout.addWidget(self.perf_status)
        controls_layout.addStretch()
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.save_button)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet(
            "QTextEdit { background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 13px; line-height: 1.5;} QScrollBar:vertical { background: #21262d; width: 8px; border-radius: 4px;} QScrollBar::handle:vertical { background: #484f58; border-radius: 4px; min-height: 20px;} QScrollBar::handle:vertical:hover { background: #58a6ff;}")
        font = QFont("JetBrains Mono", 12)
        font.setStyleHint(QFont.Monospace)
        self.text_area.setFont(font)
        layout.addLayout(controls_layout)
        layout.addWidget(self.text_area)
        self.setLayout(layout)
        self._add_initial_message()

    def _add_initial_message(self):
        self.text_area.append(
            "<div style='color: #7c3aed; font-weight: bold; margin-bottom: 8px;'>╭─────────────────── AvA Development Terminal ───────────────────╮<br/>│               Real-time streaming & progress               │<br/>│                    Ready for workflows! 🚀                │<br/>╰────────────────────────────────────────────────────────────╯</div>")

    @Slot(str, str, str, str)
    def stream_log_rich(self, agent_name: str, type_key: str, content: str, indent_level_str: str):
        if not indent_level_str.strip():
            indent_level = 0
        else:
            try:
                indent_level = int(indent_level_str)
            except ValueError:
                indent_level = 0

        # If the message is not an LLM chunk, reset the tracking for our streaming logic
        if type_key != "llm_chunk":
            self._last_stream_message_info = None

        cursor = self.text_area.textCursor()
        cursor.movePosition(QTextCursor.End)

        if type_key == "llm_chunk":
            current_info = (agent_name, indent_level_str)
            escaped_content = html.escape(content).replace('\n', '<br />')

            # Check if this chunk belongs to the previous message
            if self._last_stream_message_info == current_info:
                # If so, just insert the text without any header
                cursor.insertHtml(f'<span style="color: #98c379;">{escaped_content}</span>')
            else:
                # Otherwise, it's a new stream. Print a header first.
                self._last_stream_message_info = current_info
                indent_px = indent_level * 20
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                agent_colors = {"Planner": "#DA70D6", "Coder": "#6495ED"}
                agent_color = agent_colors.get(agent_name, "#6495ED")
                # Insert the header, which also moves the cursor
                header_html = f"""
                <div style="margin: 1px 0; padding-left: {indent_px}px; font-family: 'JetBrains Mono', monospace;">
                    <span style="color: #6e7681; font-size: 10px;">[{timestamp}]</span>
                    <span style="color: {agent_color}; font-weight: bold;">{agent_name}</span>
                    <span style="color: #8b949e;"> 💬 </span>"""
                self.text_area.append(header_html)  # append adds a newline, which separates headers
                # Now insert the first chunk of text without an extra newline
                cursor.insertHtml(f'<span style="color: #98c379;">{escaped_content}</span>')

            if self.auto_scroll_checkbox.isChecked():
                self._scroll_to_bottom()
            return

        # --- This part handles all non-streaming messages ---
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        indent_px = indent_level * 20

        agent_colors = {"Planner": "#DA70D6", "Coder": "#6495ED", "Assembler": "#FFA500", "Reviewer": "#32CD32",
                        "WorkflowEngine": "#7c3aed", "Orchestrator": "#FF69B4", "System": "#B0C4DE",
                        "Terminal": "#3fb950"}
        type_icons = {"thought": "💡", "thought_detail": "💭", "code_chunk": "```", "status": "ℹ️", "error": "❌",
                      "warning": "⚠️", "success": "✅", "file_op": "📁", "stage_start": "▶️", "stage_end": "⏹️",
                      "fallback": "🔄", "debug": "🐞", "info": "🔹"}

        agent_color = agent_colors.get(agent_name, "#c9d1d9")
        icon = type_icons.get(type_key, "")
        escaped_content = html.escape(content)

        if type_key == "code_chunk":
            formatted_message = f"""
            <div style="margin: 1px 0; padding-left: {indent_px}px; font-family: 'JetBrains Mono', monospace;">
                <span style="color: #6e7681; font-size: 10px;">[{timestamp}]</span>
                <span style="color: {agent_color}; opacity: 0.8;"> {agent_name} {icon}</span>
                <div style="background:#161b22; color:#bae67e; padding: 5px 8px; border-radius:4px; margin-top: 2px; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere;">{escaped_content}</div>
            </div>
            """
        else:
            content_color_map = {"error": "#f85149", "warning": "#d29922", "success": "#3fb950",
                                 "stage_start": agent_color, "stage_end": agent_color}
            content_color = content_color_map.get(type_key, "#c9d1d9")
            font_weight = "bold" if type_key in ["stage_start", "stage_end", "success", "error"] else "normal"
            formatted_message = f"""
            <div style="margin: 1px 0; padding-left: {indent_px}px; font-family: 'JetBrains Mono', monospace;">
                <span style="color: #6e7681; font-size: 10px;">[{timestamp}]</span>
                <span style="color: {agent_color}; font-weight: bold;">{agent_name}</span>
                <span style="color: #8b949e;"> {icon} </span>
                <span style="color: {content_color}; font-weight: {font_weight};">{escaped_content}</span>
            </div>
            """

        self.text_area.append(formatted_message)
        if self.auto_scroll_checkbox.isChecked(): self._scroll_to_bottom()

    @Slot(str, int)
    def stream_log(self, message: str, indent: int = 0):
        self.stream_log_rich("System", "info", message, str(indent))

    def _scroll_to_bottom(self):
        cursor = self.text_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_area.setTextCursor(cursor)

    def add_separator(self, title: str = ""):
        separator_html = f"<div style='color: #484f58; text-align: center; margin: 5px 0;'>{html.escape('─' * 15)} {html.escape(title)} {html.escape('─' * 15) if title else html.escape('─' * 35)}</div>"
        self.text_area.append(separator_html)

    @Slot(str, str)
    def update_workflow_progress(self, stage: str, description: str):
        self.workflow_progress.update_stage(stage, description)

    @Slot(int, int)
    def update_task_progress(self, completed: int, total: int):
        self.workflow_progress.update_progress(completed, total)

    def update_cache_status(self, cache_size: int, hit_rate: float):
        self.cache_status.setText(f"Cache: {cache_size} ({hit_rate:.1%})")
        color = "#3fb950" if hit_rate >= 0.5 else "#ffb900" if hit_rate >= 0.2 else "#888"
        self.cache_status.setStyleSheet(f"color: {color}; font-size: 11px;")

    def update_performance_stats(self, speed_info: str):
        self.perf_status.setText(f"Perf: {speed_info}")

    def start_file_generation(self, file_path: str):
        from pathlib import Path
        self.streaming_indicator.start_streaming(f"Gen: {Path(file_path).name}")

    def complete_file_generation(self, file_path: str, success: bool):
        self.streaming_indicator.stop_streaming("● Ready")

    @Slot()
    def clear_terminal(self):
        self.text_area.clear()
        self._add_initial_message()
        self.workflow_progress.reset_progress()
        self.streaming_indicator.stop_streaming()

    @Slot()
    def save_log(self):
        try:
            from PySide6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(self, "Save Log",
                                                      f"ava_term_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                                      "HTML Files (*.html);;Text Files (*.txt)")
            if filename:
                content_to_save = self.text_area.toHtml() if filename.endswith(
                    ".html") else self.text_area.toPlainText()
                with open(filename, 'w', encoding='utf-8') as f: f.write(content_to_save)
                self.stream_log_rich("Terminal", "success", f"Log saved: {filename}", "0")
        except Exception as e:
            self.stream_log_rich("Terminal", "error", f"Save log failed: {e}", "0")


class TerminalWindow(StreamingTerminal):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA Terminal - Enhanced Streaming & Progress")
        self.resize(1000, 750)
        self.setStyleSheet("QWidget { background-color: #0d1117; color: #c9d1d9; }")