# gui/terminals.py - Enhanced with Better Streaming Visual Feedback

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QCheckBox, QProgressBar, QLabel
from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor, QColor
from datetime import datetime
import queue
import threading
import re
import html


class StreamingProgressDisplay:
    """Clean, hierarchical progress display with enhanced code streaming"""

    def __init__(self, terminal_instance):
        self.terminal = terminal_instance
        self.current_stage = None
        self.current_file = None

    def start_workflow(self, prompt: str):
        self.terminal.add_separator("WORKFLOW START")
        self.terminal.stream_log_rich("WorkflowEngine", "status", "🚀 Starting AvA workflow", "0")
        self.terminal.stream_log_rich("WorkflowEngine", "info", f"📋 Request: {prompt[:80]}...", "0")
        self.terminal.stream_log_rich("WorkflowEngine", "info", "", "0")

    def start_stage(self, stage: str, description: str):
        self.current_stage = stage

    def start_file(self, file_path: str):
        self.current_file = file_path

    def stream_progress(self, agent: str, content: str, progress_type: str = "thought", indent_level: int = 1):
        self.terminal.stream_log_rich(agent, progress_type, content, str(indent_level))

    def complete_file(self, file_path: str, status: str):
        status_icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"

    def complete_workflow(self, result: dict):
        self.terminal.stream_log_rich("WorkflowEngine", "info", "", "0")
        success = result.get('success', True)
        status_key = "success" if success else "error"
        icon = "🎉" if success else "❌"
        self.terminal.stream_log_rich("WorkflowEngine", status_key,
                                      f"{icon} Workflow complete: {result.get('file_count', 0)} files.",
                                      "0")


class StreamingTerminal(QWidget):
    workflow_progress = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA Streaming Terminal")
        self.resize(1000, 750)
        self.setStyleSheet("QWidget { background-color: #0d1117; color: #c9d1d9; }")

        # Track streaming state for better visual feedback
        self._last_stream_message_info = None
        self._current_streaming_agent = None
        self._stream_start_time = None
        self._last_chunk_time = None

        self._setup_ui()
        self._add_initial_message()

        # Enhanced progress display
        self.progress_display = StreamingProgressDisplay(self)

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header with enhanced controls
        header_layout = QHBoxLayout()

        # Title
        title_label = QLabel("🤖 AvA Streaming Terminal")
        title_label.setStyleSheet("color: #f0f6fc; font-size: 16px; font-weight: bold; margin-bottom: 5px;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Enhanced auto-scroll checkbox
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet("color: #f0f6fc; margin-right: 10px;")
        header_layout.addWidget(self.auto_scroll_checkbox)

        # Stream status indicator
        self.stream_status = QLabel("● Ready")
        self.stream_status.setStyleSheet("color: #7c3aed; font-weight: bold; margin-right: 10px;")
        header_layout.addWidget(self.stream_status)

        # Control buttons
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_terminal)
        clear_btn.setStyleSheet(
            "QPushButton { background-color: #21262d; color: #f0f6fc; border: 1px solid #30363d; padding: 5px 10px; border-radius: 3px; } QPushButton:hover { background-color: #30363d; }")
        header_layout.addWidget(clear_btn)

        save_btn = QPushButton("Save Log")
        save_btn.clicked.connect(self.save_log)
        save_btn.setStyleSheet(
            "QPushButton { background-color: #21262d; color: #f0f6fc; border: 1px solid #30363d; padding: 5px 10px; border-radius: 3px; } QPushButton:hover { background-color: #30363d; }")
        header_layout.addWidget(save_btn)

        layout.addLayout(header_layout)

        # Main text area with enhanced styling
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        font = QFont("JetBrains Mono", 10)
        font.setStyleHint(QFont.Monospace)
        self.text_area.setFont(font)
        self.text_area.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 10px;
                selection-background-color: #264f78;
            }
        """)
        layout.addWidget(self.text_area)

        # Enhanced progress bar
        self.workflow_progress = WorkflowProgressBar()
        layout.addWidget(self.workflow_progress)

        self.setLayout(layout)

    def _add_initial_message(self):
        welcome_html = f"""
        <div style="text-align: center; margin: 20px 0; padding: 20px; border: 2px solid #30363d; border-radius: 8px; background-color: #161b22;">
            <div style="font-size: 18px; color: #f0f6fc; margin-bottom: 10px;">
                🤖 <strong>AvA Enhanced Streaming Terminal</strong>
            </div>
            <div style="color: #8b949e; margin-bottom: 15px;">
                Real-time AI workflow monitoring with enhanced code streaming
            </div>
            <div style="color: #7c3aed; font-weight: bold;">
                Ready for workflow execution...
            </div>
        </div>
        """
        self.text_area.append(welcome_html)

    def add_separator(self, text: str):
        """Add a visual separator for major workflow phases"""
        separator_html = f"""
        <div style="margin: 15px 0; text-align: center;">
            <div style="border-top: 2px solid #30363d; margin: 10px 0;"></div>
            <span style="background-color: #0d1117; padding: 0 15px; color: #7c3aed; font-weight: bold;">
                {text}
            </span>
            <div style="border-top: 2px solid #30363d; margin: 10px 0;"></div>
        </div>
        """
        self.text_area.append(separator_html)

    @Slot(str, str, str, str)
    def stream_log_rich(self, agent_name: str, type_key: str, content: str, indent_level_str: str):
        """Enhanced streaming with better visual feedback and code highlighting"""
        if not indent_level_str.strip():
            indent_level = 0
        else:
            try:
                indent_level = int(indent_level_str)
            except ValueError:
                indent_level = 0

        # Enhanced agent name mapping for better visual consistency
        agent_display_name = self._normalize_agent_name(agent_name)

        # If the message is not an LLM chunk, reset the tracking for streaming logic
        if type_key != "llm_chunk":
            self._last_stream_message_info = None
            if self._current_streaming_agent:
                self._end_streaming_session()

        cursor = self.text_area.textCursor()
        cursor.movePosition(QTextCursor.End)

        if type_key == "llm_chunk":
            self._handle_streaming_chunk(agent_display_name, content, indent_level_str, cursor)
            return

        # Handle all non-streaming messages with enhanced formatting
        self._handle_regular_message(agent_display_name, type_key, content, indent_level, cursor)

    def _normalize_agent_name(self, agent_name: str) -> str:
        """Normalize agent names for consistent display"""
        name_mapping = {
            "Architect": "Planner",  # Fix the mismatch!
            "Coder": "Coder",
            "Reviewer": "Reviewer",
            "WorkflowEngine": "WorkflowEngine"
        }
        return name_mapping.get(agent_name, agent_name)

    def _handle_streaming_chunk(self, agent_name: str, content: str, indent_level_str: str, cursor):
        """Handle streaming chunks with enhanced visual feedback"""
        current_info = (agent_name, indent_level_str)
        escaped_content = html.escape(content).replace('\n', '<br />')

        # Start new streaming session if needed
        if self._last_stream_message_info != current_info:
            if self._current_streaming_agent:
                self._end_streaming_session()
            self._start_streaming_session(agent_name, indent_level_str)

            # Insert streaming header
            self._insert_streaming_header(agent_name, indent_level_str)

        # Update stream timing
        self._last_chunk_time = datetime.now()

        # Insert the chunk with syntax highlighting for code
        if self._looks_like_code(content):
            cursor.insertHtml(
                f'<span style="color: #79c0ff; font-family: \'JetBrains Mono\', monospace;">{escaped_content}</span>')
        else:
            cursor.insertHtml(f'<span style="color: #98c379;">{escaped_content}</span>')

        # Auto-scroll if enabled
        if self.auto_scroll_checkbox.isChecked():
            self._scroll_to_bottom()

    def _start_streaming_session(self, agent_name: str, indent_level_str: str):
        """Start a new streaming session with visual indicators"""
        self._current_streaming_agent = agent_name
        self._stream_start_time = datetime.now()
        self._last_stream_message_info = (agent_name, indent_level_str)

        # Update status indicator
        self.stream_status.setText(f"🟢 Streaming {agent_name}")
        self.stream_status.setStyleSheet("color: #3fb950; font-weight: bold; margin-right: 10px;")

    def _end_streaming_session(self):
        """End streaming session and update indicators"""
        if self._current_streaming_agent and self._stream_start_time:
            duration = (datetime.now() - self._stream_start_time).total_seconds()

            # Add completion indicator
            cursor = self.text_area.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertHtml(f'<span style="color: #6e7681; font-size: 10px;"> [{duration:.1f}s]</span><br />')

        self._current_streaming_agent = None
        self._stream_start_time = None
        self.stream_status.setText("● Ready")
        self.stream_status.setStyleSheet("color: #7c3aed; font-weight: bold; margin-right: 10px;")

    def _insert_streaming_header(self, agent_name: str, indent_level_str: str):
        """Insert a header for new streaming content"""
        indent_level = int(indent_level_str) if indent_level_str.strip() else 0
        indent_px = indent_level * 20
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Enhanced agent colors
        agent_colors = {
            "Planner": "#DA70D6",
            "Coder": "#6495ED",
            "Reviewer": "#32CD32",
            "WorkflowEngine": "#7c3aed"
        }
        agent_color = agent_colors.get(agent_name, "#6495ED")

        # Enhanced streaming icon based on agent
        agent_icons = {
            "Planner": "🧠",
            "Coder": "⌨️",
            "Reviewer": "🔍",
            "WorkflowEngine": "⚙️"
        }
        agent_icon = agent_icons.get(agent_name, "💬")

        header_html = f"""
        <div style="margin: 1px 0; padding-left: {indent_px}px; font-family: 'JetBrains Mono', monospace;">
            <span style="color: #6e7681; font-size: 10px;">[{timestamp}]</span>
            <span style="color: {agent_color}; font-weight: bold;">{agent_name}</span>
            <span style="color: #8b949e;"> {agent_icon} </span>"""
        self.text_area.append(header_html)

    def _handle_regular_message(self, agent_name: str, type_key: str, content: str, indent_level: int, cursor):
        """Handle non-streaming messages with enhanced formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        indent_px = indent_level * 20

        # Enhanced agent colors
        agent_colors = {
            "Planner": "#DA70D6",
            "Coder": "#6495ED",
            "Reviewer": "#32CD32",
            "WorkflowEngine": "#7c3aed",
            "System": "#B0C4DE",
            "Terminal": "#3fb950"
        }

        # Enhanced type icons and colors
        type_config = {
            "thought": {"icon": "💡", "color": "#f7cc2e"},
            "thought_detail": {"icon": "💭", "color": "#8b949e"},
            "success": {"icon": "✅", "color": "#3fb950"},
            "error": {"icon": "❌", "color": "#f85149"},
            "warning": {"icon": "⚠️", "color": "#d29922"},
            "info": {"icon": "ℹ️", "color": "#58a6ff"},
            "debug": {"icon": "🐛", "color": "#6e7681"},
            "security": {"icon": "🛡️", "color": "#f85149"},
            "performance": {"icon": "⚡", "color": "#d29922"},
            "stage_start": {"icon": "▶️", "color": "#7c3aed"},
            "file_op": {"icon": "📄", "color": "#58a6ff"}
        }

        type_info = type_config.get(type_key, {"icon": "📝", "color": "#c9d1d9"})
        agent_color = agent_colors.get(agent_name, "#6495ED")

        escaped_content = html.escape(content).replace('\n', '<br />')

        message_html = f"""
        <div style="margin: 2px 0; padding-left: {indent_px}px; font-family: 'JetBrains Mono', monospace;">
            <span style="color: #6e7681; font-size: 10px;">[{timestamp}]</span>
            <span style="color: {agent_color}; font-weight: bold;">{agent_name}</span>
            <span style="color: {type_info['color']}; margin: 0 5px;">{type_info['icon']}</span>
            <span style="color: {type_info['color']};">{escaped_content}</span>
        </div>
        """

        self.text_area.append(message_html)

        if self.auto_scroll_checkbox.isChecked():
            self._scroll_to_bottom()

    def _looks_like_code(self, content: str) -> bool:
        """Detect if content looks like code for syntax highlighting"""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
            'try:', 'except:', 'with ', 'async def', '    ', '\t'
        ]
        return any(indicator in content for indicator in code_indicators)

    def _scroll_to_bottom(self):
        """Scroll to bottom of terminal"""
        scrollbar = self.text_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot()
    def clear_terminal(self):
        """Clear terminal and reset state"""
        self.text_area.clear()
        self._add_initial_message()
        self.workflow_progress.reset_progress()
        self._end_streaming_session()

    @Slot()
    def save_log(self):
        """Save terminal log to file"""
        try:
            from PySide6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Log",
                f"ava_term_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "HTML Files (*.html);;Text Files (*.txt)"
            )
            if filename:
                content = self.text_area.toHtml() if filename.endswith(".html") else self.text_area.toPlainText()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.stream_log_rich("Terminal", "success", f"Log saved: {filename}", "0")
        except Exception as e:
            self.stream_log_rich("Terminal", "error", f"Save failed: {e}", "0")


class WorkflowProgressBar(QWidget):
    """Enhanced progress bar for workflow stages"""

    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.reset_progress()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #30363d;
                border-radius: 3px;
                background-color: #21262d;
                color: #f0f6fc;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #7c3aed;
                border-radius: 2px;
            }
        """)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #8b949e; font-size: 11px; margin-top: 2px;")

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def update_progress(self, stage: str, description: str, progress_percent: int = None):
        """Update progress bar with current stage"""
        stage_mapping = {
            "planning": {"text": "🧠 Planning", "value": 20},
            "coding": {"text": "⌨️ Coding", "value": 60},
            "review": {"text": "🔍 Review", "value": 90},
            "complete": {"text": "✅ Complete", "value": 100}
        }

        stage_info = stage_mapping.get(stage, {"text": f"⚙️ {stage.title()}", "value": 50})

        if progress_percent is not None:
            self.progress_bar.setValue(progress_percent)
        else:
            self.progress_bar.setValue(stage_info["value"])

        self.progress_bar.setFormat(stage_info["text"])
        self.status_label.setText(description)

    def reset_progress(self):
        """Reset progress bar to initial state"""
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        self.status_label.setText("Waiting for workflow...")


class TerminalWindow(StreamingTerminal):
    """Main terminal window with enhanced features"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA Terminal - Enhanced Streaming & Code Generation")
        self.resize(1200, 800)
        self.setStyleSheet("QWidget { background-color: #0d1117; color: #c9d1d9; }")