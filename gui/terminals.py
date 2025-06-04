from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QCheckBox, QProgressBar, QLabel
from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from datetime import datetime
import queue
import threading
import re


class StreamingProgressDisplay:
    """Clean, hierarchical progress display like aider"""

    def __init__(self, terminal):
        self.terminal = terminal
        self.current_stage = None
        self.current_file = None
        self.indent_level = 0
        self.active_streams = {}

    def start_workflow(self, prompt: str):
        self.terminal.add_separator("WORKFLOW START")
        self.terminal.log("🚀 Starting AvA workflow")
        self.terminal.log(f"📋 Request: {prompt[:80]}...")
        self.terminal.log("")

    def start_stage(self, stage: str, description: str):
        self.current_stage = stage
        self.terminal.log(f"▶ {stage.upper()}: {description}")
        self.indent_level = 1

    def start_file(self, file_path: str):
        self.current_file = file_path
        self.terminal.log(f"  📄 {file_path}", indent=1)
        self.indent_level = 2

    def stream_progress(self, content: str, progress_type: str = "code"):
        """Stream with appropriate visual indicators"""
        if progress_type == "planning":
            self.terminal.stream_log(f"    🧠 {content}", indent=2)
        elif progress_type == "coding":
            self.terminal.stream_log(f"    ⚡ {content}", indent=2)
        elif progress_type == "review":
            self.terminal.stream_log(f"    👁️ {content}", indent=2)
        else:
            self.terminal.stream_log(f"    {content}", indent=2)

    def complete_file(self, file_path: str, status: str):
        status_icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
        self.terminal.log(f"  {status_icon} {file_path} - {status}")

    def complete_workflow(self, result: dict):
        self.terminal.log("")
        self.terminal.log(f"🎉 Workflow complete: {result['file_count']} files generated")
        self.terminal.log(f"📁 Project: {result['project_dir']}")


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

        # Stage indicator
        self.stage_label = QLabel("Workflow: Idle")
        self.stage_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.stage_label.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(self.stage_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: #2d2d30;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Task counter
        self.task_label = QLabel("Tasks: 0/0")
        self.task_label.setFont(QFont("Segoe UI", 9))
        self.task_label.setStyleSheet("color: #cccccc; background: transparent;")
        layout.addWidget(self.task_label)

        self.setLayout(layout)
        self.setMaximumHeight(60)

    def update_stage(self, stage: str, description: str):
        """Update workflow stage"""
        stage_colors = {
            "idle": "#888888",
            "initializing": "#ffb900",
            "context_discovery": "#a5a5f0",
            "planning": "#a5a5f0",
            "generation": "#00d7ff",
            "finalization": "#3fb950",
            "complete": "#3fb950",
            "error": "#f85149"
        }

        stage_emojis = {
            "idle": "⏸️",
            "initializing": "🚀",
            "context_discovery": "🔍",
            "planning": "🧠",
            "generation": "⚡",
            "finalization": "📄",
            "complete": "✅",
            "error": "❌"
        }

        color = stage_colors.get(stage, "#cccccc")
        emoji = stage_emojis.get(stage, "ℹ️")

        self.stage_label.setText(f"{emoji} {description}")
        self.stage_label.setStyleSheet(f"color: {color}; background: transparent; font-weight: bold;")

    def update_progress(self, completed: int, total: int):
        """Update task progress"""
        if total > 0:
            progress = int((completed / total) * 100)
            self.progress_bar.setValue(progress)
            self.task_label.setText(f"Tasks: {completed}/{total} ({progress}%)")
        else:
            self.progress_bar.setValue(0)
            self.task_label.setText("Tasks: 0/0")

    def reset_progress(self):
        """Reset progress to idle state"""
        self.update_stage("idle", "Workflow: Idle")
        self.update_progress(0, 0)


class StreamingIndicator(QWidget):
    """Animated indicator for streaming status"""

    def __init__(self):
        super().__init__()
        self.setFixedSize(120, 20)
        self.is_streaming = False
        self.animation_step = 0

        # Animation timer
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
        """Start streaming animation"""
        self.is_streaming = True
        self.animation_step = 0
        self.base_message = message
        self.label.setStyleSheet("color: #00d7ff; font-size: 11px; font-weight: bold;")
        self.timer.start()

    def stop_streaming(self, message: str = "● Ready"):
        """Stop streaming animation"""
        self.is_streaming = False
        self.timer.stop()
        self.label.setText(message)
        self.label.setStyleSheet("color: #3fb950; font-size: 11px; font-weight: bold;")

    def _animate(self):
        """Animate streaming indicator"""
        if not self.is_streaming:
            return

        dots = ["⚡", "⚡.", "⚡..", "⚡..."]
        dot = dots[self.animation_step % len(dots)]
        self.label.setText(f"{dot} {self.base_message}")
        self.animation_step += 1


class StreamingTerminal(QWidget):
    """
    🎯 Aider-Style Terminal with Real-time Streaming & Progress Visualization

    Features:
    - Clean hierarchical progress display
    - Real-time code streaming
    - Visual workflow tracking
    - Professional layout like aider
    """

    def __init__(self):
        super().__init__()
        self.message_queue = queue.Queue()
        self.stream_buffer = ""
        self.current_indent = 0
        self._init_ui()
        self._setup_timer()

        # Enhanced progress display
        self.progress_display = StreamingProgressDisplay(self)

    def _init_ui(self):
        layout = QVBoxLayout()

        # Workflow progress section
        self.workflow_progress = WorkflowProgressWidget()
        layout.addWidget(self.workflow_progress)

        # Control panel
        controls_layout = QHBoxLayout()

        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet("color: #c6c6c6;")

        # Streaming indicator
        self.streaming_indicator = StreamingIndicator()

        # Cache status
        self.cache_status = QLabel("Cache: 0 items")
        self.cache_status.setStyleSheet("color: #888; font-size: 11px;")

        # Performance stats
        self.perf_status = QLabel("Speed: Ready")
        self.perf_status.setStyleSheet("color: #888; font-size: 11px;")

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_terminal)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d; color: #ffffff;
                border: 1px solid #3e3e3e; border-radius: 4px;
                padding: 4px 8px; font-size: 12px;
            }
            QPushButton:hover { background-color: #3e3e3e; }
        """)

        self.save_button = QPushButton("Save Log")
        self.save_button.clicked.connect(self.save_log)
        self.save_button.setStyleSheet(self.clear_button.styleSheet())

        controls_layout.addWidget(self.auto_scroll_checkbox)
        controls_layout.addWidget(self.streaming_indicator)
        controls_layout.addWidget(self.cache_status)
        controls_layout.addWidget(self.perf_status)
        controls_layout.addStretch()
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.save_button)

        # Terminal display with enhanced styling
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 12px;
                font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                line-height: 1.5;
            }
            QScrollBar:vertical {
                background: #21262d;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #484f58;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #58a6ff;
            }
        """)

        # Set monospace font
        font = QFont("JetBrains Mono", 12)
        font.setStyleHint(QFont.Monospace)
        self.text_area.setFont(font)

        layout.addLayout(controls_layout)
        layout.addWidget(self.text_area)
        self.setLayout(layout)

        self._add_initial_message()

    def _setup_timer(self):
        """Setup timer for smooth message processing"""
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_queue)
        self.timer.start(30)  # 30ms for smooth streaming

    def _add_initial_message(self):
        """Add clean welcome message"""
        welcome_msg = """
<div style="color: #7c3aed; font-weight: bold; margin-bottom: 8px;">
╭────────────────────────────────────────────────────────────╮
│                    AvA Development Terminal                │
│               Real-time streaming & progress               │
│                    Ready for workflows! 🚀                │
╰────────────────────────────────────────────────────────────╯
</div>
        """.strip()
        self.text_area.append(welcome_msg)

    def log(self, message: str, log_level: str = "info", indent: int = 0):
        """
        🎯 Main logging method - clean like aider
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.message_queue.put({
            'timestamp': timestamp,
            'text': message,
            'type': log_level,
            'indent': indent
        })

    def stream_log(self, message: str, indent: int = 0, update_last: bool = False):
        """
        📺 Stream content with live updates (like aider's streaming)
        """
        if update_last:
            # Update the last line (for progress updates)
            self.message_queue.put({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'text': message,
                'type': 'stream_update',
                'indent': indent,
                'update_last': True
            })
        else:
            # Add new streaming line
            self.message_queue.put({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'text': message,
                'type': 'stream',
                'indent': indent
            })

    def _process_queue(self):
        """Process queued messages with enhanced streaming"""
        messages_processed = 0
        max_messages = 20  # Higher for smooth streaming

        while not self.message_queue.empty() and messages_processed < max_messages:
            try:
                message = self.message_queue.get_nowait()
                self._display_message(message)
                messages_processed += 1
            except queue.Empty:
                break

    def _display_message(self, message: dict):
        """Display message with aider-style formatting"""
        timestamp = message['timestamp']
        text = message['text']
        msg_type = message['type']
        indent = message.get('indent', 0)

        # Clean indentation (like aider)
        indent_str = "  " * indent

        # Get color and icon
        color = self._get_message_color(text, msg_type)
        icon = self._get_message_icon(text, msg_type)

        # Special handling for different message types
        if msg_type == 'stream':
            # Streaming messages - minimal formatting
            formatted_message = f"""
            <div style="margin: 1px 0; font-family: monospace;">
                <span style="color: #6e7681; font-size: 10px;">[{timestamp}]</span>
                <span style="color: {color};">{indent_str}{icon} {self._escape_html(text)}</span>
            </div>
            """
        elif msg_type == 'stream_update':
            # Update last line (for progress bars, etc.)
            # For now, just add as new line - more complex cursor manipulation could be added
            formatted_message = f"""
            <div style="margin: 1px 0; font-family: monospace;">
                <span style="color: {color};">{indent_str}⟳ {self._escape_html(text)}</span>
            </div>
            """
        else:
            # Regular messages
            formatted_message = f"""
            <div style="margin: 2px 0; font-family: monospace; padding: 1px 0;">
                <span style="color: #6e7681; font-size: 11px;">[{timestamp}]</span>
                <span style="color: {color}; font-weight: 500;">{indent_str}{icon} {self._escape_html(text)}</span>
            </div>
            """

        self.text_area.append(formatted_message)

        # Smart auto-scroll
        if self.auto_scroll_checkbox.isChecked():
            self._scroll_to_bottom()

    def _get_message_color(self, text: str, msg_type: str) -> str:
        """Enhanced color coding for better readability"""
        text_lower = text.lower()

        # Message type colors
        if msg_type == 'stream':
            return "#58a6ff"  # Blue for streaming
        elif msg_type == 'debug':
            return "#8b949e"  # Gray for debug

        # Content-based colors
        if any(word in text_lower for word in ['error', 'failed', 'exception', '❌']):
            return "#f85149"  # Red
        elif any(word in text_lower for word in ['✅', 'completed', 'success', 'approved']):
            return "#3fb950"  # Green
        elif any(word in text_lower for word in ['warning', '⚠️']):
            return "#d29922"  # Yellow
        elif any(word in text_lower for word in ['creating:', 'generating', 'streaming']):
            return "#00d7ff"  # Bright blue
        elif text.startswith('▶'):
            return "#a5a5f0"  # Purple for stages
        elif text.startswith('  📄'):
            return "#79c0ff"  # Light blue for files
        elif 'iteration' in text_lower:
            return "#f0883e"  # Orange for iterations

        return "#c9d1d9"  # Default

    def _get_message_icon(self, text: str, msg_type: str) -> str:
        """Smart icon detection"""
        # Don't add icon if text already has emoji
        if re.search(r'[🚀🧠📋⚙️📄✅❌⚠📊🔧📝⚡🎯🔍👁]', text):
            return ""

        text_lower = text.lower()

        if 'error' in text_lower or 'failed' in text_lower:
            return "❌"
        elif 'success' in text_lower or 'completed' in text_lower:
            return "✅"
        elif 'creating' in text_lower or 'generating' in text_lower:
            return "⚡"
        elif msg_type == 'stream':
            return "📺"

        return ""

    def _escape_html(self, text: str) -> str:
        """Safe HTML escaping"""
        if '<span style=' in text:
            return text
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def _scroll_to_bottom(self):
        """Smooth scroll to bottom"""
        cursor = self.text_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_area.setTextCursor(cursor)

    def add_separator(self, title: str = ""):
        """Add visual separator like aider"""
        if title:
            separator = f"{'─' * 15} {title} {'─' * 15}"
        else:
            separator = '─' * 50

        self.log(separator, "separator")

    # Enhanced status updates
    def update_workflow_progress(self, stage: str, description: str):
        """Update workflow progress"""
        self.workflow_progress.update_stage(stage, description)
        self.progress_display.start_stage(stage, description)

    def update_task_progress(self, completed: int, total: int):
        """Update task progress"""
        self.workflow_progress.update_progress(completed, total)

    def update_cache_status(self, cache_size: int, hit_rate: float):
        """Update cache status"""
        self.cache_status.setText(f"Cache: {cache_size} items ({hit_rate:.1%})")

        color = "#3fb950" if hit_rate >= 0.5 else "#ffb900" if hit_rate >= 0.2 else "#888"
        self.cache_status.setStyleSheet(f"color: {color}; font-size: 11px;")

    def update_performance_stats(self, speed_info: str):
        """Update performance information"""
        self.perf_status.setText(f"Speed: {speed_info}")

    def start_file_generation(self, file_path: str):
        """Start file generation tracking"""
        self.progress_display.start_file(file_path)
        self.streaming_indicator.start_streaming(f"Generating {file_path}")

    def complete_file_generation(self, file_path: str, success: bool):
        """Complete file generation tracking"""
        status = "success" if success else "error"
        self.progress_display.complete_file(file_path, status)
        self.streaming_indicator.stop_streaming("● Ready")

    @Slot()
    def clear_terminal(self):
        """Clear terminal with reset"""
        self.text_area.clear()
        self._add_initial_message()
        self.workflow_progress.reset_progress()
        self.streaming_indicator.stop_streaming()

    @Slot()
    def save_log(self):
        """Save terminal log"""
        try:
            from PySide6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Terminal Log",
                f"ava_terminal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text Files (*.txt);;All Files (*)"
            )

            if filename:
                plain_text = self.text_area.toPlainText()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(plain_text)
                self.log(f"Log saved: {filename}", "success")

        except Exception as e:
            self.log(f"Save failed: {e}", "error")


class TerminalWindow(StreamingTerminal):
    """Standalone terminal window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA Terminal - Enhanced Streaming & Progress")
        self.resize(1000, 750)

        self.setStyleSheet("""
            QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
            }
        """)