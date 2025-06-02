# gui/terminals.py - Enhanced with Progress Visualization

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QCheckBox, QProgressBar, QLabel
from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from datetime import datetime
import queue
import threading


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
            "starting": "#ffb900",
            "planning": "#a5a5f0",
            "decomposition": "#f0883e",
            "generation": "#00d7ff",
            "finalization": "#3fb950",
            "complete": "#3fb950",
            "error": "#f85149"
        }

        stage_emojis = {
            "idle": "⏸️",
            "starting": "🚀",
            "planning": "🧠",
            "decomposition": "📋",
            "generation": "⚙️",
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
        self.setFixedSize(100, 20)
        self.is_streaming = False
        self.animation_step = 0

        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.setInterval(200)  # 200ms animation

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

        dots = ["", ".", "..", "..."]
        dot = dots[self.animation_step % len(dots)]
        self.label.setText(f"⚡ {self.base_message}{dot}")
        self.animation_step += 1


class StreamingTerminal(QWidget):
    """
    Enhanced Aider-like terminal widget with real-time streaming,
    progress visualization, and workflow tracking.
    """

    def __init__(self):
        super().__init__()
        self.message_queue = queue.Queue()
        self.stream_buffer = ""
        self.current_stream_type = None
        self._init_ui()
        self._setup_timer()

    def _init_ui(self):
        layout = QVBoxLayout()

        # NEW: Workflow progress section
        self.workflow_progress = WorkflowProgressWidget()
        layout.addWidget(self.workflow_progress)

        # Control panel
        controls_layout = QHBoxLayout()

        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet("color: #c6c6c6;")

        # NEW: Streaming indicator
        self.streaming_indicator = StreamingIndicator()

        # Cache status label
        self.cache_status = QLabel("Cache: 0 items")
        self.cache_status.setStyleSheet("color: #888; font-size: 11px;")

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_terminal)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
            }
        """)

        self.save_button = QPushButton("Save Log")
        self.save_button.clicked.connect(self.save_log)
        self.save_button.setStyleSheet(self.clear_button.styleSheet())

        controls_layout.addWidget(self.auto_scroll_checkbox)
        controls_layout.addWidget(self.streaming_indicator)
        controls_layout.addWidget(self.cache_status)
        controls_layout.addStretch()
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.save_button)

        # Terminal display
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)

        # Enhanced terminal styling with better contrast
        self.text_area.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #58a6ff;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 10px;
                font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                line-height: 1.4;
            }
        """)

        # Set font explicitly
        font = QFont("JetBrains Mono", 12)
        font.setStyleHint(QFont.Monospace)
        self.text_area.setFont(font)

        layout.addLayout(controls_layout)
        layout.addWidget(self.text_area)
        self.setLayout(layout)

        # Add initial welcome message
        self._add_initial_message()

    def _setup_timer(self):
        """Setup timer for processing queued messages"""
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_queue)
        self.timer.start(50)  # Process queue every 50ms for smooth streaming

    def _add_initial_message(self):
        """Add welcome message to terminal"""
        welcome_msg = """
<div style="color: #7c3aed; font-weight: bold;">
╭─────────────────────────────────────────────────────────────╮
│                     AvA Development Terminal                │
│            Ready for streaming workflows! 🚀⚡              │
│                Enhanced with progress tracking              │
╰─────────────────────────────────────────────────────────────╯
</div>
        """.strip()
        self.text_area.append(welcome_msg)

    def append_text(self, text: str, message_type: str = "INFO"):
        """Queue text to be added to terminal with timestamp and color coding"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Thread-safe queuing
        self.message_queue.put({
            'timestamp': timestamp,
            'text': text,
            'type': message_type
        })

    def log(self, message: str, message_type: str = "info"):
        """
        Add message to terminal - main entry point
        """
        self.append_text(message, message_type)

    def _process_queue(self):
        """Process queued messages and update display"""
        messages_processed = 0
        max_messages_per_cycle = 15  # Increased for better streaming

        while not self.message_queue.empty() and messages_processed < max_messages_per_cycle:
            try:
                message = self.message_queue.get_nowait()
                self._display_message(message)
                messages_processed += 1
            except queue.Empty:
                break

    def _display_message(self, message: dict):
        """Display a single message with appropriate formatting"""
        timestamp = message['timestamp']
        text = message['text']
        msg_type = message['type']

        # Enhanced color coding and icons
        color = self._get_message_color(text, msg_type)
        icon = self._get_message_icon(text, msg_type)

        # Special handling for progress messages
        if "%" in text and ("progress" in text.lower() or "executing:" in text.lower()):
            # Extract percentage for special formatting
            import re
            percent_match = re.search(r'\[(\d+)%\]', text)
            if percent_match:
                percent = percent_match.group(1)
                text = text.replace(f"[{percent}%]",
                                    f"<span style='color: #00d7ff; font-weight: bold;'>[{percent}%]</span>")

        # Format the message with enhanced HTML
        formatted_message = f"""
        <div style="margin: 2px 0; font-family: monospace; padding: 2px 0;">
            <span style="color: #6e7681; font-size: 11px;">[{timestamp}]</span>
            <span style="color: {color}; font-weight: 500;">{icon} {self._escape_html(text)}</span>
        </div>
        """

        # Add to display
        self.text_area.append(formatted_message)

        # Auto-scroll if enabled
        if self.auto_scroll_checkbox.isChecked():
            self._scroll_to_bottom()

    def _get_message_color(self, text: str, msg_type: str) -> str:
        """Enhanced color determination with more categories"""
        text_lower = text.lower()

        # Error states
        if any(keyword in text_lower for keyword in ['error', 'failed', 'exception', '❌']):
            return "#f85149"  # Red

        # Success states
        if any(keyword in text_lower for keyword in ['✅', 'completed', 'success', 'finished', 'written:']):
            return "#3fb950"  # Green

        # Warning states
        if any(keyword in text_lower for keyword in ['warning', 'warn', '⚠️']):
            return "#d29922"  # Yellow

        # Progress and work states
        if any(keyword in text_lower for keyword in ['executing:', 'generating', 'processing', 'assembling']):
            return "#00d7ff"  # Bright blue for active work

        # Workflow stages
        if text.startswith('[PLANNING]') or '🧠' in text:
            return "#a5a5f0"  # Purple for planning
        elif text.startswith('[TASK]') or '⚙️' in text:
            return "#58a6ff"  # Blue for tasks
        elif text.startswith('[FILE]') or '📄' in text:
            return "#3fb950"  # Green for files
        elif text.startswith('[CODE]') or '🔧' in text:
            return "#f0883e"  # Orange for code

        # Cache operations
        elif 'cache' in text_lower and '⚡' in text:
            return "#ffb900"  # Gold for cache hits

        # Streaming operations
        elif 'streaming' in text_lower or 'chars)' in text:
            return "#40e0ff"  # Light blue for streaming

        return "#c9d1d9"  # Default white

    def _get_message_icon(self, text: str, msg_type: str) -> str:
        """Enhanced icon determination"""
        text_lower = text.lower()

        # Check for existing emojis first
        if any(emoji in text for emoji in ['🚀', '🧠', '📋', '⚙️', '📄', '✅', '❌', '⚠️', '📊', '🔧', '📝', '⚡']):
            return ""  # No additional icon needed

        # Progress indicators
        if '[' in text and '%' in text and ']' in text:
            return "⏳"

        # Error states
        if 'error' in text_lower or 'failed' in text_lower:
            return "❌"

        # Success states
        if 'completed' in text_lower or 'success' in text_lower:
            return "✅"

        # Work in progress
        if any(keyword in text_lower for keyword in ['calling', 'querying', 'analyzing']):
            return "⚙️"

        # Cache operations
        if 'cache' in text_lower:
            return "💾"

        return "ℹ️"

    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in text but preserve our formatting"""
        # Don't escape if text already contains our span tags
        if '<span style=' in text:
            return text
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def _scroll_to_bottom(self):
        """Scroll terminal to bottom"""
        cursor = self.text_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_area.setTextCursor(cursor)

    @Slot()
    def clear_terminal(self):
        """Clear terminal contents"""
        self.text_area.clear()
        self._add_initial_message()
        self.workflow_progress.reset_progress()

    @Slot()
    def save_log(self):
        """Save terminal log to file"""
        try:
            from PySide6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Terminal Log",
                f"ava_terminal_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text Files (*.txt);;All Files (*)"
            )

            if filename:
                # Get plain text content (strip HTML)
                plain_text = self.text_area.toPlainText()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(plain_text)

                self.append_text(f"Log saved to: {filename}", "SUCCESS")

        except Exception as e:
            self.append_text(f"Failed to save log: {e}", "ERROR")

    def stream_llm_response(self, response_generator):
        """
        Stream LLM response in real-time with enhanced feedback
        """

        def stream_worker():
            try:
                self.streaming_indicator.start_streaming("LLM Streaming")
                chunk_count = 0

                for chunk in response_generator:
                    self.append_text(chunk, "LLM_STREAM")
                    chunk_count += 1

                    # Update streaming indicator periodically
                    if chunk_count % 5 == 0:
                        self.streaming_indicator.start_streaming(f"LLM Streaming ({chunk_count} chunks)")

                self.streaming_indicator.stop_streaming("● Stream Complete")

            except Exception as e:
                self.streaming_indicator.stop_streaming("● Stream Error")
                self.append_text(f"Streaming error: {e}", "ERROR")

        # Run streaming in separate thread to avoid blocking UI
        thread = threading.Thread(target=stream_worker, daemon=True)
        thread.start()

    def update_workflow_progress(self, stage: str, description: str):
        """Update workflow progress visualization"""
        self.workflow_progress.update_stage(stage, description)

    def update_task_progress(self, completed: int, total: int):
        """Update task progress counter"""
        self.workflow_progress.update_progress(completed, total)

    def update_cache_status(self, cache_size: int, hit_rate: float):
        """Update cache status display"""
        self.cache_status.setText(f"Cache: {cache_size} items ({hit_rate:.1%} hit)")

        # Color code based on hit rate
        if hit_rate >= 0.5:
            color = "#3fb950"  # Green for good hit rate
        elif hit_rate >= 0.2:
            color = "#ffb900"  # Yellow for moderate hit rate
        else:
            color = "#888"  # Gray for low hit rate

        self.cache_status.setStyleSheet(f"color: {color}; font-size: 11px;")

    def add_separator(self, title: str = ""):
        """Add a visual separator to the terminal"""
        if title:
            separator = f"\n{'─' * 20} {title} {'─' * 20}\n"
        else:
            separator = f"\n{'─' * 60}\n"

        self.append_text(separator, "SEPARATOR")


class TerminalWindow(StreamingTerminal):
    """
    Standalone terminal window that can be opened separately.
    Inherits all functionality from StreamingTerminal.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA Terminal - Enhanced with Streaming & Progress")
        self.resize(900, 700)  # Slightly larger for progress widgets

        # Add window-specific styling
        self.setStyleSheet("""
            QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
            }
        """)