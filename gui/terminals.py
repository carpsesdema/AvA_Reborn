# terminals.py (updated)

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QCheckBox
from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from datetime import datetime
import queue
import threading


class StreamingTerminal(QWidget):
    """
    Enhanced Aider-like terminal widget with real-time streaming,
    color coding, and auto-scroll functionality.
    """

    def __init__(self):
        super().__init__()
        self.message_queue = queue.Queue()
        self._init_ui()
        self._setup_timer()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Control panel
        controls_layout = QHBoxLayout()

        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet("color: #c6c6c6;")

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
        controls_layout.addStretch()
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.save_button)

        # Terminal display
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)

        # Enhanced terminal styling
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
│                      Ready for action! 🚀                   │
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

    def _process_queue(self):
        """Process queued messages and update display"""
        messages_processed = 0
        max_messages_per_cycle = 10  # Prevent UI freezing

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

        # Color coding based on message type and content
        color = self._get_message_color(text, msg_type)
        icon = self._get_message_icon(text, msg_type)

        # Format the message with HTML
        formatted_message = f"""
        <div style="margin: 2px 0; font-family: monospace;">
            <span style="color: #6e7681; font-size: 11px;">[{timestamp}]</span>
            <span style="color: {color};">{icon} {self._escape_html(text)}</span>
        </div>
        """

        # Add to display
        self.text_area.append(formatted_message)

        # Auto-scroll if enabled
        if self.auto_scroll_checkbox.isChecked():
            self._scroll_to_bottom()

    def _get_message_color(self, text: str, msg_type: str) -> str:
        """Determine color based on message content and type"""
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ['error', 'failed', 'exception']):
            return "#f85149"  # Red for errors
        elif any(keyword in text_lower for keyword in ['✓', 'completed', 'success', 'finished']):
            return "#3fb950"  # Green for success
        elif any(keyword in text_lower for keyword in ['warning', 'warn']):
            return "#d29922"  # Yellow for warnings
        elif text.startswith('[PLANNING]'):
            return "#a5a5f0"  # Purple for planning
        elif text.startswith('[TASK]'):
            return "#58a6ff"  # Blue for tasks
        elif text.startswith('[FILE]'):
            return "#3fb950"  # Green for files
        elif text.startswith('[CODE]'):
            return "#f0883e"  # Orange for code
        else:
            return "#c9d1d9"  # Default white

    def _get_message_icon(self, text: str, msg_type: str) -> str:
        """Get appropriate icon for message"""
        text_lower = text.lower()

        if 'error' in text_lower or 'failed' in text_lower:
            return "❌"
        elif '✓' in text or 'completed' in text_lower or 'success' in text_lower:
            return "✅"
        elif 'warning' in text_lower:
            return "⚠️"
        elif text.startswith('[PLANNING]'):
            return "🧠"
        elif text.startswith('[TASK]'):
            return "⚙️"
        elif text.startswith('[FILE]'):
            return "📄"
        elif text.startswith('[CODE]'):
            return "🔧"
        else:
            return "ℹ️"

    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in text"""
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
        Stream LLM response in real-time.
        Usage: terminal.stream_llm_response(llm_client.stream_chat(...))
        """

        def stream_worker():
            try:
                for chunk in response_generator:
                    self.append_text(chunk, "LLM")
            except Exception as e:
                self.append_text(f"Streaming error: {e}", "ERROR")

        # Run streaming in separate thread to avoid blocking UI
        thread = threading.Thread(target=stream_worker, daemon=True)
        thread.start()

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
        self.setWindowTitle("AvA Terminal")
        self.resize(800, 600)

        # Add window-specific styling
        self.setStyleSheet("""
            QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
            }
        """)