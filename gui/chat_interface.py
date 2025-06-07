# gui/chat_interface.py - Modern Chat Interface with Beautiful Bubbles

from datetime import datetime

from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QScrollArea, QFrame, QLabel, QSizePolicy
)

from gui.components import ModernButton, StatusIndicator, Colors, Typography


class ChatBubble(QFrame):
    """Modern chat bubble with proper styling and animations"""

    def __init__(self, message: str, sender: str, role: str, timestamp: datetime):
        super().__init__()
        self.message = message
        self.sender = sender
        self.role = role
        self.timestamp = timestamp

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Setup the bubble UI structure"""
        self.setContentsMargins(0, 0, 0, 0)

        # Main layout with proper alignment
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create bubble content
        bubble_content = self._create_bubble_content()

        if self.role == "user":
            # User messages align right
            main_layout.addStretch()
            main_layout.addWidget(bubble_content)
            main_layout.addSpacing(20)
        else:
            # Assistant messages align left
            main_layout.addSpacing(20)
            main_layout.addWidget(bubble_content)
            main_layout.addStretch()

        self.setLayout(main_layout)

    def _create_bubble_content(self):
        """Create the actual bubble content widget"""
        bubble = QFrame()
        bubble.setMaximumWidth(600)  # Limit bubble width
        bubble.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(bubble)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header with sender and timestamp
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Sender name with icon
        sender_label = QLabel()
        if self.role == "user":
            icon = "👤"
            name = "You"
        elif self.role == "streaming":
            icon = "⚡"
            name = "AvA"
        else:
            icon = "🤖"
            name = "AvA"

        sender_label.setText(f"{icon} {name}")
        sender_label.setFont(Typography.heading_small())

        # Timestamp
        time_label = QLabel(self.timestamp.strftime("%H:%M"))
        time_label.setFont(Typography.body_small())

        header_layout.addWidget(sender_label)
        header_layout.addStretch()
        header_layout.addWidget(time_label)

        # Message content
        content_label = QLabel()
        content_label.setText(self.message)
        content_label.setWordWrap(True)
        content_label.setFont(Typography.body())
        content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        content_label.setOpenExternalLinks(True)

        layout.addLayout(header_layout)
        layout.addWidget(content_label)

        return bubble

    def _apply_style(self):
        """Apply bubble styling based on role"""
        if self.role == "user":
            # User bubble - blue accent
            bubble_bg = Colors.ACCENT_BLUE
            text_color = Colors.TEXT_PRIMARY
            sender_color = Colors.TEXT_PRIMARY
            time_color = "rgba(240, 246, 252, 0.7)"

        elif self.role == "streaming":
            # Streaming bubble - orange accent
            bubble_bg = Colors.ACCENT_ORANGE
            text_color = Colors.TEXT_PRIMARY
            sender_color = Colors.TEXT_PRIMARY
            time_color = "rgba(240, 246, 252, 0.7)"

        else:
            # Assistant bubble - elevated background
            bubble_bg = Colors.ELEVATED_BG
            text_color = Colors.TEXT_PRIMARY
            sender_color = Colors.ACCENT_GREEN
            time_color = Colors.TEXT_MUTED

        self.setStyleSheet(f"""
            ChatBubble {{
                background: transparent;
                border: none;
            }}
            ChatBubble QFrame {{
                background: {bubble_bg};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 16px;
                margin: 2px;
            }}
            ChatBubble QLabel {{
                background: transparent;
                border: none;
                color: {text_color};
            }}
        """)

        # Special styling for header elements
        for label in self.findChildren(QLabel):
            if "👤" in label.text() or "🤖" in label.text() or "⚡" in label.text():
                label.setStyleSheet(f"color: {sender_color}; font-weight: 600;")
            elif ":" in label.text() and len(label.text()) == 5:  # Time format
                label.setStyleSheet(f"color: {time_color}; font-size: 10px;")


class ChatScrollArea(QScrollArea):
    """Custom scroll area for chat messages"""

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """Setup scroll area"""
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(16)
        self.content_layout.addStretch()  # Push messages to bottom initially

        self.setWidget(self.content_widget)

        # Style the scroll area
        self.setStyleSheet(f"""
            QScrollArea {{
                background: {Colors.PRIMARY_BG};
                border: none;
                border-radius: 12px;
            }}
            QScrollBar:vertical {{
                background: {Colors.SECONDARY_BG};
                width: 8px;
                border-radius: 4px;
                margin: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER_DEFAULT};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Colors.ACCENT_BLUE};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
                height: 0px;
            }}
        """)

    def add_bubble(self, bubble: ChatBubble):
        """Add a new chat bubble"""
        # Remove the stretch before adding new bubble
        if self.content_layout.count() > 0:
            last_item = self.content_layout.itemAt(self.content_layout.count() - 1)
            if last_item and last_item.spacerItem():
                self.content_layout.removeItem(last_item)

        self.content_layout.addWidget(bubble)
        self.content_layout.addStretch()  # Add stretch back

        # Auto-scroll to bottom
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        """Smooth scroll to bottom"""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class ModernChatInput(QFrame):
    """Modern chat input with send button"""

    message_sent = Signal(str)

    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Setup input UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(8)

        # Message input
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setFont(Typography.body())
        self.input_field.returnPressed.connect(self._send_message)

        # Send button
        self.send_button = ModernButton("Send", button_type="primary")
        self.send_button.setMaximumWidth(80)
        self.send_button.clicked.connect(self._send_message)

        layout.addWidget(self.input_field)
        layout.addWidget(self.send_button)
        self.setLayout(layout)

    def _apply_style(self):
        """Apply input styling"""
        self.setStyleSheet(f"""
            ModernChatInput {{
                background: {Colors.SECONDARY_BG};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 12px;
                margin: 4px;
            }}
            QLineEdit {{
                background: transparent;
                border: none;
                color: {Colors.TEXT_PRIMARY};
                padding: 8px 12px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                outline: none;
            }}
        """)

    def _send_message(self):
        """Send the message"""
        message = self.input_field.text().strip()
        if message:
            self.message_sent.emit(message)
            self.input_field.clear()

    def focus_input(self):
        """Focus the input field"""
        self.input_field.setFocus()


class ChatInterface(QWidget):
    """Main modern chat interface"""

    message_sent = Signal(str)
    workflow_requested = Signal(str, list)

    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self._setup_ui()
        self._add_welcome_message()

    def _setup_ui(self):
        """Setup the main chat interface"""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Chat display area
        self.chat_scroll = ChatScrollArea()
        layout.addWidget(self.chat_scroll, 1)

        # Chat input
        self.chat_input = ModernChatInput()
        self.chat_input.message_sent.connect(self._handle_message_sent)
        layout.addWidget(self.chat_input)

        # Status bar
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 8, 0, 0)

        # AI Specialists status
        self.specialists_indicator = StatusIndicator("ready")
        self.specialists_text = QLabel("AI Specialists: Ready")
        self.specialists_text.setFont(Typography.body_small())
        self.specialists_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        status_layout.addWidget(self.specialists_indicator)
        status_layout.addWidget(self.specialists_text)
        status_layout.addStretch()

        # Performance status
        self.performance_indicator = StatusIndicator("ready")
        self.performance_text = QLabel("Performance: Ready")
        self.performance_text.setFont(Typography.body_small())
        self.performance_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        status_layout.addWidget(self.performance_indicator)
        status_layout.addWidget(self.performance_text)
        status_layout.addStretch()

        # RAG status
        self.rag_indicator = StatusIndicator("working")
        self.rag_text = QLabel("RAG: Initializing...")
        self.rag_text.setFont(Typography.body_small())
        self.rag_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        status_layout.addWidget(self.rag_indicator)
        status_layout.addWidget(self.rag_text)

        layout.addLayout(status_layout)
        self.setLayout(layout)

        # Apply main styling
        self.setStyleSheet(f"""
            ChatInterface {{
                background: {Colors.PRIMARY_BG};
            }}
        """)

    def _add_welcome_message(self):
        """Add the welcome message"""
        welcome_text = """Hello! I'm AvA, your fast professional AI development assistant.

🚀 **Ready to build something amazing?**

I use specialized AI agents:
- **Planner** - Creates smart project architecture
- **Coder** - Generates clean, professional code  
- **Assembler** - Integrates everything seamlessly
- **Reviewer** - Ensures quality and best practices

✨ **Just tell me what you want to build!**
Examples: "Create a calculator GUI", "Build a web API", "Make a file organizer tool" """

        bubble = ChatBubble(
            message=welcome_text,
            sender="AvA",
            role="assistant",
            timestamp=datetime.now()
        )
        self.chat_scroll.add_bubble(bubble)

    def _handle_message_sent(self, message: str):
        """Handle sent messages"""
        # Add user message
        self.add_user_message(message)

        # Store in history
        self.conversation_history.append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

        # Keep last 10 messages
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

        # Emit signal
        self.message_sent.emit(message)

    def add_user_message(self, message: str):
        """Add a user message bubble"""
        bubble = ChatBubble(
            message=message,
            sender="You",
            role="user",
            timestamp=datetime.now()
        )
        self.chat_scroll.add_bubble(bubble)

    def add_assistant_response(self, response: str):
        """Add assistant response bubble"""
        bubble = ChatBubble(
            message=response,
            sender="AvA",
            role="assistant",
            timestamp=datetime.now()
        )
        self.chat_scroll.add_bubble(bubble)

        # Store in history
        self.conversation_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })

    def add_workflow_status(self, status: str):
        """Add workflow status message"""
        bubble = ChatBubble(
            message=f"🔄 {status}",
            sender="AvA",
            role="streaming",
            timestamp=datetime.now()
        )
        self.chat_scroll.add_bubble(bubble)

    def update_specialists_status(self, text: str, status: str = "ready"):
        """Update specialists status"""
        self.specialists_text.setText(text)
        self.specialists_indicator.update_status(status)

    def update_performance_status(self, text: str, status: str = "ready"):
        """Update performance status"""
        self.performance_text.setText(text)
        self.performance_indicator.update_status(status)

    def update_rag_status(self, text: str, status: str = "working"):
        """Update RAG status"""
        self.rag_text.setText(text)
        self.rag_indicator.update_status(status)