# gui/main_window.py - COMPLETE WORKING REPLACEMENT (Corrected to use internal sidebar and fix New Project button)

from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFrame, QLabel, QTextEdit, QScrollArea, QComboBox, QSlider, QPushButton
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from gui.components import ModernButton, StatusIndicator


# We are NOT using gui.enhanced_sidebar here, but the AvALeftSidebar defined below.


class ChatDisplay(QTextEdit):
    """Actual chat display"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                color: #cccccc;
                padding: 12px;
                font-family: "Segoe UI";
                font-size: 14px;
            }
        """)
        self.append(self._format_message("AvA",
                                         "Hello! I'm AvA. I can help you build applications or just chat. What would you like to work on?",
                                         "assistant"))

    def add_user_message(self, message: str):
        self.append(self._format_message("You", message, "user"))

    def add_assistant_message(self, message: str):
        self.append(self._format_message("AvA", message, "assistant"))

    def _format_message(self, sender: str, message: str, role: str) -> str:
        if role == "user":
            color = "#00d7ff"
            bg = "#2d2d30"
        else:  # assistant
            color = "#3fb950"  # Green for AvA's responses
            bg = "#252526"

        return f"""
        <div style="margin: 8px 0; padding: 8px 12px; background: {bg}; border-radius: 8px; border-left: 3px solid {color};">
            <div style="font-weight: bold; color: {color}; margin-bottom: 4px;">{sender}:</div>
            <div style="color: #cccccc; line-height: 1.4;">{message}</div>
        </div>
        """


class ChatInterface(QWidget):
    message_sent = Signal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.chat_display = ChatDisplay()
        layout.addWidget(self.chat_display, 1)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Chat with AvA... (type 'build' or 'create' to start a project)")
        self.input_field.setMinimumHeight(45)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: #2d2d30;
                border: 2px solid #404040;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                padding: 12px 16px;
            }
            QLineEdit:focus {
                border-color: #00d7ff;
            }
            QLineEdit::placeholder {
                color: #888;
            }
        """)
        self.input_field.returnPressed.connect(self._send_message)

        self.send_btn = ModernButton("Send", button_type="accent")
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setMinimumHeight(45)
        self.send_btn.setMinimumWidth(80)

        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_btn)

        # Status Bar Layout (as per your screenshot)
        status_bar_layout = QHBoxLayout()
        status_bar_layout.setContentsMargins(0, 8, 0, 0)  # Align with input field's bottom

        # LLM Status (left side of status bar)
        self.llm_status_indicator = StatusIndicator("ready")  # Blue dot
        self.llm_status_text = QLabel("LLM: Gemini: gemini-2.5-pro")  # Example text
        self.llm_status_text.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")

        status_bar_layout.addWidget(self.llm_status_indicator)
        status_bar_layout.addWidget(self.llm_status_text)
        status_bar_layout.addStretch(1)  # Pushes RAG status to the right

        # RAG Status (right side of status bar)
        self.rag_status_indicator = StatusIndicator("error")  # Red dot for "Initialization Failed"
        self.rag_status_text_label = QLabel("RAG: Initialization Failed")  # Example text
        self.rag_status_text_label.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")

        status_bar_layout.addWidget(self.rag_status_indicator)
        status_bar_layout.addWidget(self.rag_status_text_label)

        layout.addLayout(input_layout)
        layout.addLayout(status_bar_layout)
        self.setLayout(layout)

    def _send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return
        self.chat_display.add_user_message(message)
        self.input_field.clear()
        self.message_sent.emit(message)

    def update_llm_status(self, text: str, indicator_status: str = "ready"):
        self.llm_status_text.setText(text)
        self.llm_status_indicator.update_status(indicator_status)

    def update_rag_ui_status(self, text: str, color_hex_or_status_key: str):
        self.rag_status_text_label.setText(text)
        if color_hex_or_status_key.startswith("#"):
            self.rag_status_indicator.setStyleSheet(f"""
                StatusIndicator {{
                    background: {color_hex_or_status_key};
                    border: 2px solid #1e1e1e; /* Match theme */
                    border-radius: 6px;
                }}
            """)
        else:
            self.rag_status_indicator.update_status(color_hex_or_status_key)


class AvALeftSidebar(QWidget):  # This is the internal sidebar class
    model_changed = Signal(str, str)  # (type, name)
    temperature_changed = Signal(float)
    action_triggered = Signal(str)  # For Chat Actions panel
    new_project_sidebar_action = Signal()  # For the "New Project" button in THIS sidebar

    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)  # As per your design
        self._init_ui()
        # Connections for buttons within this sidebar will be made in _init_ui

    def _init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)  # Overall sidebar padding
        main_layout.setSpacing(12)  # Spacing between panels

        # --- PROJECT PANEL (matching your screenshot's internal sidebar) ---
        projects_frame = QFrame()
        projects_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2e, stop:1 #252526);
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 0px; /* No margin for the panel itself if main_layout has padding */
            }
        """)
        projects_panel_layout = QVBoxLayout(projects_frame)  # Use projects_frame as parent
        projects_panel_layout.setContentsMargins(12, 8, 12, 12)  # Padding inside the panel
        projects_panel_layout.setSpacing(8)  # Spacing for elements in this panel

        # Title for project panel (not explicitly in screenshot, but good practice)
        # For screenshot accuracy, we can use the "Projects" / "Sessions" tabs as title-like elements

        # Projects/Sessions Tabs
        project_tabs_layout = QHBoxLayout()
        # TODO: Add actual tab switching functionality if needed, for now, static buttons
        projects_tab_btn = QPushButton("Projects")
        sessions_tab_btn = QPushButton("Sessions")
        projects_tab_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; border: none; padding: 5px; border-top-left-radius: 3px; border-bottom-left-radius: 3px;} QPushButton:hover{background-color: #005A9E;}")
        sessions_tab_btn.setStyleSheet(
            "QPushButton { background-color: #505050; color: #CCCCCC; border: none; padding: 5px; border-top-right-radius: 3px; border-bottom-right-radius: 3px;} QPushButton:hover{background-color: #606060;}")

        project_tabs_layout.addWidget(projects_tab_btn, 1)  # Stretch factor
        project_tabs_layout.addWidget(sessions_tab_btn, 1)  # Stretch factor
        project_tabs_layout.setSpacing(0)  # No space between tab buttons
        projects_panel_layout.addLayout(project_tabs_layout)

        projects_label = QLabel("Projects:")
        projects_label.setStyleSheet("color: #cccccc; font-weight: normal; margin-top: 8px;")  # Adjusted font weight
        projects_panel_layout.addWidget(projects_label)

        # Project List Area (Simplified to match screenshot's appearance)
        self.project_list_display = QFrame()  # Using a QFrame for the blue selected item
        self.project_list_display.setMinimumHeight(60)  # Approximate height
        self.project_list_display.setStyleSheet("background-color: #0078D4; border-radius: 4px; padding: 8px;")
        project_list_item_layout = QVBoxLayout(self.project_list_display)

        self.default_project_label = QLabel("Default Project")  # Static text for now
        self.default_project_label.setStyleSheet("color: white; font-weight: bold;")
        # Adding a dummy progress bar as in the screenshot (visual only for now)
        dummy_progress = QFrame()
        dummy_progress.setFixedHeight(4)
        dummy_progress.setStyleSheet("background-color: #00D7FF; border-radius: 2px; margin-top: 4px;")
        project_list_item_layout.addWidget(self.default_project_label)
        project_list_item_layout.addWidget(dummy_progress)
        project_list_item_layout.addStretch()  # Push content up
        projects_panel_layout.addWidget(self.project_list_display)

        # --- "New Project" Button ---
        self.new_project_btn = ModernButton("📁 New Project",
                                            button_type="primary")  # "primary" style from your components
        self.new_project_btn.setStyleSheet(projects_tab_btn.styleSheet().replace("#0078D4", "#005A9E").replace("white",
                                                                                                               "#FFFFFF"))  # Re-use and adjust style
        self.new_project_btn.clicked.connect(self.new_project_sidebar_action.emit)  # EMIT THE SIGNAL
        projects_panel_layout.addWidget(self.new_project_btn)

        sessions_label = QLabel("Sessions (Current Project):")
        sessions_label.setStyleSheet("color: #cccccc; font-weight: normal; margin-top: 8px;")
        projects_panel_layout.addWidget(sessions_label)

        # Session List Area (Simplified)
        self.session_list_display = QFrame()
        self.session_list_display.setMinimumHeight(30)
        self.session_list_display.setStyleSheet(
            "background-color: #00D7FF; border-radius: 4px; padding: 8px;")  # Bright blue
        session_list_item_layout = QVBoxLayout(self.session_list_display)
        main_chat_label = QLabel("Main Chat")
        main_chat_label.setStyleSheet("color: #1E1E1E; font-weight: bold;")  # Dark text on light blue
        session_list_item_layout.addWidget(main_chat_label)
        session_list_item_layout.addStretch()
        projects_panel_layout.addWidget(self.session_list_display)

        main_layout.addWidget(projects_frame)

        # --- LLM CONFIG PANEL (matching screenshot) ---
        llm_frame = QFrame()
        llm_frame.setStyleSheet(projects_frame.styleSheet())  # Reuse panel style
        llm_panel_layout = QVBoxLayout(llm_frame)
        llm_panel_layout.setContentsMargins(12, 8, 12, 12)
        llm_panel_layout.setSpacing(8)

        llm_title = QLabel("LLM Configuration")
        llm_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        llm_title.setStyleSheet("color: #00d7ff; margin-bottom: 6px; background: transparent; border: none;")
        llm_panel_layout.addWidget(llm_title)

        # Chat LLM
        chat_llm_layout = QHBoxLayout()
        chat_llm_label = QLabel("Chat LLM:")
        chat_llm_label.setStyleSheet("color: #cccccc;")
        self.chat_combo = QComboBox()
        self.chat_combo.addItems(["Gemini: gemini-2.5-pro-p", "OpenAI: gpt-4o", "Anthropic: claude-3.5-sonnet"])
        self.chat_combo.setStyleSheet(
            "QComboBox { background: #1e1e1e; border: 1px solid #404040; border-radius: 4px; padding: 4px 8px; color: #cccccc; font-size: 10px;} QComboBox:hover{border-color:#00d7ff;} QComboBox::drop-down{border:none; width:12px;} QComboBox QAbstractItemView { background: #2d2d30; border: 1px solid #00d7ff; selection-background-color: #00d7ff; selection-color: #1e1e1e; }")
        chat_llm_status = StatusIndicator("ready")  # Blue dot
        chat_llm_layout.addWidget(chat_llm_label)
        chat_llm_layout.addWidget(self.chat_combo, 1)
        chat_llm_layout.addWidget(chat_llm_status)
        llm_panel_layout.addLayout(chat_llm_layout)

        # Specialized LLM
        spec_llm_label = QLabel("Specialized LLM (Code Gen):")
        spec_llm_label.setStyleSheet("color: #cccccc; margin-top: 5px;")
        llm_panel_layout.addWidget(spec_llm_label)

        spec_llm_layout = QHBoxLayout()
        self.code_combo = QComboBox()
        self.code_combo.addItems(["Ollama (Gen): qwen2.5-coder", "OpenAI: gpt-4o"])
        self.code_combo.setStyleSheet(self.chat_combo.styleSheet())  # Reuse style
        spec_llm_status = StatusIndicator("ready")
        spec_llm_layout.addWidget(self.code_combo, 1)
        spec_llm_layout.addWidget(spec_llm_status)
        llm_panel_layout.addLayout(spec_llm_layout)

        # Temperature
        temp_label_layout = QHBoxLayout()
        temp_label = QLabel("Temperature (Chat):")
        temp_label.setStyleSheet("color: #cccccc; margin-top: 5px;")
        self.temp_value_label = QLabel("0.70")
        self.temp_value_label.setStyleSheet("color: #00d7ff; font-weight: bold; margin-top: 5px;")
        temp_label_layout.addWidget(temp_label)
        temp_label_layout.addStretch()
        temp_label_layout.addWidget(self.temp_value_label)
        llm_panel_layout.addLayout(temp_label_layout)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.setStyleSheet(
            "QSlider::groove:horizontal {border:1px solid #404040; height:4px; background:#1e1e1e; border-radius:2px;} QSlider::handle:horizontal {background:#00d7ff; border:2px solid #00d7ff; width:14px; height:14px; border-radius:7px; margin:-5px 0;} QSlider::sub-page:horizontal{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #00d7ff,stop:1 #0078d4); border-radius:2px;}")
        self.temp_slider.valueChanged.connect(lambda v: self.temp_value_label.setText(f"{v / 100.0:.2f}"))
        self.temp_slider.valueChanged.connect(lambda v: self.temperature_changed.emit(v / 100.0))
        llm_panel_layout.addWidget(self.temp_slider)

        configure_persona_btn = ModernButton("🎭 Configure Persona", button_type="secondary")
        llm_panel_layout.addWidget(configure_persona_btn)
        main_layout.addWidget(llm_frame)

        # --- KNOWLEDGE BASE (RAG) PANEL ---
        rag_frame = QFrame()
        rag_frame.setStyleSheet(projects_frame.styleSheet())  # Reuse panel style
        rag_panel_layout = QVBoxLayout(rag_frame)
        rag_panel_layout.setContentsMargins(12, 8, 12, 12)
        rag_panel_layout.setSpacing(8)

        rag_title = QLabel("Knowledge Base (RAG)")
        rag_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rag_title.setStyleSheet("color: #00d7ff; margin-bottom: 6px; background:transparent; border:none;")
        rag_panel_layout.addWidget(rag_title)

        scan_dir_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        scan_dir_btn.clicked.connect(lambda: self.action_triggered.emit("scan_directory"))
        rag_panel_layout.addWidget(scan_dir_btn)

        add_files_btn = ModernButton("📄 Add Files (Project)", button_type="secondary")
        add_files_btn.clicked.connect(lambda: self.action_triggered.emit("add_files"))
        rag_panel_layout.addWidget(add_files_btn)

        rag_status_layout = QHBoxLayout()
        rag_status_label = QLabel("RAG:")
        rag_status_label.setStyleSheet("color: #cccccc;")
        self.rag_sidebar_status_text = QLabel("Initializing embedder...")  # Text from screenshot
        self.rag_sidebar_status_text.setStyleSheet("color: #ffb900; font-size: 9px;")  # Amber color
        rag_status_layout.addWidget(rag_status_label)
        rag_status_layout.addWidget(self.rag_sidebar_status_text, 1)
        rag_panel_layout.addLayout(rag_status_layout)
        main_layout.addWidget(rag_frame)

        # --- CHAT ACTIONS PANEL ---
        actions_frame = QFrame()
        actions_frame.setStyleSheet(projects_frame.styleSheet())  # Reuse panel style
        actions_panel_layout = QVBoxLayout(actions_frame)
        actions_panel_layout.setContentsMargins(12, 8, 12, 12)
        actions_panel_layout.setSpacing(8)

        actions_title = QLabel("Chat Actions")
        actions_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        actions_title.setStyleSheet("color: #00d7ff; margin-bottom: 6px; background:transparent; border:none;")
        actions_panel_layout.addWidget(actions_title)

        action_buttons = [  # Matching screenshot
            ("💬 New Session", "new_session"),
            ("📊 View LLM Log", "view_log"),  # Changed icon to text
            ("⚡ View Generated Code", "view_code"),  # Changed icon to text
            ("🔨 Force Code Gen", "force_gen"),
            ("🔄 Check for Updates", "check_updates")
        ]
        for text, action_id in action_buttons:
            btn = ModernButton(text, button_type="secondary")
            btn.clicked.connect(lambda checked, a=action_id: self.action_triggered.emit(a))
            actions_panel_layout.addWidget(btn)
        main_layout.addWidget(actions_frame)

        main_layout.addStretch()  # Pushes all panels to the top
        self.setLayout(main_layout)

        self.setStyleSheet("""
            AvALeftSidebar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #252526);
                border-right: 2px solid #00d7ff;
            }
        """)
        self.chat_combo.currentIndexChanged.connect(
            lambda: self.model_changed.emit("chat", self.chat_combo.currentText())
        )
        self.code_combo.currentIndexChanged.connect(
            lambda: self.model_changed.emit("code", self.code_combo.currentText())
        )

    def get_current_models(self):
        return {
            "chat_model": self.chat_combo.currentText(),
            "code_model": self.code_combo.currentText(),
            "temperature": self.temp_slider.value() / 100.0
        }

    def update_sidebar_rag_status(self, text: str, color_hex: str):
        """Updates the RAG status text and color in the RAG panel of the sidebar."""
        self.rag_sidebar_status_text.setText(text)
        self.rag_sidebar_status_text.setStyleSheet(f"color: {color_hex}; font-size: 9px;")


class AvAMainWindow(QMainWindow):
    workflow_requested = Signal(str)
    new_project_requested = Signal()

    def __init__(self, ava_app=None, config=None):
        super().__init__()
        self.ava_app = ava_app
        self.config = config

        self.setWindowTitle("AvA - AI Development Assistant")
        self.setGeometry(100, 100, 1400, 900)
        self._apply_theme()
        self._init_ui()  # This now uses the internal AvALeftSidebar by default
        self._connect_signals()

        if self.ava_app:
            self.ava_app.rag_status_changed.connect(self.update_rag_status_display)
            self._update_initial_ui_status()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #1e1e1e;
                color: #cccccc;
            }
        """)

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = AvALeftSidebar()  # Uses the AvALeftSidebar defined in THIS file
        self.chat_interface = ChatInterface()

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.chat_interface, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        self.chat_interface.message_sent.connect(self.workflow_requested)

        # Connect actions from the INTERNAL sidebar
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)
        # Connect "New Project" button from the INTERNAL sidebar
        self.sidebar.new_project_sidebar_action.connect(self.new_project_requested.emit)

        self.sidebar.temperature_changed.connect(self._on_temperature_changed)
        self.sidebar.model_changed.connect(self._on_model_changed)

    def _update_initial_ui_status(self):
        if not self.ava_app: return

        chat_model_name = self.ava_app.current_config.get("chat_model", "Default LLM")
        self.chat_interface.update_llm_status(f"LLM: {chat_model_name.split(':')[-1].strip()}", "ready")

        rag_info = self.ava_app.get_status().get("rag", {})
        rag_text = rag_info.get("status_text", "RAG: Unknown")

        # Determine color based on RAG status text more reliably
        rag_color = "#888888"  # Default grey
        if rag_info.get("ready"):
            rag_color = "#4ade80"  # Green for ready
        elif "Initializing" in rag_text or "loading" in rag_text.lower():
            rag_color = "#ffb900"  # Amber
        elif "Error" in rag_text or "Fail" in rag_text or "Missing" in rag_text:
            rag_color = "#ef4444"  # Red

        self.update_rag_status_display(rag_text, rag_color)
        self.sidebar.update_sidebar_rag_status(rag_text, rag_color)  # Update sidebar's RAG status too

        self.update_project_display(self.ava_app.current_project)

    def _on_temperature_changed(self, temp: float):
        if self.ava_app:
            self.ava_app.update_configuration({"temperature": temp})
            # Update LLM status to reflect change, or just keep it generic
            current_llm_name = self.ava_app.current_config.get("chat_model", "LLM").split(':')[-1].strip()
            self.chat_interface.update_llm_status(f"LLM: {current_llm_name} (Temp: {temp:.2f})", "ready")

    def _on_model_changed(self, model_type: str, model_name: str):
        if self.ava_app:
            new_config = {}
            if model_type == "chat":
                new_config["chat_model"] = model_name
            elif model_type == "code":
                new_config["code_model"] = model_name
            self.ava_app.update_configuration(new_config)

            # Update status bar to show the primary chat model
            chat_model_display_name = self.ava_app.current_config.get("chat_model", model_name).split(':')[-1].strip()
            current_temp = self.ava_app.current_config.get("temperature", 0.7)  # Get current temp
            self.chat_interface.update_llm_status(f"LLM: {chat_model_display_name} (Temp: {current_temp:.2f})", "ready")

    def _handle_sidebar_action(self, action: str):
        if self.ava_app:
            if action == "open_terminal" or action == "view_log":
                self.ava_app._open_terminal()
            elif action == "open_code_viewer" or action == "view_code":
                self.ava_app._open_code_viewer()
            elif action == "new_session":
                self.chat_interface.chat_display.clear()
                self.chat_interface.chat_display.add_assistant_message("New session started! How can I help you today?")
                self.ava_app.current_session = "New Session"
                self.update_project_display(self.ava_app.current_project)
            elif action == "scan_directory":
                if self.ava_app.rag_manager:
                    self.ava_app.rag_manager.scan_directory_dialog(parent_widget=self)
                else:
                    self.chat_interface.add_assistant_message("RAG Manager not available for scanning.")
            elif action == "add_files":
                if self.ava_app.rag_manager:
                    self.ava_app.rag_manager.add_files_dialog(parent_widget=self)
                else:
                    self.chat_interface.add_assistant_message("RAG Manager not available for adding files.")
            elif action == "force_gen":
                self.chat_interface.add_assistant_message("Force code generation triggered (logic to be implemented).")
            elif action == "check_updates":
                self.chat_interface.add_assistant_message("Checking for updates (feature not yet implemented).")
        else:
            print(f"AvAApp not available to handle action: {action}")

    @Slot(str)
    def on_workflow_started(self, prompt: str, metadata=None):
        self.chat_interface.update_llm_status(f"Working on: {prompt[:30]}...", "working")

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        success = result.get("success", False)
        chat_model_name = self.ava_app.current_config.get("chat_model", "LLM").split(':')[-1].strip()
        current_temp = self.ava_app.current_config.get("temperature", 0.7)

        if success:
            project_name = result.get("project_name", "your project")
            num_files = result.get("file_count", 0)
            message = f"✅ Workflow for '{project_name}' completed! Generated {num_files} files. View them in the Code Viewer."
            self.chat_interface.add_assistant_message(message)
            self.chat_interface.update_llm_status(f"LLM: {chat_model_name} (Temp: {current_temp:.2f})", "success")
        else:
            error_msg = result.get("error", "An unknown error occurred.")
            self.chat_interface.add_assistant_message(f"❌ Workflow failed: {error_msg}")
            self.chat_interface.update_llm_status(f"LLM: {chat_model_name} (Temp: {current_temp:.2f})", "error")

    @Slot(str, str)
    def on_app_error_occurred(self, component: str, error_message: str):
        self.chat_interface.add_assistant_message(f"⚠️ Error in {component}: {error_message}")
        # Potentially update a more general app status indicator if you add one
        self.chat_interface.update_llm_status(f"Error in {component}", "error")

    @Slot(str, str)
    def update_rag_status_display(self, status_text: str, color_or_key: str):
        self.chat_interface.update_rag_ui_status(status_text, color_or_key)
        # Also update the RAG status in the sidebar RAG panel
        if hasattr(self.sidebar, 'update_sidebar_rag_status'):
            self.sidebar.update_sidebar_rag_status(status_text, color_or_key)

    def update_project_display(self, project_name: str):
        session_name = self.ava_app.current_session if self.ava_app else "N/A"
        # LLM info for title is now fetched from current_config for consistency
        chat_model_from_config = "N/A"
        if self.ava_app and self.ava_app.current_config:
            chat_model_from_config = self.ava_app.current_config.get("chat_model", "N/A").split(':')[-1].strip()

        self.setWindowTitle(f"AvA [{project_name}] - Session: {session_name} (LLM: {chat_model_from_config})")

        # Update the static label in the sidebar for the current project
        if hasattr(self.sidebar, 'default_project_label'):
            self.sidebar.default_project_label.setText(project_name)