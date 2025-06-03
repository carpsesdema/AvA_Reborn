# gui/model_config_dialog.py - ENHANCED with Individual AI Personalities 🔥

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QPushButton, QFrame, QGroupBox, QMessageBox,
    QTextEdit, QScrollArea, QWidget, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from gui.components import StatusIndicator, ModernButton

try:
    from core.llm_client import LLMRole
except ImportError:
    class LLMRole:
        PLANNER = "planner"
        CODER = "coder"
        ASSEMBLER = "assembler"
        REVIEWER = "reviewer"
        CHAT = "chat"


class PersonalityEditor(QFrame):
    """🎭 Individual AI Personality Editor"""

    personality_changed = Signal()

    def __init__(self, role_name: str, default_personality: str = ""):
        super().__init__()
        self.role_name = role_name
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            PersonalityEditor {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin: 4px;
            }
        """)
        self._init_ui(default_personality)

    def _init_ui(self, default_personality: str):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel(f"🎭 {self.role_name} Personality")
        header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(header)

        # Personality text editor
        self.personality_text = QTextEdit()
        self.personality_text.setMaximumHeight(120)
        self.personality_text.setStyleSheet("""
            QTextEdit {
                background: #252526;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                font-family: "Segoe UI";
                font-size: 11px;
            }
        """)

        # Set default personality based on role
        if not default_personality:
            default_personality = self._get_default_personality()

        self.personality_text.setPlainText(default_personality)
        self.personality_text.textChanged.connect(self.personality_changed.emit)
        layout.addWidget(self.personality_text)

        # Quick personality buttons
        button_layout = QHBoxLayout()

        personalities = self._get_personality_presets()
        for name, personality in personalities.items():
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton {
                    background: #2d2d30;
                    color: #cccccc;
                    border: 1px solid #404040;
                    border-radius: 3px;
                    padding: 4px 8px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: #3e3e42;
                    border-color: #00d7ff;
                }
            """)
            btn.clicked.connect(lambda checked, p=personality: self.set_personality(p))
            button_layout.addWidget(btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _get_default_personality(self) -> str:
        """Get default personality based on role"""
        personalities = {
            "Planner": "You are a senior software architect with 15+ years of experience. You think strategically, break down complex problems into clear steps, and always consider scalability and maintainability. You communicate clearly and provide detailed technical specifications.",

            "Coder": "You are a coding specialist who writes clean, efficient, and well-documented code. You follow best practices, use proper error handling, and write code that is both functional and elegant. You focus on getting things done with high quality.",

            "Assembler": "You are a meticulous code integrator who ensures all pieces work together seamlessly. You have an eye for detail, maintain consistent code style, and create professional, production-ready files with proper organization and documentation."
        }
        return personalities.get(self.role_name,
                                 "You are a helpful AI assistant focused on producing high-quality results.")

    def _get_personality_presets(self) -> dict:
        """Get personality presets for quick selection"""
        if self.role_name == "Planner":
            return {
                "Strategic": "You are a strategic software architect who thinks big-picture and focuses on scalable, maintainable solutions.",
                "Detailed": "You are a meticulous planner who creates comprehensive specifications with thorough documentation.",
                "Agile": "You are an agile-focused architect who prioritizes MVP and iterative development approaches."
            }
        elif self.role_name == "Coder":
            return {
                "Clean": "You write exceptionally clean, readable code with excellent documentation and error handling.",
                "Performance": "You focus on writing high-performance, optimized code that runs efficiently and scales well.",
                "Creative": "You write innovative, elegant solutions using modern patterns and creative problem-solving approaches."
            }
        elif self.role_name == "Assembler":
            return {
                "Professional": "You create polished, enterprise-grade code with perfect organization and comprehensive documentation.",
                "Minimalist": "You focus on clean, minimal code that does exactly what's needed without unnecessary complexity.",
                "Robust": "You prioritize error handling, edge cases, and creating bulletproof code that handles all scenarios."
            }
        else:
            return {
                "Helpful": "You are helpful and focused on providing excellent results.",
                "Professional": "You maintain a professional approach and deliver high-quality output.",
                "Detailed": "You provide thorough, detailed responses with comprehensive information."
            }

    def set_personality(self, personality: str):
        """Set personality text"""
        self.personality_text.setPlainText(personality)

    def get_personality(self) -> str:
        """Get current personality text"""
        return self.personality_text.toPlainText().strip()


class RoleConfigSection(QFrame):
    """Configuration section for a single AI role with personality"""

    config_changed = Signal()

    def __init__(self, role_name: str, role_enum, default_temp: float = 0.7):
        super().__init__()
        self.role_name = role_name
        self.role_enum = role_enum
        self.default_temp = default_temp

        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            RoleConfigSection {
                background: #252526;
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 6px;
                padding: 8px;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header with role name and status
        header_layout = QHBoxLayout()

        role_label = QLabel(self.role_name)
        role_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        role_label.setStyleSheet("color: #00d7ff; background: transparent;")

        self.status_indicator = StatusIndicator("offline")

        header_layout.addWidget(role_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_indicator)

        layout.addLayout(header_layout)

        # Model selection row
        model_layout = QHBoxLayout()

        model_label = QLabel("Model:")
        model_label.setMinimumWidth(80)
        model_label.setFont(QFont("Segoe UI", 11))
        model_label.setStyleSheet("color: #cccccc; background: transparent;")

        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(35)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 8px 12px;
                color: #cccccc;
                font-size: 11px;
                font-weight: 500;
            }
            QComboBox:hover {
                border-color: #00d7ff;
            }
            QComboBox:focus {
                border-color: #40e0ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background: #2d2d30;
                border: 2px solid #00d7ff;
                border-radius: 6px;
                selection-background-color: #00d7ff;
                selection-color: #1e1e1e;
                color: #cccccc;
                outline: none;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px 12px;
                border-radius: 4px;
                margin: 2px;
            }
            QComboBox QAbstractItemView::item:hover {
                background: #3e3e42;
            }
        """)
        self.model_combo.currentTextChanged.connect(lambda: self.config_changed.emit())

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)

        layout.addLayout(model_layout)

        # Temperature slider section
        temp_header = QHBoxLayout()

        temp_label = QLabel("Temperature:")
        temp_label.setFont(QFont("Segoe UI", 11))
        temp_label.setStyleSheet("color: #cccccc; background: transparent;")

        self.temp_value = QLabel(f"{self.default_temp:.2f}")
        self.temp_value.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.temp_value.setStyleSheet("color: #00d7ff; background: transparent;")

        temp_header.addWidget(temp_label)
        temp_header.addStretch()
        temp_header.addWidget(self.temp_value)

        layout.addLayout(temp_header)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(int(self.default_temp * 100))
        self.temp_slider.setMinimumHeight(30)
        self.temp_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 2px solid #404040;
                height: 8px;
                background: #1e1e1e;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border: 2px solid #00d7ff;
                width: 20px;
                height: 20px;
                border-radius: 12px;
                margin: -8px 0;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #40e0ff, stop:1 #00a8ff);
                border-color: #40e0ff;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 6px;
            }
        """)
        self.temp_slider.valueChanged.connect(self._on_temperature_changed)
        self.temp_slider.valueChanged.connect(lambda: self.config_changed.emit())

        layout.addWidget(self.temp_slider)

        # 🎭 NEW: Personality Editor
        self.personality_editor = PersonalityEditor(self.role_name)
        self.personality_editor.personality_changed.connect(self.config_changed.emit)
        layout.addWidget(self.personality_editor)

        self.setLayout(layout)

    def _on_temperature_changed(self, value):
        temp_val = value / 100.0
        self.temp_value.setText(f"{temp_val:.2f}")

    def populate_models(self, available_models: dict):
        """Populate the model dropdown with available models for this role"""
        self.model_combo.clear()

        suitable_models = []
        for model_key, model_info in available_models.items():
            suitable_models.append((model_key, model_info))

        if suitable_models:
            for model_key, model_display in suitable_models:
                self.model_combo.addItem(model_display, model_key)
            self.status_indicator.update_status("ready")
        else:
            self.model_combo.addItem("No models available", None)
            self.status_indicator.update_status("error")

    def get_selected_model(self):
        """Get the currently selected model key"""
        return self.model_combo.currentData()

    def set_selected_model(self, model_key: str):
        """Set the selected model by key"""
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model_key:
                self.model_combo.setCurrentIndex(i)
                break

    def get_temperature(self):
        """Get the current temperature value"""
        return self.temp_slider.value() / 100.0

    def set_temperature(self, temp: float):
        """Set the temperature value"""
        self.temp_slider.setValue(int(temp * 100))

    def get_personality(self) -> str:
        """🎭 Get AI personality"""
        return self.personality_editor.get_personality()

    def set_personality(self, personality: str):
        """🎭 Set AI personality"""
        self.personality_editor.set_personality(personality)


class ModelConfigurationDialog(QDialog):
    """🔥 ENHANCED Model Configuration Dialog with Individual AI Personalities"""

    configuration_applied = Signal(dict)  # Emits new configuration

    def __init__(self, llm_client=None, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client

        self.setWindowTitle("AvA - AI Specialist Configuration")
        self.setModal(True)
        self.setFixedSize(800, 700)  # Larger for personality editors

        self._init_ui()
        self._apply_theme()
        self._populate_models()
        self._load_current_config()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Dialog header
        header_label = QLabel("🤖 AI Specialist Configuration")
        header_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_label.setStyleSheet("color: #00d7ff; margin-bottom: 8px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        subtitle = QLabel("Configure specialized models and personalities for each AI role")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setStyleSheet("color: #888; margin-bottom: 20px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #2d2d30;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #404040;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00d7ff;
            }
        """)

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(16)

        # Role configuration sections
        self.planner_section = RoleConfigSection("🧠 Planner & Reviewer", LLMRole.PLANNER, 0.7)
        self.coder_section = RoleConfigSection("⚙️ Code Generator", LLMRole.CODER, 0.1)
        self.assembler_section = RoleConfigSection("📄 Code Assembler", LLMRole.ASSEMBLER, 0.3)

        content_layout.addWidget(self.planner_section)
        content_layout.addWidget(self.coder_section)
        content_layout.addWidget(self.assembler_section)

        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)

        layout.addWidget(scroll_area, 1)

        # Button bar
        button_layout = QHBoxLayout()

        self.test_button = ModernButton("🔍 Test All Models", button_type="secondary")
        self.test_button.clicked.connect(self._test_all_models)

        self.reset_button = ModernButton("🔄 Reset to Defaults", button_type="secondary")
        self.reset_button.clicked.connect(self._reset_to_defaults)

        self.cancel_button = ModernButton("Cancel", button_type="secondary")
        self.cancel_button.clicked.connect(self.reject)

        self.apply_button = ModernButton("🚀 Apply Configuration", button_type="primary")
        self.apply_button.clicked.connect(self._apply_configuration)

        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.apply_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connect change signals
        self.planner_section.config_changed.connect(self._validate_configuration)
        self.coder_section.config_changed.connect(self._validate_configuration)
        self.assembler_section.config_changed.connect(self._validate_configuration)

    def _apply_theme(self):
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #252526);
                color: #cccccc;
                border: 3px solid #00d7ff;
                border-radius: 15px;
            }
        """)

    def _populate_models(self):
        """Populate model dropdowns with available models"""
        if not self.llm_client:
            return

        # Get all available models from the LLM client
        available_models = {}

        if hasattr(self.llm_client, 'models'):
            for model_key, model_config in self.llm_client.models.items():
                display_name = f"{model_config.provider}/{model_config.model}"
                available_models[model_key] = display_name

        # Populate each section
        self.planner_section.populate_models(available_models)
        self.coder_section.populate_models(available_models)
        self.assembler_section.populate_models(available_models)

    def _load_current_config(self):
        """Load current configuration from LLM client"""
        if not self.llm_client or not hasattr(self.llm_client, 'role_assignments'):
            return

        assignments = self.llm_client.role_assignments

        # Set current models
        if LLMRole.PLANNER in assignments:
            self.planner_section.set_selected_model(assignments[LLMRole.PLANNER])

        if LLMRole.CODER in assignments:
            self.coder_section.set_selected_model(assignments[LLMRole.CODER])

        if LLMRole.ASSEMBLER in assignments:
            self.assembler_section.set_selected_model(assignments[LLMRole.ASSEMBLER])

        # Set temperatures from model configs
        if hasattr(self.llm_client, 'models'):
            for role, model_key in assignments.items():
                if model_key in self.llm_client.models:
                    temp = self.llm_client.models[model_key].temperature

                    if role == LLMRole.PLANNER:
                        self.planner_section.set_temperature(temp)
                    elif role == LLMRole.CODER:
                        self.coder_section.set_temperature(temp)
                    elif role == LLMRole.ASSEMBLER:
                        self.assembler_section.set_temperature(temp)

    def _validate_configuration(self):
        """Validate current configuration and update UI"""
        valid = True

        # Check if all roles have models selected
        if not self.planner_section.get_selected_model():
            valid = False
        if not self.coder_section.get_selected_model():
            valid = False
        if not self.assembler_section.get_selected_model():
            valid = False

        self.apply_button.setEnabled(valid)

    def _reset_to_defaults(self):
        """Reset all personalities to defaults"""
        self.planner_section.personality_editor.set_personality(
            self.planner_section.personality_editor._get_default_personality()
        )
        self.coder_section.personality_editor.set_personality(
            self.coder_section.personality_editor._get_default_personality()
        )
        self.assembler_section.personality_editor.set_personality(
            self.assembler_section.personality_editor._get_default_personality()
        )

    def _test_all_models(self):
        """Test connectivity to all selected models"""
        QMessageBox.information(
            self,
            "Model Testing",
            "🔥 Model testing will send a quick test prompt to each AI specialist to verify they're working correctly.\n\n" +
            "This feature will be implemented in the next update!"
        )

    def _apply_configuration(self):
        """Apply the new configuration with personalities"""
        if not self.llm_client:
            self.reject()
            return

        # Build new configuration
        new_config = {
            'role_assignments': {
                LLMRole.PLANNER: self.planner_section.get_selected_model(),
                LLMRole.CODER: self.coder_section.get_selected_model(),
                LLMRole.ASSEMBLER: self.assembler_section.get_selected_model(),
                LLMRole.REVIEWER: self.planner_section.get_selected_model(),  # Same as planner
            },
            'temperatures': {
                LLMRole.PLANNER: self.planner_section.get_temperature(),
                LLMRole.CODER: self.coder_section.get_temperature(),
                LLMRole.ASSEMBLER: self.assembler_section.get_temperature(),
                LLMRole.REVIEWER: self.planner_section.get_temperature(),
            },
            # 🎭 NEW: Individual AI Personalities
            'personalities': {
                LLMRole.PLANNER: self.planner_section.get_personality(),
                LLMRole.CODER: self.coder_section.get_personality(),
                LLMRole.ASSEMBLER: self.assembler_section.get_personality(),
                LLMRole.REVIEWER: self.planner_section.get_personality(),
            }
        }

        # Apply to LLM client
        try:
            # Update role assignments
            for role, model_key in new_config['role_assignments'].items():
                if model_key:
                    self.llm_client.role_assignments[role] = model_key

            # Update temperatures in model configs
            for role, temp in new_config['temperatures'].items():
                model_key = new_config['role_assignments'].get(role)
                if model_key and model_key in self.llm_client.models:
                    self.llm_client.models[model_key].temperature = temp

            # 🎭 Store personalities in LLM client
            if not hasattr(self.llm_client, 'personalities'):
                self.llm_client.personalities = {}

            for role, personality in new_config['personalities'].items():
                self.llm_client.personalities[role] = personality

            # Emit signal with new configuration
            self.configuration_applied.emit(new_config)

            QMessageBox.information(
                self,
                "🚀 Configuration Applied",
                "Model configuration and AI personalities have been successfully applied!\n\n" +
                "Your AI specialists are now ready with their optimized models and unique personalities. " +
                "Each specialist will now work with their individual characteristics and approaches!"
            )

            self.accept()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Failed to apply configuration:\n{e}"
            )

    def get_configuration_summary(self):
        """Get a summary of the current configuration for display"""
        planner_model = self.planner_section.model_combo.currentText()
        coder_model = self.coder_section.model_combo.currentText()
        assembler_model = self.assembler_section.model_combo.currentText()

        return {
            'planner': planner_model,
            'coder': coder_model,
            'assembler': assembler_model,
            'planner_temp': self.planner_section.get_temperature(),
            'coder_temp': self.coder_section.get_temperature(),
            'assembler_temp': self.assembler_section.get_temperature(),
            'planner_personality': self.planner_section.get_personality()[:100] + "...",
            'coder_personality': self.coder_section.get_personality()[:100] + "...",
            'assembler_personality': self.assembler_section.get_personality()[:100] + "...",
        }