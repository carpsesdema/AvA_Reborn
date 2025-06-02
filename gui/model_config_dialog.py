# gui/model_config_dialog.py - Clean Model Configuration Dialog

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QPushButton, QFrame, QGroupBox, QMessageBox
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


class RoleConfigSection(QFrame):
    """Configuration section for a single AI role"""

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
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin: 4px;
                padding: 8px;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header with role name and status
        header_layout = QHBoxLayout()

        role_label = QLabel(self.role_name)
        role_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        role_label.setStyleSheet("color: #00d7ff; background: transparent;")

        self.status_indicator = StatusIndicator("offline")

        header_layout.addWidget(role_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_indicator)

        layout.addLayout(header_layout)

        # Model selection
        model_layout = QHBoxLayout()

        model_label = QLabel("Model:")
        model_label.setMinimumWidth(60)
        model_label.setStyleSheet("color: #cccccc; font-size: 10px;")

        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px 10px;
                color: #cccccc;
                font-size: 10px;
                min-height: 24px;
            }
            QComboBox:hover {
                border-color: #00d7ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #cccccc;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background: #2d2d30;
                border: 1px solid #00d7ff;
                selection-background-color: #00d7ff;
                selection-color: #1e1e1e;
                color: #cccccc;
                outline: none;
            }
        """)
        self.model_combo.currentTextChanged.connect(self.config_changed.emit)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)

        layout.addLayout(model_layout)

        # Temperature slider
        temp_header = QHBoxLayout()

        temp_label = QLabel("Temperature:")
        temp_label.setStyleSheet("color: #cccccc; font-size: 10px;")

        self.temp_value = QLabel(f"{self.default_temp:.2f}")
        self.temp_value.setStyleSheet("color: #00d7ff; font-weight: bold; font-size: 10px;")

        temp_header.addWidget(temp_label)
        temp_header.addStretch()
        temp_header.addWidget(self.temp_value)

        layout.addLayout(temp_header)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(int(self.default_temp * 100))
        self.temp_slider.setFixedHeight(24)
        self.temp_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #404040;
                height: 4px;
                background: #1e1e1e;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00d7ff;
                border: 2px solid #00d7ff;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #40e0ff;
                border-color: #40e0ff;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 2px;
            }
        """)
        self.temp_slider.valueChanged.connect(self._on_temperature_changed)
        self.temp_slider.valueChanged.connect(self.config_changed.emit)

        layout.addWidget(self.temp_slider)

        self.setLayout(layout)

    def _on_temperature_changed(self, value):
        temp_val = value / 100.0
        self.temp_value.setText(f"{temp_val:.2f}")

    def populate_models(self, available_models: dict):
        """Populate the model dropdown with available models for this role"""
        self.model_combo.clear()

        suitable_models = []
        for model_key, model_info in available_models.items():
            # Model info format: "provider/model_name"
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


class ModelConfigurationDialog(QDialog):
    """Professional Model Configuration Dialog"""

    configuration_applied = Signal(dict)  # Emits new configuration

    def __init__(self, llm_client=None, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client

        self.setWindowTitle("AvA - Model Configuration")
        self.setModal(True)
        self.setFixedSize(600, 500)

        self._init_ui()
        self._apply_theme()
        self._populate_models()
        self._load_current_config()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Dialog header
        header_label = QLabel("AI Specialist Model Configuration")
        header_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header_label.setStyleSheet("color: #00d7ff; margin-bottom: 8px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        subtitle = QLabel("Configure specialized models for each AI role")
        subtitle.setStyleSheet("color: #888; font-size: 11px; margin-bottom: 16px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Role configuration sections
        self.planner_section = RoleConfigSection("🧠 Planner & Reviewer", LLMRole.PLANNER, 0.7)
        self.coder_section = RoleConfigSection("⚙️ Code Generator", LLMRole.CODER, 0.1)
        self.assembler_section = RoleConfigSection("📄 Code Assembler", LLMRole.ASSEMBLER, 0.3)

        layout.addWidget(self.planner_section)
        layout.addWidget(self.coder_section)
        layout.addWidget(self.assembler_section)

        layout.addStretch()

        # Button bar
        button_layout = QHBoxLayout()

        self.test_button = ModernButton("🔍 Test All Models", button_type="secondary")
        self.test_button.clicked.connect(self._test_all_models)

        self.cancel_button = ModernButton("Cancel", button_type="secondary")
        self.cancel_button.clicked.connect(self.reject)

        self.apply_button = ModernButton("Apply Configuration", button_type="primary")
        self.apply_button.clicked.connect(self._apply_configuration)

        button_layout.addWidget(self.test_button)
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
                background: #1e1e1e;
                color: #cccccc;
                border: 2px solid #00d7ff;
                border-radius: 12px;
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

    def _test_all_models(self):
        """Test connectivity to all selected models"""
        # Placeholder for model testing
        # In a real implementation, this would attempt to make test calls
        QMessageBox.information(
            self,
            "Model Testing",
            "Model testing functionality will be implemented in a future update.\n\n"
            "For now, green indicators show models with valid API keys."
        )

    def _apply_configuration(self):
        """Apply the new configuration"""
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

            # Emit signal with new configuration
            self.configuration_applied.emit(new_config)

            QMessageBox.information(
                self,
                "Configuration Applied",
                "Model configuration has been successfully applied!\n\n"
                "Your AI specialists are now ready with their optimized models."
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
        }