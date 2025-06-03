"""gui/model_config_dialog.py - CLEAN WORKING VERSION - NO ERRORS"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QPushButton, QFrame, QMessageBox,
    QTextEdit, QScrollArea, QWidget, QListWidget,
    QListWidgetItem, QInputDialog
)

from gui.components import StatusIndicator, ModernButton


# Simple role constants
class LLMRole:
    PLANNER = "planner"
    CODER = "coder"
    ASSEMBLER = "assembler"
    REVIEWER = "reviewer"
    CHAT = "chat"


@dataclass
class PersonalityPreset:
    """Individual personality preset for a specific role"""
    name: str
    description: str
    personality: str
    temperature: float
    role: str
    author: str = "User"
    created_date: str = None

    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()


class PersonalityManager:
    """Simple Personality Manager - Saves/Loads Individual Role Personalities"""

    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.presets_file = self.config_dir / "personality_presets.json"
        self.individual_presets = self._load_individual_presets()
        self._create_builtin_presets()

    def _load_individual_presets(self) -> Dict[str, List[PersonalityPreset]]:
        """Load individual presets grouped by role"""
        if not self.presets_file.exists():
            return {}

        try:
            with open(self.presets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            presets = {}
            for role, preset_list in data.items():
                presets[role] = [PersonalityPreset(**preset) for preset in preset_list]

            return presets
        except Exception:
            return {}

    def _save_individual_presets(self):
        """Save individual presets to file"""
        try:
            data = {}
            for role, preset_list in self.individual_presets.items():
                data[role] = [asdict(preset) for preset in preset_list]

            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _create_builtin_presets(self):
        """Create built-in personality presets"""
        builtin_presets = {
            LLMRole.PLANNER: [
                PersonalityPreset(
                    name="Strategic Architect",
                    description="Senior software architect focused on scalable solutions",
                    personality="You are a senior software architect with 15+ years of experience. You think strategically, break down complex problems into clear steps, and always consider scalability, maintainability, and best practices.",
                    temperature=0.7,
                    role=LLMRole.PLANNER,
                    author="AvA Built-in"
                )
            ],
            LLMRole.CODER: [
                PersonalityPreset(
                    name="Clean Code Expert",
                    description="Focused on clean, maintainable code",
                    personality="You are a coding specialist who writes clean, efficient, and well-documented code. You follow best practices, use proper error handling, and write code that is both functional and elegant.",
                    temperature=0.1,
                    role=LLMRole.CODER,
                    author="AvA Built-in"
                )
            ],
            LLMRole.ASSEMBLER: [
                PersonalityPreset(
                    name="Integration Expert",
                    description="Meticulous code integrator",
                    personality="You are a meticulous code integrator who ensures all pieces work together seamlessly. You create professional, production-ready files with proper organization, imports, and documentation.",
                    temperature=0.3,
                    role=LLMRole.ASSEMBLER,
                    author="AvA Built-in"
                )
            ]
        }

        # Add built-in presets if they don't exist
        for role, presets in builtin_presets.items():
            if role not in self.individual_presets:
                self.individual_presets[role] = []

            existing_names = {p.name for p in self.individual_presets[role] if p.author == "AvA Built-in"}

            for preset in presets:
                if preset.name not in existing_names:
                    self.individual_presets[role].append(preset)

        self._save_individual_presets()

    def get_individual_presets_for_role(self, role: str) -> List[PersonalityPreset]:
        """Get all presets for a specific role"""
        return self.individual_presets.get(role, [])

    def save_individual_preset(self, role: str, name: str, description: str,
                               personality: str, temperature: float) -> bool:
        """Save an individual personality preset"""
        try:
            if role not in self.individual_presets:
                self.individual_presets[role] = []

            # Check if preset with same name exists
            existing_names = {p.name for p in self.individual_presets[role]}
            if name in existing_names:
                # Update existing preset
                for i, preset in enumerate(self.individual_presets[role]):
                    if preset.name == name:
                        self.individual_presets[role][i] = PersonalityPreset(
                            name=name,
                            description=description,
                            personality=personality,
                            temperature=temperature,
                            role=role
                        )
                        break
            else:
                # Add new preset
                preset = PersonalityPreset(
                    name=name,
                    description=description,
                    personality=personality,
                    temperature=temperature,
                    role=role
                )
                self.individual_presets[role].append(preset)

            self._save_individual_presets()
            return True
        except Exception:
            return False

    def delete_individual_preset(self, role: str, name: str) -> bool:
        """Delete an individual preset"""
        try:
            if role in self.individual_presets:
                self.individual_presets[role] = [
                    p for p in self.individual_presets[role]
                    if p.name != name or p.author == "AvA Built-in"
                ]
                self._save_individual_presets()
                return True
        except Exception:
            pass
        return False


class PersonalityPresetWidget(QFrame):
    """Widget for managing personality presets"""

    preset_selected = Signal(str)

    def __init__(self, role_name: str, role_enum: str, personality_manager: PersonalityManager, parent_editor):
        super().__init__()
        self.role_name = role_name
        self.role_enum = role_enum
        self.personality_manager = personality_manager
        self.parent_editor = parent_editor

        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            PersonalityPresetWidget {
                background: #1a1a1a;
                border: 1px solid #404040;
                border-radius: 6px;
                margin: 2px;
            }
        """)
        self._init_ui()
        self._refresh_presets()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        header = QLabel(f"🎭 {self.role_name} Presets")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(header)

        # Preset list
        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(100)
        self.preset_list.setStyleSheet("""
            QListWidget {
                background: #252526;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #cccccc;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background: #00d7ff;
                color: #1e1e1e;
            }
            QListWidget::item:hover {
                background: #3e3e42;
            }
        """)
        self.preset_list.itemClicked.connect(self._on_preset_selected)
        layout.addWidget(self.preset_list)

        # Buttons
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("💾")
        self.save_btn.setToolTip("Save current personality as preset")
        self.save_btn.setMaximumWidth(30)
        self.save_btn.clicked.connect(self._save_preset)

        self.delete_btn = QPushButton("🗑️")
        self.delete_btn.setToolTip("Delete selected preset")
        self.delete_btn.setMaximumWidth(30)
        self.delete_btn.clicked.connect(self._delete_preset)

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addStretch()

        for btn in [self.save_btn, self.delete_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #2d2d30;
                    color: #cccccc;
                    border: 1px solid #404040;
                    border-radius: 3px;
                    padding: 4px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: #3e3e42;
                    border-color: #00d7ff;
                }
            """)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _refresh_presets(self):
        """Refresh the preset list"""
        self.preset_list.clear()
        presets = self.personality_manager.get_individual_presets_for_role(self.role_enum)

        for preset in presets:
            item = QListWidgetItem(f"{preset.name}")
            item.setToolTip(f"{preset.description}\nTemp: {preset.temperature}")
            item.setData(Qt.ItemDataRole.UserRole, preset)

            if preset.author == "AvA Built-in":
                item.setText(f"⭐ {preset.name}")

            self.preset_list.addItem(item)

    def _on_preset_selected(self, item):
        """Handle preset selection"""
        preset = item.data(Qt.ItemDataRole.UserRole)
        if preset:
            self.preset_selected.emit(preset.personality)

    def _save_preset(self):
        """Save current personality as preset"""
        name, ok = QInputDialog.getText(
            self, "Save Personality Preset",
            f"Enter name for {self.role_name} personality preset:"
        )

        if ok and name:
            description, ok2 = QInputDialog.getText(
                self, "Preset Description",
                "Enter description (optional):"
            )

            if ok2:
                current_personality = self.parent_editor.get_personality()
                current_temp = self.parent_editor.get_current_temperature()

                success = self.personality_manager.save_individual_preset(
                    role=self.role_enum,
                    name=name,
                    description=description or f"Custom {self.role_name} personality",
                    personality=current_personality,
                    temperature=current_temp
                )

                if success:
                    self._refresh_presets()
                    QMessageBox.information(self, "Success", f"Preset '{name}' saved!")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save preset.")

    def _delete_preset(self):
        """Delete selected preset"""
        current_item = self.preset_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select a preset to delete.")
            return

        preset = current_item.data(Qt.ItemDataRole.UserRole)
        if preset.author == "AvA Built-in":
            QMessageBox.warning(self, "Cannot Delete", "Built-in presets cannot be deleted.")
            return

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete preset '{preset.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            success = self.personality_manager.delete_individual_preset(self.role_enum, preset.name)
            if success:
                self._refresh_presets()
                QMessageBox.information(self, "Success", f"Preset '{preset.name}' deleted!")


class PersonalityEditor(QFrame):
    """Enhanced personality editor with save/load functionality"""

    personality_changed = Signal()

    def __init__(self, role_name: str, role_enum: str, personality_manager: PersonalityManager, parent_section):
        super().__init__()
        self.role_name = role_name
        self.role_enum = role_enum
        self.personality_manager = personality_manager
        self.parent_section = parent_section

        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            PersonalityEditor {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin: 4px;
            }
        """)
        self._init_ui()

    def _init_ui(self):
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
        self.personality_text.setMaximumHeight(100)
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

        # Set default personality
        default_personality = self._get_default_personality()
        self.personality_text.setPlainText(default_personality)
        self.personality_text.textChanged.connect(self.personality_changed.emit)
        layout.addWidget(self.personality_text)

        # Preset widget
        self.preset_widget = PersonalityPresetWidget(self.role_name, self.role_enum, self.personality_manager, self)
        self.preset_widget.preset_selected.connect(self.set_personality)
        layout.addWidget(self.preset_widget)

        self.setLayout(layout)

    def _get_default_personality(self) -> str:
        """Get default personality based on role"""
        personalities = {
            "Planner": "You are a senior software architect with 15+ years of experience. You think strategically, break down complex problems into clear steps, and always consider scalability and maintainability.",
            "Coder": "You are a coding specialist who writes clean, efficient, and well-documented code. You follow best practices, use proper error handling, and write code that is both functional and elegant.",
            "Assembler": "You are a meticulous code integrator who ensures all pieces work together seamlessly. You create professional, production-ready files with proper organization and documentation."
        }
        return personalities.get(self.role_name.replace(" & Reviewer", "").replace("Code ", ""),
                                 "You are a helpful AI assistant focused on producing high-quality results.")

    def set_personality(self, personality: str):
        """Set personality text"""
        self.personality_text.setPlainText(personality)

    def get_personality(self) -> str:
        """Get current personality text"""
        return self.personality_text.toPlainText().strip()

    def get_current_temperature(self) -> float:
        """Get temperature from parent section"""
        if hasattr(self.parent_section, 'get_temperature'):
            return self.parent_section.get_temperature()
        return 0.7


class RoleSection(QFrame):
    """Custom QFrame with role-specific methods"""

    def __init__(self, role_name: str, role_enum: str, default_temp: float, personality_manager: PersonalityManager):
        super().__init__()
        self.role_name = role_name
        self.role_enum = role_enum
        self.default_temp = default_temp
        self.personality_manager = personality_manager

        # Initialize widgets
        self.model_combo = None
        self.temp_slider = None
        self.temp_value = None
        self.personality_editor = None
        self.status_indicator = None

        self._init_ui()

    def _init_ui(self):
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            RoleSection {
                background: #252526;
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 6px;
                padding: 8px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header_layout = QHBoxLayout()
        role_label = QLabel(self.role_name)
        role_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        role_label.setStyleSheet("color: #00d7ff;")

        self.status_indicator = StatusIndicator("offline")

        header_layout.addWidget(role_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_indicator)
        layout.addLayout(header_layout)

        # Model and temperature controls
        controls_layout = QHBoxLayout()

        # Model selection
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 8px;
                color: #cccccc;
            }
        """)

        # Temperature slider
        temp_label = QLabel("Temp:")
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(int(self.default_temp * 100))
        self.temp_value = QLabel(f"{self.default_temp:.2f}")

        self.temp_slider.valueChanged.connect(lambda v: self.temp_value.setText(f"{v / 100:.2f}"))

        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_combo, 2)
        controls_layout.addWidget(temp_label)
        controls_layout.addWidget(self.temp_slider, 1)
        controls_layout.addWidget(self.temp_value)

        layout.addLayout(controls_layout)

        # Enhanced personality editor
        clean_role_name = self.role_name.replace("🧠 ", "").replace("⚙️ ", "").replace("📄 ", "")
        self.personality_editor = PersonalityEditor(
            clean_role_name,
            self.role_enum,
            self.personality_manager,
            self
        )
        layout.addWidget(self.personality_editor)

        self.setLayout(layout)

    def get_selected_model(self):
        return self.model_combo.currentData() if self.model_combo else None

    def get_temperature(self):
        return self.temp_slider.value() / 100.0 if self.temp_slider else self.default_temp

    def get_personality(self):
        return self.personality_editor.get_personality() if self.personality_editor else ""

    def set_temperature(self, temp: float):
        if self.temp_slider:
            self.temp_slider.setValue(int(temp * 100))

    def set_personality(self, personality: str):
        if self.personality_editor:
            self.personality_editor.set_personality(personality)


class ModelConfigurationDialog(QDialog):
    """CLEAN Model Configuration Dialog with Working Personality Save/Load"""

    configuration_applied = Signal(dict)

    def __init__(self, llm_client=None, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.personality_manager = PersonalityManager()

        self.setWindowTitle("AvA - AI Specialist Configuration")
        self.setModal(True)
        self.setFixedSize(800, 700)

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

        subtitle = QLabel("Configure models, temperatures, and personalities with save/load presets")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setStyleSheet("color: #888; margin-bottom: 20px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Scrollable content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout()

        # Create role sections
        self.planner_section = RoleSection("🧠 Planner & Reviewer", LLMRole.PLANNER, 0.7, self.personality_manager)
        self.coder_section = RoleSection("⚙️ Code Generator", LLMRole.CODER, 0.1, self.personality_manager)
        self.assembler_section = RoleSection("📄 Code Assembler", LLMRole.ASSEMBLER, 0.3, self.personality_manager)

        content_layout.addWidget(self.planner_section)
        content_layout.addWidget(self.coder_section)
        content_layout.addWidget(self.assembler_section)

        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)

        layout.addWidget(scroll_area, 1)

        # Button bar
        button_layout = QHBoxLayout()

        self.test_button = ModernButton("🔍 Test Models", button_type="secondary")
        self.test_button.clicked.connect(self._test_models)

        self.reset_button = ModernButton("🔄 Reset", button_type="secondary")
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

        available_models = {}
        if hasattr(self.llm_client, 'models'):
            for model_key, model_config in self.llm_client.models.items():
                display_name = f"{model_config.provider}/{model_config.model}"
                available_models[model_key] = display_name

        # Populate each section
        for section in [self.planner_section, self.coder_section, self.assembler_section]:
            section.model_combo.clear()
            for model_key, display_name in available_models.items():
                section.model_combo.addItem(display_name, model_key)

            if available_models:
                section.status_indicator.update_status("ready")
            else:
                section.status_indicator.update_status("error")

    def _load_current_config(self):
        """Load current configuration from LLM client"""
        if not self.llm_client or not hasattr(self.llm_client, 'role_assignments'):
            return

        assignments = self.llm_client.role_assignments
        sections = {
            LLMRole.PLANNER: self.planner_section,
            LLMRole.CODER: self.coder_section,
            LLMRole.ASSEMBLER: self.assembler_section
        }

        for role, section in sections.items():
            if role in assignments:
                model_key = assignments[role]
                for i in range(section.model_combo.count()):
                    if section.model_combo.itemData(i) == model_key:
                        section.model_combo.setCurrentIndex(i)
                        break

        if hasattr(self.llm_client, 'personalities'):
            for role, personality in self.llm_client.personalities.items():
                if role in sections:
                    sections[role].set_personality(personality)

    def _reset_to_defaults(self):
        """Reset all personalities to defaults"""
        reply = QMessageBox.question(
            self, "Reset to Defaults",
            "Reset all personalities and temperatures to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.planner_section.personality_editor.set_personality(
                self.planner_section.personality_editor._get_default_personality()
            )
            self.planner_section.set_temperature(0.7)

            self.coder_section.personality_editor.set_personality(
                self.coder_section.personality_editor._get_default_personality()
            )
            self.coder_section.set_temperature(0.1)

            self.assembler_section.personality_editor.set_personality(
                self.assembler_section.personality_editor._get_default_personality()
            )
            self.assembler_section.set_temperature(0.3)

    def _test_models(self):
        """Test connectivity to all selected models"""
        QMessageBox.information(
            self,
            "Model Testing",
            "🔥 Model testing will send a quick test prompt to each AI specialist.\n\n" +
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
                LLMRole.REVIEWER: self.planner_section.get_selected_model(),
            },
            'temperatures': {
                LLMRole.PLANNER: self.planner_section.get_temperature(),
                LLMRole.CODER: self.coder_section.get_temperature(),
                LLMRole.ASSEMBLER: self.assembler_section.get_temperature(),
                LLMRole.REVIEWER: self.planner_section.get_temperature(),
            },
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

            # Store personalities in LLM client
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
                "✅ Individual AI personalities loaded\n" +
                "✅ Model assignments updated\n" +
                "✅ Temperature settings applied\n" +
                "✅ Personality save/load system ready\n\n" +
                "Your AI specialists are now ready with their optimized configurations!"
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