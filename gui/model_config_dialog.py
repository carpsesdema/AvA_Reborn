# gui/model_config_dialog.py - CLEAN WORKING VERSION - INCLUDES REVIEWER AND STRUCTURER ROLES

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
from core.llm_client import LLMRole  # IMPORT THE CORRECT ENUM


@dataclass
class PersonalityPreset:
    """Individual personality preset for a specific role"""
    name: str
    description: str
    personality: str
    temperature: float
    role: str  # Store role as string (e.g., "planner") internally for presets
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
        """Load individual presets grouped by role (string key)"""
        if not self.presets_file.exists():
            return {}

        try:
            with open(self.presets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            presets = {}
            for role_str, preset_list in data.items():  # role_str is "planner", "coder"
                presets[role_str] = [PersonalityPreset(**preset) for preset in preset_list]

            return presets
        except Exception as e:
            print(f"Error loading presets from {self.presets_file}: {e}")
            return {}

    def _save_individual_presets(self):
        """Save individual presets to file"""
        try:
            data = {}
            for role_str, preset_list in self.individual_presets.items():  # role_str is "planner", "coder"
                data[role_str] = [asdict(preset) for preset in preset_list]

            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving presets to {self.presets_file}: {e}")

    def _create_builtin_presets(self):
        """Create built-in personality presets"""
        builtin_presets_def = {
            LLMRole.STRUCTURER.value: [ # ADDED BUILT-IN FOR STRUCTURER
                PersonalityPreset(
                    name="Project Structurer",
                    description="Determines file structure and outputs simple JSON.",
                    personality="You are the STRUCTURER AI. Your only job is to determine a project's file structure based on a user request and output it as a simple JSON object.",
                    temperature=0.1, role=LLMRole.STRUCTURER.value, author="AvA Built-in"
                )
            ],
            LLMRole.PLANNER.value: [
                PersonalityPreset(
                    name="Strategic Architect",
                    description="Senior software architect focused on scalable solutions",
                    personality="You are a senior software architect with 15+ years of experience. You think strategically, break down complex problems into clear steps, and always consider scalability, maintainability, and best practices.",
                    temperature=0.7, role=LLMRole.PLANNER.value, author="AvA Built-in"
                )
            ],
            LLMRole.CODER.value: [
                PersonalityPreset(
                    name="Clean Code Expert",
                    description="Focused on clean, maintainable code",
                    personality="You are a coding specialist who writes clean, efficient, and well-documented code. You follow best practices, use proper error handling, and write code that is both functional and elegant.",
                    temperature=0.1, role=LLMRole.CODER.value, author="AvA Built-in"
                )
            ],
            LLMRole.ASSEMBLER.value: [
                PersonalityPreset(
                    name="Integration Expert",
                    description="Meticulous code integrator",
                    personality="You are a meticulous code integrator who ensures all pieces work together seamlessly. You create professional, production-ready files with proper organization, imports, and documentation.",
                    temperature=0.3, role=LLMRole.ASSEMBLER.value, author="AvA Built-in"
                )
            ],
            LLMRole.REVIEWER.value: [
                PersonalityPreset(
                    name="Detail-Oriented Reviewer",
                    description="Focuses on code quality, best practices, security, and performance.",
                    personality="You are a detail-oriented code reviewer. Your primary goal is to ensure code quality, adherence to best practices, security, and performance. Provide constructive feedback and clear justifications for any issues found.",
                    temperature=0.5, role=LLMRole.REVIEWER.value, author="AvA Built-in"
                )
            ],
            LLMRole.CHAT.value: [
                PersonalityPreset(
                    name="AvA - Friendly Assistant",
                    description="Default friendly and helpful AI assistant personality.",
                    personality="You are AvA, a friendly and helpful AI development assistant. Engage in natural conversation and guide users through their development tasks. You have specialized AI agents: Planner, Coder, Assembler, and Reviewer to help build applications.",
                    temperature=0.7, role=LLMRole.CHAT.value, author="AvA Built-in"
                )
            ]
        }

        changed = False
        for role_str, presets_to_add in builtin_presets_def.items():
            if role_str not in self.individual_presets:
                self.individual_presets[role_str] = []
                changed = True

            existing_names = {p.name for p in self.individual_presets[role_str] if p.author == "AvA Built-in"}

            for preset in presets_to_add:
                if preset.name not in existing_names:
                    self.individual_presets[role_str].append(preset)
                    changed = True

        if changed:
            self._save_individual_presets()

    def get_individual_presets_for_role(self, role_str: str) -> List[PersonalityPreset]:
        return self.individual_presets.get(role_str, [])

    def save_individual_preset(self, role_str: str, name: str, description: str,
                               personality: str, temperature: float) -> bool:
        try:
            if role_str not in self.individual_presets:
                self.individual_presets[role_str] = []

            existing_idx = -1
            for i, preset in enumerate(self.individual_presets[role_str]):
                if preset.name == name:
                    if preset.author == "User":
                        existing_idx = i
                    else:
                        pass
                    break

            new_preset = PersonalityPreset(
                name=name, description=description, personality=personality,
                temperature=temperature, role=role_str, author="User"
            )

            if existing_idx != -1:
                self.individual_presets[role_str][existing_idx] = new_preset
            else:
                self.individual_presets[role_str].append(new_preset)

            self._save_individual_presets()
            return True
        except Exception as e:
            print(f"Error saving preset: {e}")
            return False

    def delete_individual_preset(self, role_str: str, name: str) -> bool:
        try:
            if role_str in self.individual_presets:
                initial_len = len(self.individual_presets[role_str])
                self.individual_presets[role_str] = [
                    p for p in self.individual_presets[role_str]
                    if not (p.name == name and p.author == "User")
                ]
                if len(self.individual_presets[role_str]) < initial_len:
                    self._save_individual_presets()
                    return True
        except Exception as e:
            print(f"Error deleting preset: {e}")
        return False


class PersonalityPresetWidget(QFrame):
    preset_selected = Signal(PersonalityPreset)

    def __init__(self, role_name_display: str, role_value_str: str, personality_manager: PersonalityManager,
                 parent_editor):
        super().__init__()
        self.role_name_display = role_name_display
        self.role_value_str = role_value_str
        self.personality_manager = personality_manager
        self.parent_editor = parent_editor
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet(
            "PersonalityPresetWidget { background: #1a1a1a; border: 1px solid #404040; border-radius: 6px; margin: 2px; }")
        self._init_ui()
        self._refresh_presets()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        header = QLabel(f"🎭 {self.role_name_display} Presets")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(header)
        self.preset_list = QListWidget()
        self.preset_list.setMaximumHeight(100)
        self.preset_list.setStyleSheet(
            "QListWidget { background: #252526; border: 1px solid #404040; border-radius: 4px; color: #cccccc; font-size: 10px; } QListWidget::item { padding: 4px 8px; border-radius: 2px; } QListWidget::item:selected { background: #00d7ff; color: #1e1e1e; } QListWidget::item:hover { background: #3e3e42; }")
        self.preset_list.itemClicked.connect(self._on_preset_selected)
        layout.addWidget(self.preset_list)
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾")
        self.save_btn.setToolTip("Save current personality as preset")
        self.delete_btn = QPushButton("🗑️")
        self.delete_btn.setToolTip("Delete selected preset (user presets only)")
        for btn in [self.save_btn, self.delete_btn]:
            btn.setMaximumWidth(30)
            btn.setStyleSheet(
                "QPushButton { background: #2d2d30; color: #cccccc; border: 1px solid #404040; border-radius: 3px; padding: 4px; font-size: 12px; } QPushButton:hover { background: #3e3e42; border-color: #00d7ff; }")
        self.save_btn.clicked.connect(self._save_preset)
        self.delete_btn.clicked.connect(self._delete_preset)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _refresh_presets(self):
        self.preset_list.clear()
        presets = self.personality_manager.get_individual_presets_for_role(self.role_value_str)
        for preset in presets:
            item_text = f"⭐ {preset.name}" if preset.author == "AvA Built-in" else preset.name
            item = QListWidgetItem(item_text)
            item.setToolTip(
                f"Role: {preset.role}\nAuthor: {preset.author}\nTemp: {preset.temperature:.2f}\n{preset.description}")
            item.setData(Qt.ItemDataRole.UserRole, preset)
            self.preset_list.addItem(item)

    def _on_preset_selected(self, item):
        preset = item.data(Qt.ItemDataRole.UserRole)
        if preset: self.preset_selected.emit(preset)

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", f"Preset name for {self.role_name_display}:")
        if ok and name:
            desc, ok2 = QInputDialog.getText(self, "Preset Description", "Description (optional):")
            if ok2:
                current_personality = self.parent_editor.get_personality()
                current_temp = self.parent_editor.get_current_temperature()
                if self.personality_manager.save_individual_preset(self.role_value_str, name,
                                                                   desc or f"Custom {self.role_name_display} personality",
                                                                   current_personality, current_temp):
                    self._refresh_presets()
                    QMessageBox.information(self, "Success", f"Preset '{name}' saved.")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save preset.")

    def _delete_preset(self):
        item = self.preset_list.currentItem()
        if not item: QMessageBox.information(self, "No Selection", "Select preset to delete."); return
        preset = item.data(Qt.ItemDataRole.UserRole)
        if preset.author == "AvA Built-in": QMessageBox.warning(self, "Cannot Delete",
                                                                "Built-in presets cannot be deleted."); return
        if QMessageBox.question(self, "Confirm Delete", f"Delete '{preset.name}'?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            if self.personality_manager.delete_individual_preset(self.role_value_str, preset.name):
                self._refresh_presets()
                QMessageBox.information(self, "Success", f"Preset '{preset.name}' deleted.")
            else:
                QMessageBox.warning(self, "Error", "Failed to delete preset or preset not found.")


class PersonalityEditor(QFrame):
    personality_changed = Signal()

    def __init__(self, role_name_display: str, role_value_str: str, personality_manager: PersonalityManager,
                 parent_section):
        super().__init__()
        self.role_name_display = role_name_display
        self.role_value_str = role_value_str
        self.personality_manager = personality_manager
        self.parent_section = parent_section
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet(
            "PersonalityEditor { background: #1e1e1e; border: 1px solid #3e3e42; border-radius: 6px; margin: 4px; }")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        header = QLabel(f"✏️ {self.role_name_display} Personality Text")
        header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(header)
        self.personality_text = QTextEdit()
        self.personality_text.setMinimumHeight(100)
        self.personality_text.setStyleSheet(
            "QTextEdit { background: #252526; color: #cccccc; border: 1px solid #404040; border-radius: 4px; padding: 8px; font-family: \"Segoe UI\"; font-size: 11px; }")
        self.personality_text.setPlainText(self._get_default_personality())
        self.personality_text.textChanged.connect(self.personality_changed.emit)
        layout.addWidget(self.personality_text)
        self.preset_widget = PersonalityPresetWidget(self.role_name_display, self.role_value_str,
                                                     self.personality_manager, self)
        self.preset_widget.preset_selected.connect(self._apply_preset)
        layout.addWidget(self.preset_widget)
        self.setLayout(layout)

    def _get_default_personality(self) -> str:
        defaults = {
            LLMRole.STRUCTURER.value: "You are the STRUCTURER AI...",
            LLMRole.PLANNER.value: "You are a senior software architect...",
            LLMRole.CODER.value: "You are a coding specialist...",
            LLMRole.ASSEMBLER.value: "You are a meticulous code integrator...",
            LLMRole.REVIEWER.value: "You are a detail-oriented code reviewer...",
            LLMRole.CHAT.value: "You are AvA, a friendly AI assistant..."
        }
        return defaults.get(self.role_value_str, "You are a helpful AI assistant.")

    def _apply_preset(self, preset: PersonalityPreset):
        self.personality_text.setPlainText(preset.personality)
        if hasattr(self.parent_section, 'set_temperature'):
            self.parent_section.set_temperature(preset.temperature)
        self.personality_changed.emit()

    def get_personality(self) -> str: return self.personality_text.toPlainText().strip()

    def set_personality(self, personality: str): self.personality_text.setPlainText(personality)

    def get_current_temperature(self) -> float:
        return self.parent_section.get_temperature() if hasattr(self.parent_section, 'get_temperature') else 0.7


class RoleSection(QFrame):
    def __init__(self, role_name_display: str, role_enum_member: LLMRole, default_temp: float,
                 personality_manager: PersonalityManager):
        super().__init__()
        self.role_name_display = role_name_display
        self.role_enum_member = role_enum_member
        self.default_temp = default_temp
        self.personality_manager = personality_manager
        self.model_combo = None
        self.temp_slider = None
        self.temp_value = None
        self.personality_editor = None
        self.status_indicator = None
        self._init_ui()

    def _init_ui(self):
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet(
            "RoleSection { background: #252526; border: 2px solid #00d7ff; border-radius: 8px; margin: 6px; padding: 8px; }")
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        header_layout = QHBoxLayout()
        role_label = QLabel(self.role_name_display)
        role_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        role_label.setStyleSheet("color: #00d7ff;")
        self.status_indicator = StatusIndicator("offline")
        header_layout.addWidget(role_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_indicator)
        layout.addLayout(header_layout)
        controls_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(
            "QComboBox { background: #1e1e1e; border: 2px solid #404040; border-radius: 6px; padding: 8px; color: #cccccc; }")
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

        editor_role_name = self.role_name_display.split(" ", 1)[
            1] if " " in self.role_name_display else self.role_name_display
        self.personality_editor = PersonalityEditor(
            editor_role_name,
            self.role_enum_member.value,
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
        if self.temp_slider: self.temp_slider.setValue(int(temp * 100))

    def set_personality(self, personality: str):
        if self.personality_editor: self.personality_editor.set_personality(personality)


class ModelConfigurationDialog(QDialog):
    configuration_applied = Signal(dict)

    def __init__(self, llm_client=None, parent=None):
        super().__init__(parent)
        self.llm_client = llm_client
        self.personality_manager = PersonalityManager()
        self.setWindowTitle("AvA - AI Specialist Configuration")
        self.setModal(True)
        self.setMinimumSize(800, 750)
        self._init_ui()
        self._apply_theme()
        self._populate_models()
        self._load_current_config()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(15)
        header_label = QLabel("🤖 AI Specialist Configuration")
        header_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_label.setStyleSheet("color: #00d7ff; margin-bottom: 8px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)
        subtitle = QLabel(
            "Configure models, temperatures, and personalities. Includes save/load presets per role.")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setStyleSheet("color: #888; margin-bottom: 15px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; } QScrollBar:vertical { border: none; background: #2d2d30; width: 8px; border-radius: 4px; } QScrollBar::handle:vertical { background: #404040; border-radius: 4px; min-height: 20px; }")

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(10)

        # ADDED THE STRUCTURER SECTION
        self.structurer_section = RoleSection("🗺️ Structurer", LLMRole.STRUCTURER, 0.1, self.personality_manager)
        self.planner_section = RoleSection("🧠 Planner", LLMRole.PLANNER, 0.7, self.personality_manager)
        self.coder_section = RoleSection("⚙️ Code Generator", LLMRole.CODER, 0.1, self.personality_manager)
        self.assembler_section = RoleSection("📄 Code Assembler", LLMRole.ASSEMBLER, 0.3, self.personality_manager)
        self.reviewer_section = RoleSection("🧐 Code Reviewer", LLMRole.REVIEWER, 0.5, self.personality_manager)
        self.chat_section = RoleSection("💬 General Chat", LLMRole.CHAT, 0.7, self.personality_manager)

        content_layout.addWidget(self.structurer_section) # ADDED
        content_layout.addWidget(self.planner_section)
        content_layout.addWidget(self.coder_section)
        content_layout.addWidget(self.assembler_section)
        content_layout.addWidget(self.reviewer_section)
        content_layout.addWidget(self.chat_section)

        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area, 1)
        button_layout = QHBoxLayout()
        self.test_button = ModernButton("🔍 Test Models", button_type="secondary")
        self.test_button.clicked.connect(self._test_models)
        self.reset_button = ModernButton("🔄 Reset All", button_type="secondary")
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
        self.setStyleSheet(
            "QDialog { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e1e1e, stop:1 #252526); color: #cccccc; border: 3px solid #00d7ff; border-radius: 15px; }")

    def _populate_models(self):
        if not self.llm_client or not hasattr(self.llm_client, 'models'): return
        available_models = {key: f"{config.provider}/{config.model}" for key, config in self.llm_client.models.items()}
        # ADDED STRUCTURER TO THE LIST OF SECTIONS
        all_sections = [self.structurer_section, self.planner_section, self.coder_section, self.assembler_section, self.reviewer_section, self.chat_section]
        for section in all_sections:
            section.model_combo.clear()
            for key, name in available_models.items(): section.model_combo.addItem(name, key)
            section.status_indicator.update_status("ready" if available_models else "error")

    def _load_current_config(self):
        if not self.llm_client or not hasattr(self.llm_client, 'role_assignments'): return

        assignments = self.llm_client.role_assignments
        personalities = getattr(self.llm_client, 'personalities', {})

        sections_map = {
            LLMRole.STRUCTURER: self.structurer_section, # ADDED
            LLMRole.PLANNER: self.planner_section,
            LLMRole.CODER: self.coder_section,
            LLMRole.ASSEMBLER: self.assembler_section,
            LLMRole.REVIEWER: self.reviewer_section,
            LLMRole.CHAT: self.chat_section,
        }

        for role_enum, section_widget in sections_map.items():
            model_key = assignments.get(role_enum)
            if model_key:
                for i in range(section_widget.model_combo.count()):
                    if section_widget.model_combo.itemData(i) == model_key:
                        section_widget.model_combo.setCurrentIndex(i)
                        if model_key in self.llm_client.models:
                            section_widget.set_temperature(self.llm_client.models[model_key].temperature)
                        break

            personality_text = personalities.get(role_enum)
            if personality_text:
                section_widget.set_personality(personality_text)
            else:
                section_widget.set_personality(section_widget.personality_editor._get_default_personality())

    def _reset_to_defaults(self):
        if QMessageBox.question(self, "Reset All Settings",
                                "Reset models, temperatures, and personalities to defaults?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            # ADDED STRUCTURER TO THE LIST OF SECTIONS
            all_sections = [self.structurer_section, self.planner_section, self.coder_section, self.assembler_section, self.reviewer_section, self.chat_section]
            for section in all_sections:
                section.set_personality(section.personality_editor._get_default_personality())
                section.set_temperature(section.default_temp)
                if section.model_combo.count() > 0:
                    found_suitable = False
                    for i in range(section.model_combo.count()):
                        model_key_for_item = section.model_combo.itemData(i)
                        if model_key_for_item in self.llm_client.models:
                            if section.role_enum_member in self.llm_client.models[model_key_for_item].suitable_roles:
                                section.model_combo.setCurrentIndex(i)
                                found_suitable = True
                                break
                    if not found_suitable and section.model_combo.count() > 0:
                        section.model_combo.setCurrentIndex(0)

    def _test_models(self):
        QMessageBox.information(self, "Model Testing", "Feature to be implemented.")

    def _apply_configuration(self):
        if not self.llm_client: self.reject(); return

        all_sections = {
            LLMRole.STRUCTURER: self.structurer_section, # ADDED
            LLMRole.PLANNER: self.planner_section,
            LLMRole.CODER: self.coder_section,
            LLMRole.ASSEMBLER: self.assembler_section,
            LLMRole.REVIEWER: self.reviewer_section,
            LLMRole.CHAT: self.chat_section,
        }

        new_assignments = {}
        new_personalities = {}

        for role_enum, section_widget in all_sections.items():
            selected_model_key = section_widget.get_selected_model()
            if selected_model_key:
                new_assignments[role_enum] = selected_model_key
                if selected_model_key in self.llm_client.models:
                    self.llm_client.models[selected_model_key].temperature = section_widget.get_temperature()
            new_personalities[role_enum] = section_widget.get_personality()

        try:
            self.llm_client.role_assignments.clear()
            self.llm_client.role_assignments.update(new_assignments)

            if not hasattr(self.llm_client, 'personalities'): self.llm_client.personalities = {}
            self.llm_client.personalities.clear()
            self.llm_client.personalities.update(new_personalities)

            self.configuration_applied.emit({
                'role_assignments': {k.value: v for k, v in new_assignments.items()},
                'personalities': {k.value: v for k, v in new_personalities.items()}
            })
            QMessageBox.information(self, "Configuration Applied", "AI Specialist configuration updated.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply configuration: {e}")

    def get_configuration_summary(self):
        return {
            LLMRole.STRUCTURER.value: self.structurer_section.model_combo.currentText(), # ADDED
            LLMRole.PLANNER.value: self.planner_section.model_combo.currentText(),
            LLMRole.CODER.value: self.coder_section.model_combo.currentText(),
            LLMRole.ASSEMBLER.value: self.assembler_section.model_combo.currentText(),
            LLMRole.REVIEWER.value: self.reviewer_section.model_combo.currentText(),
            LLMRole.CHAT.value: self.chat_section.model_combo.currentText(),
        }