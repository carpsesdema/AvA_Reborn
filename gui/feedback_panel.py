# gui/feedback_panel.py - User Feedback Controls for AI Collaboration

from datetime import datetime
from typing import Dict, List, Any

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QSlider, QComboBox, QFrame, QGroupBox, QCheckBox, QTabWidget,
    QProgressBar, QMessageBox
)

from gui.components import ModernButton


class CollaborationViewer(QFrame):
    """Widget to display AI collaboration in real-time"""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            CollaborationViewer {
                background: #1e1e1e;
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 4px;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel("🤝 AI Collaboration Monitor")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(header)

        # Collaboration display
        self.collaboration_display = QTextEdit()
        self.collaboration_display.setReadOnly(True)
        self.collaboration_display.setMaximumHeight(200)
        self.collaboration_display.setStyleSheet("""
            QTextEdit {
                background: #252526;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.collaboration_display)

        self.setLayout(layout)

        # Add initial message
        self.add_collaboration_message("system", "AI Collaboration Monitor ready")

    def add_collaboration_message(self, from_ai: str, content: str, to_ai: str = None):
        """Add a collaboration message to the display"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if to_ai:
            message = f"[{timestamp}] {from_ai} → {to_ai}: {content}"
        else:
            message = f"[{timestamp}] {from_ai}: {content}"

        # Color coding
        colors = {
            "planner": "#a5a5f0",
            "coder": "#58a6ff",
            "assembler": "#f0883e",
            "reviewer": "#3fb950",
            "system": "#cccccc"
        }

        color = colors.get(from_ai, "#cccccc")

        self.collaboration_display.append(
            f'<span style="color: {color};">{message}</span>'
        )

        # Auto-scroll to bottom
        scrollbar = self.collaboration_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class IterationControls(QGroupBox):
    """Controls for managing iterations and improvements"""

    iteration_requested = Signal(str, str)  # file_path, feedback
    feedback_settings_changed = Signal(dict)

    def __init__(self):
        super().__init__("🔄 Iteration Controls")
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00d7ff;
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Iteration settings
        settings_layout = QHBoxLayout()

        # Enable iterations checkbox
        self.enable_iterations = QCheckBox("Enable Iterations")
        self.enable_iterations.setChecked(True)
        self.enable_iterations.setStyleSheet("color: #cccccc; font-weight: normal;")
        settings_layout.addWidget(self.enable_iterations)

        # Max iterations
        settings_layout.addWidget(QLabel("Max:"))
        self.max_iterations = QComboBox()
        self.max_iterations.addItems(["1", "2", "3", "4", "5"])
        self.max_iterations.setCurrentText("3")
        self.max_iterations.setStyleSheet("""
            QComboBox {
                background: #2d2d30;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
                font-weight: normal;
            }
        """)
        settings_layout.addWidget(self.max_iterations)

        # Pause for feedback
        self.pause_for_feedback = QCheckBox("Pause for User Feedback")
        self.pause_for_feedback.setStyleSheet("color: #cccccc; font-weight: normal;")
        settings_layout.addWidget(self.pause_for_feedback)

        settings_layout.addStretch()
        layout.addLayout(settings_layout)

        # Current file iteration request
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Request Iteration:"))

        self.file_selector = QComboBox()
        self.file_selector.setStyleSheet("""
            QComboBox {
                background: #2d2d30;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
                font-weight: normal;
                min-width: 120px;
            }
        """)
        file_layout.addWidget(self.file_selector)

        self.request_iteration_btn = ModernButton("Request", button_type="accent")
        self.request_iteration_btn.setMaximumWidth(80)
        self.request_iteration_btn.clicked.connect(self._request_iteration)
        file_layout.addWidget(self.request_iteration_btn)

        layout.addLayout(file_layout)

        # Feedback text
        self.feedback_text = QTextEdit()
        self.feedback_text.setPlaceholderText("Enter specific feedback for improvement...")
        self.feedback_text.setMaximumHeight(80)
        self.feedback_text.setStyleSheet("""
            QTextEdit {
                background: #252526;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                font-weight: normal;
            }
        """)
        layout.addWidget(self.feedback_text)

        # Connect signals
        self.enable_iterations.toggled.connect(self._emit_settings_changed)
        self.max_iterations.currentTextChanged.connect(self._emit_settings_changed)
        self.pause_for_feedback.toggled.connect(self._emit_settings_changed)

        self.setLayout(layout)

    def _request_iteration(self):
        """Request an iteration for the selected file"""
        file_path = self.file_selector.currentText()
        feedback = self.feedback_text.toPlainText().strip()

        if not file_path or file_path == "No files available":
            QMessageBox.warning(self, "No File Selected", "Please select a file to iterate on.")
            return

        if not feedback:
            QMessageBox.warning(self, "No Feedback", "Please provide specific feedback for improvement.")
            return

        self.iteration_requested.emit(file_path, feedback)
        self.feedback_text.clear()

    def _emit_settings_changed(self):
        """Emit settings changed signal"""
        settings = {
            "enable_iterations": self.enable_iterations.isChecked(),
            "max_iterations": int(self.max_iterations.currentText()),
            "pause_for_feedback": self.pause_for_feedback.isChecked()
        }
        self.feedback_settings_changed.emit(settings)

    def update_available_files(self, files: List[str]):
        """Update the list of available files for iteration"""
        self.file_selector.clear()
        if files:
            self.file_selector.addItems(files)
        else:
            self.file_selector.addItem("No files available")


class QualityMetrics(QFrame):
    """Display quality metrics and insights"""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QualityMetrics {
                background: #252526;
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 4px;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        header = QLabel("📊 Quality Metrics")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(header)

        # Overall score
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("Overall Score:"))

        self.overall_score_bar = QProgressBar()
        self.overall_score_bar.setRange(0, 100)
        self.overall_score_bar.setValue(0)
        self.overall_score_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 4px;
                text-align: center;
                background: #1e1e1e;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 2px;
            }
        """)
        score_layout.addWidget(self.overall_score_bar)

        self.score_label = QLabel("0/10")
        self.score_label.setStyleSheet("color: #00d7ff; font-weight: bold;")
        score_layout.addWidget(self.score_label)

        layout.addLayout(score_layout)

        # Individual metrics
        metrics_layout = QVBoxLayout()

        self.consistency_metric = self._create_metric_row("Consistency", 0)
        self.quality_metric = self._create_metric_row("Code Quality", 0)
        self.integration_metric = self._create_metric_row("Integration", 0)

        metrics_layout.addLayout(self.consistency_metric)
        metrics_layout.addLayout(self.quality_metric)
        metrics_layout.addLayout(self.integration_metric)

        layout.addLayout(metrics_layout)

        # Insights
        self.insights_label = QLabel("No insights available")
        self.insights_label.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        self.insights_label.setWordWrap(True)
        layout.addWidget(self.insights_label)

        self.setLayout(layout)

    def _create_metric_row(self, name: str, value: int) -> QHBoxLayout:
        """Create a metric display row"""
        layout = QHBoxLayout()

        label = QLabel(f"{name}:")
        label.setStyleSheet("color: #cccccc; font-size: 11px;")
        label.setMinimumWidth(80)
        layout.addWidget(label)

        bar = QProgressBar()
        bar.setRange(0, 10)
        bar.setValue(value)
        bar.setMaximumHeight(15)
        bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 2px;
                text-align: center;
                background: #1e1e1e;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background: #4ade80;
                border-radius: 1px;
            }
        """)
        layout.addWidget(bar)

        # Store reference for updates
        setattr(self, f"{name.lower().replace(' ', '_')}_bar", bar)

        return layout

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update the quality metrics display"""
        overall_score = metrics.get("overall_score", 0)
        self.overall_score_bar.setValue(int(overall_score * 10))
        self.score_label.setText(f"{overall_score:.1f}/10")

        # Update individual metrics
        consistency = metrics.get("consistency_score", 0)
        self.consistency_bar.setValue(int(consistency * 10))

        code_quality = metrics.get("code_quality", 0)
        self.code_quality_bar.setValue(int(code_quality * 10))

        integration = metrics.get("integration_score", 0)
        self.integration_bar.setValue(int(integration * 10))

        # Update insights
        improvements = metrics.get("files_needing_improvement", 0)
        if improvements > 0:
            self.insights_label.setText(f"{improvements} files need improvement")
        else:
            self.insights_label.setText("All files meet quality standards")


class WorkflowProgress(QFrame):
    """Enhanced workflow progress display"""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            WorkflowProgress {
                background: #252526;
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 4px;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel("⚡ Workflow Progress")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(header)

        # Current stage
        self.stage_label = QLabel("Stage: Idle")
        self.stage_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        layout.addWidget(self.stage_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 4px;
                text-align: center;
                background: #1e1e1e;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # File progress
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Files:"))

        self.file_progress_label = QLabel("0/0")
        self.file_progress_label.setStyleSheet("color: #00d7ff; font-weight: bold;")
        file_layout.addWidget(self.file_progress_label)

        file_layout.addStretch()

        # Iterations info
        self.iterations_label = QLabel("Iterations: 0")
        self.iterations_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        file_layout.addWidget(self.iterations_label)

        layout.addLayout(file_layout)

        # Current file
        self.current_file_label = QLabel("No active file")
        self.current_file_label.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        layout.addWidget(self.current_file_label)

        self.setLayout(layout)

    def update_stage(self, stage: str, description: str):
        """Update workflow stage"""
        self.stage_label.setText(f"Stage: {stage.title()}")

        # Update progress based on stage
        stage_progress = {
            "initializing": 5,
            "planning": 15,
            "generation": 50,
            "finalization": 90,
            "complete": 100
        }

        progress = stage_progress.get(stage, 0)
        self.progress_bar.setValue(progress)

    def update_file_progress(self, completed: int, total: int, current_file: str = None):
        """Update file progress"""
        self.file_progress_label.setText(f"{completed}/{total}")

        if current_file:
            self.current_file_label.setText(f"Processing: {current_file}")
        else:
            self.current_file_label.setText("No active file")

    def update_iterations(self, total_iterations: int):
        """Update iteration count"""
        self.iterations_label.setText(f"Iterations: {total_iterations}")


class FeedbackPanel(QWidget):
    """
    🎛️ Main Feedback Panel for AI Collaboration Control

    This panel provides comprehensive user controls for:
    - Monitoring AI collaboration in real-time
    - Controlling iteration and feedback settings
    - Viewing quality metrics and insights
    - Managing workflow progress
    - Requesting specific improvements
    """

    # Signals for communication with the workflow engine
    iteration_requested = Signal(str, str)  # file_path, feedback
    feedback_settings_changed = Signal(dict)
    user_feedback_added = Signal(str, str, int, str)  # type, content, rating, file_path

    def __init__(self):
        super().__init__()
        self.current_files = []
        self.current_metrics = {}
        self._init_ui()
        self._connect_signals()

        # Auto-update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._periodic_update)
        self.update_timer.start(2000)  # Update every 2 seconds

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Main tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #404040;
                border-radius: 4px;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d30;
                color: #cccccc;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #00d7ff;
                color: #1e1e1e;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #3e3e42;
            }
        """)

        # Tab 1: Collaboration & Control
        control_widget = QWidget()
        control_layout = QVBoxLayout()

        # Workflow progress
        self.workflow_progress = WorkflowProgress()
        control_layout.addWidget(self.workflow_progress)

        # Collaboration viewer
        self.collaboration_viewer = CollaborationViewer()
        control_layout.addWidget(self.collaboration_viewer)

        # Iteration controls
        self.iteration_controls = IterationControls()
        control_layout.addWidget(self.iteration_controls)

        control_layout.addStretch()
        control_widget.setLayout(control_layout)

        # Tab 2: Quality & Insights
        quality_widget = QWidget()
        quality_layout = QVBoxLayout()

        # Quality metrics
        self.quality_metrics = QualityMetrics()
        quality_layout.addWidget(self.quality_metrics)

        # General feedback section
        feedback_section = QGroupBox("💬 General Feedback")
        feedback_section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00d7ff;
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        feedback_layout = QVBoxLayout()

        # Feedback type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))

        self.feedback_type = QComboBox()
        self.feedback_type.addItems([
            "General", "Code Quality", "Architecture",
            "Documentation", "Performance", "User Experience"
        ])
        self.feedback_type.setStyleSheet("""
            QComboBox {
                background: #2d2d30;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
                font-weight: normal;
            }
        """)
        type_layout.addWidget(self.feedback_type)

        # Rating
        type_layout.addWidget(QLabel("Rating:"))
        self.feedback_rating = QSlider(Qt.Orientation.Horizontal)
        self.feedback_rating.setRange(1, 10)
        self.feedback_rating.setValue(5)
        self.feedback_rating.setMaximumWidth(100)
        type_layout.addWidget(self.feedback_rating)

        self.rating_label = QLabel("5")
        self.rating_label.setStyleSheet("color: #00d7ff; font-weight: bold;")
        self.feedback_rating.valueChanged.connect(lambda v: self.rating_label.setText(str(v)))
        type_layout.addWidget(self.rating_label)

        type_layout.addStretch()
        feedback_layout.addLayout(type_layout)

        # Feedback text
        self.general_feedback_text = QTextEdit()
        self.general_feedback_text.setPlaceholderText("Enter your feedback about the AI workflow and results...")
        self.general_feedback_text.setMaximumHeight(100)
        self.general_feedback_text.setStyleSheet("""
            QTextEdit {
                background: #252526;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                font-weight: normal;
            }
        """)
        feedback_layout.addWidget(self.general_feedback_text)

        # Submit feedback button
        self.submit_feedback_btn = ModernButton("Submit Feedback", button_type="primary")
        self.submit_feedback_btn.clicked.connect(self._submit_general_feedback)
        feedback_layout.addWidget(self.submit_feedback_btn)

        feedback_section.setLayout(feedback_layout)
        quality_layout.addWidget(feedback_section)

        quality_layout.addStretch()
        quality_widget.setLayout(quality_layout)

        # Add tabs
        self.tab_widget.addTab(control_widget, "🎛️ Control")
        self.tab_widget.addTab(quality_widget, "📊 Quality")

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def _connect_signals(self):
        """Connect internal signals"""
        self.iteration_controls.iteration_requested.connect(self.iteration_requested)
        self.iteration_controls.feedback_settings_changed.connect(self.feedback_settings_changed)

    def _submit_general_feedback(self):
        """Submit general feedback"""
        feedback_type = self.feedback_type.currentText()
        content = self.general_feedback_text.toPlainText().strip()
        rating = self.feedback_rating.value()

        if not content:
            QMessageBox.warning(self, "No Feedback", "Please enter feedback content.")
            return

        self.user_feedback_added.emit(feedback_type, content, rating, "")
        self.general_feedback_text.clear()

        # Show confirmation
        self.collaboration_viewer.add_collaboration_message(
            "user", f"Feedback submitted: {feedback_type} (Rating: {rating})"
        )

    def _periodic_update(self):
        """Periodic update of display elements"""
        # This could fetch current metrics from the workflow engine
        pass

    # Public interface methods for integration

    def update_workflow_stage(self, stage: str, description: str):
        """Update workflow stage display"""
        self.workflow_progress.update_stage(stage, description)

    def update_file_progress(self, completed: int, total: int, current_file: str = None):
        """Update file progress"""
        self.workflow_progress.update_file_progress(completed, total, current_file)

    def update_available_files(self, files: List[str]):
        """Update available files for iteration"""
        self.current_files = files
        self.iteration_controls.update_available_files(files)

    def add_ai_collaboration_message(self, from_ai: str, to_ai: str, content: str):
        """Add AI collaboration message"""
        self.collaboration_viewer.add_collaboration_message(from_ai, content, to_ai)

    def update_quality_metrics(self, metrics: Dict[str, Any]):
        """Update quality metrics display"""
        self.current_metrics = metrics
        self.quality_metrics.update_metrics(metrics)

    def update_iteration_count(self, total_iterations: int):
        """Update total iteration count"""
        self.workflow_progress.update_iterations(total_iterations)

    def show_file_completed(self, file_path: str, approved: bool, iterations: int):
        """Show file completion status"""
        status = "APPROVED" if approved else "NEEDS_REVIEW"
        message = f"File {file_path} completed: {status} (after {iterations} iterations)"
        self.collaboration_viewer.add_collaboration_message("system", message)

    def show_workflow_complete(self, result: Dict[str, Any]):
        """Show workflow completion summary"""
        success = result.get("success", False)
        file_count = result.get("file_count", 0)

        if success:
            message = f"Workflow completed successfully! Generated {file_count} files."
        else:
            message = f"Workflow completed with issues. Check the results."

        self.collaboration_viewer.add_collaboration_message("system", message)

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current feedback settings"""
        return {
            "enable_iterations": self.iteration_controls.enable_iterations.isChecked(),
            "max_iterations": int(self.iteration_controls.max_iterations.currentText()),
            "pause_for_feedback": self.iteration_controls.pause_for_feedback.isChecked()
        }

    def clear_display(self):
        """Clear all display elements"""
        self.collaboration_viewer.collaboration_display.clear()
        self.collaboration_viewer.add_collaboration_message("system", "Display cleared - ready for new workflow")
        self.workflow_progress.stage_label.setText("Stage: Idle")
        self.workflow_progress.progress_bar.setValue(0)
        self.workflow_progress.file_progress_label.setText("0/0")
        self.workflow_progress.current_file_label.setText("No active file")
        self.workflow_progress.iterations_label.setText("Iterations: 0")