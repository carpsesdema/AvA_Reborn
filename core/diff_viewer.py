# gui/diff_viewer.py - Visual Diff Viewer for Code Changes

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTextEdit,
    QLabel, QPushButton, QFrame, QScrollArea, QTabWidget,
    QListWidget, QListWidgetItem, QGroupBox, QRadioButton,
    QButtonGroup, QProgressBar, QComboBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QTextCharFormat, QColor, QTextCursor, QSyntaxHighlighter

from typing import Dict, List, Any, Optional
from pathlib import Path

from gui.components import ModernButton, StatusIndicator


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code in diff view"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_formats()

    def _init_formats(self):
        """Initialize text formats for syntax highlighting"""
        self.formats = {}

        # Keywords
        self.formats['keyword'] = QTextCharFormat()
        self.formats['keyword'].setColor(QColor(86, 156, 214))  # Blue
        self.formats['keyword'].setFontWeight(QFont.Weight.Bold)

        # Strings
        self.formats['string'] = QTextCharFormat()
        self.formats['string'].setColor(QColor(206, 145, 120))  # Orange

        # Comments
        self.formats['comment'] = QTextCharFormat()
        self.formats['comment'].setColor(QColor(106, 153, 85))  # Green
        self.formats['comment'].setFontItalic(True)

        # Functions
        self.formats['function'] = QTextCharFormat()
        self.formats['function'].setColor(QColor(220, 220, 170))  # Yellow

    def highlightBlock(self, text):
        """Apply syntax highlighting to a block of text"""
        import re

        # Keywords
        keywords = [
            'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except',
            'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'lambda',
            'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'
        ]

        for keyword in keywords:
            pattern = rf'\b{keyword}\b'
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self.formats['keyword'])

        # Strings
        string_patterns = [r'"[^"]*"', r"'[^']*'", r'""".*?"""', r"'''.*?'''"]
        for pattern in string_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                self.setFormat(match.start(), match.end() - match.start(), self.formats['string'])

        # Comments
        comment_pattern = r'#.*$'
        for match in re.finditer(comment_pattern, text, re.MULTILINE):
            self.setFormat(match.start(), match.end() - match.start(), self.formats['comment'])

        # Function definitions
        func_pattern = r'\bdef\s+(\w+)'
        for match in re.finditer(func_pattern, text):
            start = match.start(1)
            length = match.end(1) - start
            self.setFormat(start, length, self.formats['function'])


class DiffTextEdit(QTextEdit):
    """Enhanced text editor for displaying diffs with highlighting"""

    line_clicked = Signal(int)  # Emitted when a line is clicked

    def __init__(self, side="left"):
        super().__init__()
        self.side = side
        self.diff_data = []
        self.line_highlights = {}

        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))

        # Set up syntax highlighting
        self.highlighter = PythonSyntaxHighlighter(self.document())

        self.setStyleSheet("""
            DiffTextEdit {
                background: #1e1e1e;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                selection-background-color: #264f78;
            }
        """)

    def set_diff_content(self, content: str, diff_data: List[Dict] = None):
        """Set content with diff highlighting"""
        self.diff_data = diff_data or []
        self.setPlainText(content)
        self._apply_diff_highlighting()

    def _apply_diff_highlighting(self):
        """Apply diff-specific highlighting"""
        if not self.diff_data:
            return

        cursor = QTextCursor(self.document())
        lines = self.toPlainText().splitlines()

        for i, line_data in enumerate(self.diff_data):
            if i >= len(lines):
                break

            # Move cursor to line
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            for _ in range(i):
                cursor.movePosition(QTextCursor.MoveOperation.Down)

            # Select the entire line
            cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor)

            # Apply highlighting based on change type
            format = QTextCharFormat()

            if line_data.get('type') == 'addition':
                format.setBackground(QColor(40, 80, 40, 100))  # Green background
            elif line_data.get('type') == 'deletion':
                format.setBackground(QColor(80, 40, 40, 100))  # Red background
            elif line_data.get('type') == 'modification':
                format.setBackground(QColor(80, 80, 40, 100))  # Yellow background

            cursor.setCharFormat(format)


class ChangeWidget(QFrame):
    """Widget representing a single code change"""

    change_approved = Signal(dict, bool)  # change, approved

    def __init__(self, change_data: Dict):
        super().__init__()
        self.change_data = change_data
        self._init_ui()

    def _init_ui(self):
        """Initialize the change widget UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            ChangeWidget {
                background: #252526;
                border: 1px solid #404040;
                border-radius: 6px;
                margin: 2px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header with change type and impact
        header_layout = QHBoxLayout()

        change_type = self.change_data.get('change_type', 'unknown')
        impact = self.change_data.get('impact', 'medium')

        type_label = QLabel(f"🔄 {change_type.title()}")
        type_label.setStyleSheet("color: #00d7ff; font-weight: bold;")
        header_layout.addWidget(type_label)

        impact_colors = {
            'low': '#4ade80',
            'medium': '#ffb900',
            'high': '#ff8c00',
            'critical': '#ff4444'
        }

        impact_label = QLabel(f"Impact: {impact.title()}")
        impact_label.setStyleSheet(f"color: {impact_colors.get(impact, '#cccccc')}; font-size: 11px;")
        header_layout.addWidget(impact_label)

        header_layout.addStretch()

        # Line range
        line_start = self.change_data.get('line_start', 0)
        line_end = self.change_data.get('line_end', 0)
        lines_label = QLabel(f"Lines {line_start}-{line_end}")
        lines_label.setStyleSheet("color: #888; font-size: 10px;")
        header_layout.addWidget(lines_label)

        layout.addLayout(header_layout)

        # Description
        description = self.change_data.get('description', 'Code change')
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # AI reasoning if available
        ai_reasoning = self.change_data.get('ai_reasoning', '')
        if ai_reasoning:
            reasoning_label = QLabel(f"AI Reasoning: {ai_reasoning}")
            reasoning_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
            reasoning_label.setWordWrap(True)
            layout.addWidget(reasoning_label)

        # Code preview
        old_content = self.change_data.get('old_content', '')
        new_content = self.change_data.get('new_content', '')

        if old_content or new_content:
            code_frame = QFrame()
            code_frame.setFrameStyle(QFrame.Shape.StyledPanel)
            code_frame.setStyleSheet("background: #1e1e1e; border: 1px solid #333;")
            code_layout = QVBoxLayout(code_frame)
            code_layout.setContentsMargins(4, 4, 4, 4)

            if old_content:
                old_label = QLabel("- " + old_content.replace('\n', '\n- '))
                old_label.setStyleSheet("color: #ff6b6b; font-family: Consolas; font-size: 9px;")
                code_layout.addWidget(old_label)

            if new_content:
                new_label = QLabel("+ " + new_content.replace('\n', '\n+ '))
                new_label.setStyleSheet("color: #51cf66; font-family: Consolas; font-size: 9px;")
                code_layout.addWidget(new_label)

            layout.addWidget(code_frame)

        # Approval buttons
        button_layout = QHBoxLayout()

        self.approve_btn = ModernButton("✅ Approve", button_type="primary")
        self.approve_btn.setMaximumHeight(28)
        self.approve_btn.clicked.connect(lambda: self.change_approved.emit(self.change_data, True))

        self.reject_btn = ModernButton("❌ Reject", button_type="danger")
        self.reject_btn.setMaximumHeight(28)
        self.reject_btn.clicked.connect(lambda: self.change_approved.emit(self.change_data, False))

        button_layout.addWidget(self.approve_btn)
        button_layout.addWidget(self.reject_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        self.setLayout(layout)


class DiffSummaryPanel(QFrame):
    """Panel showing summary of all changes"""

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """Initialize summary panel UI"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            DiffSummaryPanel {
                background: #252526;
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 4px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel("📊 Change Summary")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #00d7ff; background: transparent;")
        layout.addWidget(header)

        # Stats
        self.stats_layout = QVBoxLayout()
        layout.addLayout(self.stats_layout)

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

        # Action buttons
        button_layout = QHBoxLayout()

        self.approve_all_btn = ModernButton("✅ Approve All", button_type="primary")
        self.reject_all_btn = ModernButton("❌ Reject All", button_type="danger")
        self.review_btn = ModernButton("👁️ Review Each", button_type="secondary")

        button_layout.addWidget(self.approve_all_btn)
        button_layout.addWidget(self.reject_all_btn)
        button_layout.addWidget(self.review_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

        self.setLayout(layout)

    def update_summary(self, diff_data: Dict):
        """Update summary with diff data"""
        # Clear existing stats
        for i in reversed(range(self.stats_layout.count())):
            self.stats_layout.itemAt(i).widget().setParent(None)

        # Add new stats
        changes = diff_data.get('changes', [])

        if not changes:
            no_changes = QLabel("No changes detected")
            no_changes.setStyleSheet("color: #888; font-style: italic;")
            self.stats_layout.addWidget(no_changes)
            return

        # Change type counts
        type_counts = {}
        impact_counts = {}

        for change in changes:
            change_type = change.get('change_type', 'unknown')
            impact = change.get('impact', 'medium')

            type_counts[change_type] = type_counts.get(change_type, 0) + 1
            impact_counts[impact] = impact_counts.get(impact, 0) + 1

        # Display type counts
        for change_type, count in type_counts.items():
            type_label = QLabel(f"{change_type.title()}: {count}")
            type_label.setStyleSheet("color: #cccccc; font-size: 11px;")
            self.stats_layout.addWidget(type_label)

        # Display impact summary
        high_impact = impact_counts.get('high', 0) + impact_counts.get('critical', 0)
        if high_impact > 0:
            impact_label = QLabel(f"⚠️ {high_impact} high-impact changes")
            impact_label.setStyleSheet("color: #ff8c00; font-weight: bold; font-size: 11px;")
            self.stats_layout.addWidget(impact_label)

        # Overall recommendation
        overall_impact = diff_data.get('overall_impact', 'medium')
        recommendation = diff_data.get('recommendation', 'review_needed')

        rec_colors = {
            'approve': '#4ade80',
            'review_needed': '#ffb900',
            'reject': '#ff4444'
        }

        rec_label = QLabel(f"Recommendation: {recommendation.replace('_', ' ').title()}")
        rec_label.setStyleSheet(f"color: {rec_colors.get(recommendation, '#cccccc')}; font-weight: bold;")
        self.stats_layout.addWidget(rec_label)


class DiffViewer(QWidget):
    """
    📋 Visual Diff Viewer - Side-by-side Code Comparison

    Provides comprehensive diff viewing with:
    - Side-by-side comparison
    - Syntax highlighting
    - Change approval/rejection
    - Merge conflict resolution
    - Integration with enhanced workflow
    """

    changes_approved = Signal(list)  # List of approved changes
    merge_requested = Signal(str)  # Merge strategy requested

    def __init__(self):
        super().__init__()
        self.current_diff = None
        self.approved_changes = []
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initialize the diff viewer UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Main splitter
        self.main_splitter = QSplitter()
        self.main_splitter.setOrientation(Qt.Orientation.Horizontal)

        # Left panel - Changes list and summary
        left_panel = self._create_left_panel()

        # Right panel - Side-by-side diff view
        right_panel = self._create_right_panel()

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)

        # Set initial sizes
        self.main_splitter.setSizes([350, 800])

        layout.addWidget(self.main_splitter)
        self.setLayout(layout)

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with changes list and controls"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Summary panel
        self.summary_panel = DiffSummaryPanel()
        layout.addWidget(self.summary_panel)

        # Changes list
        changes_label = QLabel("📝 Individual Changes")
        changes_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        changes_label.setStyleSheet("color: #00d7ff;")
        layout.addWidget(changes_label)

        # Scroll area for changes
        self.changes_scroll = QScrollArea()
        self.changes_scroll.setWidgetResizable(True)
        self.changes_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #2d2d30;
                width: 8px;
                border-radius: 4px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #404040;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00d7ff;
            }
        """)

        self.changes_widget = QWidget()
        self.changes_layout = QVBoxLayout(self.changes_widget)
        self.changes_layout.setContentsMargins(0, 0, 0, 0)
        self.changes_layout.setSpacing(4)

        self.changes_scroll.setWidget(self.changes_widget)
        layout.addWidget(self.changes_scroll)

        panel.setLayout(layout)
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with side-by-side diff view"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header with file info
        header_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.file_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.file_label.setStyleSheet("color: #00d7ff;")
        header_layout.addWidget(self.file_label)

        header_layout.addStretch()

        # Iteration selector
        self.iteration_combo = QComboBox()
        self.iteration_combo.setStyleSheet("""
            QComboBox {
                background: #2d2d30;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
                min-width: 100px;
            }
        """)
        header_layout.addWidget(QLabel("Iteration:"))
        header_layout.addWidget(self.iteration_combo)

        layout.addLayout(header_layout)

        # Side-by-side diff
        diff_splitter = QSplitter()
        diff_splitter.setOrientation(Qt.Orientation.Horizontal)

        # Old version (left)
        old_frame = QFrame()
        old_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        old_layout = QVBoxLayout(old_frame)
        old_layout.setContentsMargins(4, 4, 4, 4)

        old_header = QLabel("📄 Previous Version")
        old_header.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        old_layout.addWidget(old_header)

        self.old_editor = DiffTextEdit("left")
        old_layout.addWidget(self.old_editor)

        # New version (right)
        new_frame = QFrame()
        new_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        new_layout = QVBoxLayout(new_frame)
        new_layout.setContentsMargins(4, 4, 4, 4)

        new_header = QLabel("📄 New Version")
        new_header.setStyleSheet("color: #51cf66; font-weight: bold;")
        new_layout.addWidget(new_header)

        self.new_editor = DiffTextEdit("right")
        new_layout.addWidget(self.new_editor)

        diff_splitter.addWidget(old_frame)
        diff_splitter.addWidget(new_frame)
        diff_splitter.setSizes([400, 400])

        layout.addWidget(diff_splitter)

        # Action buttons
        action_layout = QHBoxLayout()

        self.apply_btn = ModernButton("✅ Apply Changes", button_type="primary")
        self.cancel_btn = ModernButton("❌ Cancel", button_type="secondary")
        self.save_diff_btn = ModernButton("💾 Save Diff", button_type="secondary")

        action_layout.addWidget(self.apply_btn)
        action_layout.addWidget(self.cancel_btn)
        action_layout.addWidget(self.save_diff_btn)
        action_layout.addStretch()

        layout.addLayout(action_layout)

        panel.setLayout(layout)
        return panel

    def _connect_signals(self):
        """Connect widget signals"""
        self.summary_panel.approve_all_btn.clicked.connect(self._approve_all_changes)
        self.summary_panel.reject_all_btn.clicked.connect(self._reject_all_changes)
        self.summary_panel.review_btn.clicked.connect(self._enter_review_mode)

        self.apply_btn.clicked.connect(self._apply_approved_changes)
        self.cancel_btn.clicked.connect(self._cancel_diff)
        self.save_diff_btn.clicked.connect(self._save_diff)

        self.iteration_combo.currentTextChanged.connect(self._load_iteration)

    def load_diff(self, diff_data: Dict):
        """Load and display a diff"""
        self.current_diff = diff_data
        self.approved_changes = []

        # Update file label
        file_path = diff_data.get('file_path', 'Unknown file')
        self.file_label.setText(f"📄 {Path(file_path).name}")

        # Update summary
        self.summary_panel.update_summary(diff_data)

        # Clear and populate changes list
        self._clear_changes_list()

        changes = diff_data.get('changes', [])
        for change in changes:
            change_widget = ChangeWidget(change)
            change_widget.change_approved.connect(self._on_change_approved)
            self.changes_layout.addWidget(change_widget)

        # Add stretch to bottom
        self.changes_layout.addStretch()

        # Load diff content
        old_content = diff_data.get('old_version', '')
        new_content = diff_data.get('new_version', '')

        self.old_editor.set_diff_content(old_content)
        self.new_editor.set_diff_content(new_content)

        # Update progress
        self._update_progress()

    def load_iteration_history(self, file_path: str, iterations: List[Dict]):
        """Load iteration history for a file"""
        self.iteration_combo.clear()

        for i, iteration in enumerate(iterations):
            label = iteration.get('label', f'Iteration {i + 1}')
            self.iteration_combo.addItem(label)

    def _clear_changes_list(self):
        """Clear the changes list"""
        while self.changes_layout.count() > 0:
            child = self.changes_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _on_change_approved(self, change_data: Dict, approved: bool):
        """Handle change approval/rejection"""
        change_id = id(change_data)  # Use object id as unique identifier

        if approved:
            if change_id not in [id(c) for c in self.approved_changes]:
                self.approved_changes.append(change_data)
        else:
            self.approved_changes = [c for c in self.approved_changes if id(c) != change_id]

        self._update_progress()

    def _update_progress(self):
        """Update progress bar based on approved changes"""
        if not self.current_diff:
            return

        total_changes = len(self.current_diff.get('changes', []))
        approved_count = len(self.approved_changes)

        if total_changes > 0:
            progress = int((approved_count / total_changes) * 100)
            self.progress_bar.setValue(progress)

        # Update apply button state
        self.apply_btn.setEnabled(approved_count > 0)

    def _approve_all_changes(self):
        """Approve all changes"""
        if not self.current_diff:
            return

        self.approved_changes = self.current_diff.get('changes', []).copy()
        self._update_progress()

        # Update UI to show all changes as approved
        for i in range(self.changes_layout.count()):
            widget = self.changes_layout.itemAt(i).widget()
            if isinstance(widget, ChangeWidget):
                widget.approve_btn.setStyleSheet("background: #4ade80; color: white;")
                widget.reject_btn.setStyleSheet("")

    def _reject_all_changes(self):
        """Reject all changes"""
        self.approved_changes = []
        self._update_progress()

        # Update UI to show all changes as rejected
        for i in range(self.changes_layout.count()):
            widget = self.changes_layout.itemAt(i).widget()
            if isinstance(widget, ChangeWidget):
                widget.approve_btn.setStyleSheet("")
                widget.reject_btn.setStyleSheet("background: #ff4444; color: white;")

    def _enter_review_mode(self):
        """Enter detailed review mode"""
        # Could highlight changes in the editors or open detailed view
        pass

    def _apply_approved_changes(self):
        """Apply the approved changes"""
        if self.approved_changes:
            self.changes_approved.emit(self.approved_changes)

    def _cancel_diff(self):
        """Cancel the diff without applying changes"""
        self.current_diff = None
        self.approved_changes = []
        self._clear_changes_list()

        self.old_editor.clear()
        self.new_editor.clear()
        self.file_label.setText("No file selected")

    def _save_diff(self):
        """Save the diff for later review"""
        # Could save to file or project state
        pass

    def _load_iteration(self, iteration_label: str):
        """Load a specific iteration"""
        # This would integrate with the diff workflow integration
        pass


# Integration with main window and code viewer
class DiffViewerWindow(QWidget):
    """Standalone diff viewer window"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AvA - Code Diff Viewer")
        self.setGeometry(200, 200, 1200, 800)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.diff_viewer = DiffViewer()
        layout.addWidget(self.diff_viewer)

        self.setLayout(layout)

        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background: #1e1e1e;
                color: #cccccc;
            }
        """)