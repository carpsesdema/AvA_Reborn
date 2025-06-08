# windows/code_viewer.py - Professional Code Viewer/IDE Window

import os
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QPushButton, QLabel,
    QFileDialog, QMessageBox, QTabWidget, QFrame, QToolBar, QMenuBar,
    QMenu, QApplication, QHeaderView, QTabBar
)
from PySide6.QtCore import Qt, Signal, Slot, QFileSystemWatcher
from PySide6.QtGui import QFont, QAction, QIcon, QSyntaxHighlighter, QTextCharFormat, QColor
import re

from gui.components import Colors
# NEW: Import our interactive terminal
from gui.interactive_terminal import InteractiveTerminal


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Professional Python syntax highlighter"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Define colors for dark theme
        keyword_color = QColor("#569cd6")  # Blue
        string_color = QColor("#ce9178")  # Orange
        comment_color = QColor("#6a9955")  # Green
        function_color = QColor("#dcdcaa")  # Yellow

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(keyword_color)
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'exec', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
            'not', 'or', 'pass', 'print', 'raise', 'return', 'try',
            'while', 'with', 'yield', 'None', 'True', 'False', 'self'
        ]
        for word in keywords:
            pattern = r'\b' + word + r'\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))

        # Strings
        string_format = QTextCharFormat()
        string_format.setForeground(string_color)
        self.highlighting_rules.append((re.compile(r'".*?"'), string_format))
        self.highlighting_rules.append((re.compile(r"'.*?'"), string_format))

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(comment_color)
        self.highlighting_rules.append((re.compile(r'#.*'), comment_format))

        # Functions
        function_format = QTextCharFormat()
        function_format.setForeground(function_color)
        self.highlighting_rules.append((re.compile(r'\bdef\s+(\w+)'), function_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), format)


class CodeEditor(QTextEdit):
    """Professional code editor with syntax highlighting"""

    file_modified = Signal(str)  # file_path

    def __init__(self):
        super().__init__()
        self.file_path = None
        self.is_modified = False

        # Setup editor
        self.setFont(QFont("JetBrains Mono", 12))
        self.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 8px;
                selection-background-color: #264f78;
            }
        """)

        # Add syntax highlighting
        self.highlighter = PythonSyntaxHighlighter(self.document())

        # Track modifications
        self.textChanged.connect(self._on_text_changed)

    def load_file(self, file_path: str):
        """Load file content into editor"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.setPlainText(content)
            self.file_path = file_path
            self.is_modified = False

            # Move cursor to beginning
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.setTextCursor(cursor)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file: {e}")

    def save_file(self) -> bool:
        """Save current content to file"""
        if not self.file_path:
            return False

        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(self.toPlainText())

            self.is_modified = False
            return True

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save file: {e}")
            return False

    def _on_text_changed(self):
        """Handle text changes"""
        if self.file_path and not self.is_modified:
            self.is_modified = True
            self.file_modified.emit(self.file_path)


class FileTree(QTreeWidget):
    """Professional file tree with project navigation"""

    file_selected = Signal(str)  # file_path
    folder_selected = Signal(str)  # folder_path

    def __init__(self):
        super().__init__()
        self.setHeaderLabel("Project Files")
        self.setStyleSheet("""
            QTreeWidget {
                background: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                outline: none;
            }
            QTreeWidget::item {
                padding: 4px;
                border-radius: 2px;
            }
            QTreeWidget::item:hover {
                background: #2d2d30;
            }
            QTreeWidget::item:selected {
                background: #0078d4;
                color: white;
            }
        """)

        # Setup behavior
        self.itemClicked.connect(self._on_item_clicked)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # File system watcher for auto-refresh
        self.watcher = QFileSystemWatcher()
        self.watcher.directoryChanged.connect(self.refresh_tree)

        self.project_root = None

    def load_project(self, project_path: str):
        """Load project directory into tree"""
        self.project_root = Path(project_path)
        if not self.project_root.exists():
            self.project_root.mkdir(parents=True, exist_ok=True)

        # Watch for changes
        if self.watcher.directories():
            self.watcher.removePaths(self.watcher.directories())
        self.watcher.addPath(str(self.project_root))

        # Build tree
        self.clear()
        self._build_tree_recursive(self.project_root, self.invisibleRootItem())

        # Expand root level
        self.expandAll()

    def _build_tree_recursive(self, path: Path, parent_item: QTreeWidgetItem):
        """Recursively build file tree"""
        try:
            # Sort: directories first, then files
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

            for item_path in items:
                if item_path.name.startswith('.') or item_path.name == '__pycache__':
                    continue  # Skip hidden files/dirs

                item = QTreeWidgetItem(parent_item)
                item.setText(0, item_path.name)
                item.setData(0, Qt.ItemDataRole.UserRole, str(item_path))

                if item_path.is_dir():
                    item.setIcon(0, self.style().standardIcon(self.style().StandardPixmap.SP_DirIcon))
                    # Watch subdirectories
                    self.watcher.addPath(str(item_path))
                    # Recurse into subdirectory
                    self._build_tree_recursive(item_path, item)
                else:
                    # Set file icon based on extension
                    if item_path.suffix == '.py':
                        item.setIcon(0, QIcon.fromTheme("python", self.style().standardIcon(
                            self.style().StandardPixmap.SP_FileIcon)))  # Placeholder
                    else:
                        item.setIcon(0, self.style().standardIcon(self.style().StandardPixmap.SP_FileIcon))

        except PermissionError:
            pass  # Skip directories we can't access

    @Slot()
    def refresh_tree(self):
        """Refresh the entire tree"""
        if self.project_root:
            self.load_project(str(self.project_root))

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item clicks"""
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        path_obj = Path(file_path)

        if path_obj.is_file():
            self.file_selected.emit(file_path)
        else:
            self.folder_selected.emit(file_path)

    def _show_context_menu(self, position):
        """Show context menu for tree items"""
        item = self.itemAt(position)
        if not item:
            return

        menu = QMenu(self)

        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_tree)
        menu.addAction(refresh_action)

        if item.data(0, Qt.ItemDataRole.UserRole):
            path_obj = Path(item.data(0, Qt.ItemDataRole.UserRole))
            if path_obj.is_dir():
                expand_action = QAction("Expand All", self)
                expand_action.triggered.connect(lambda: self.expandItem(item))
                menu.addAction(expand_action)

        menu.exec(self.mapToGlobal(position))


class CodeViewerWindow(QMainWindow):
    """
    Professional Code Viewer/IDE Window for AvA workflow
    SINGLE RESPONSIBILITY: Display and edit generated code files
    """

    file_changed = Signal(str, str)  # file_path, content

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA - Code Viewer & IDE")
        self.setGeometry(200, 100, 1200, 800)

        # State
        self.open_files = {}  # file_path -> CodeEditor widget
        self.current_project = None

        self._init_ui()
        self._apply_theme()
        self._create_menus()
        self._connect_signals()

    def _init_ui(self):
        """Initialize the UI components"""
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - File tree
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        self.file_tree = FileTree()
        left_layout.addWidget(self.file_tree)

        # Right panel - Code editor and terminal in tabs
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # Tab widget for multiple editors and the terminal
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabsClosable(True)
        self.main_tabs.tabCloseRequested.connect(self._close_tab)

        # Add the Interactive Terminal as the first, permanent tab
        self.interactive_terminal = InteractiveTerminal()
        self.main_tabs.addTab(self.interactive_terminal, "📟 Terminal")
        # Make the terminal tab not closable
        self.main_tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)

        right_layout.addWidget(self.main_tabs)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])  # Adjust initial sizes

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _apply_theme(self):
        """Apply dark theme to the window"""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {Colors.SECONDARY_BG};
                color: #cccccc;
            }}
            QSplitter::handle {{
                background-color: {Colors.BORDER_DEFAULT};
            }}
            QTabWidget::pane {{
                border-top: 2px solid {Colors.BORDER_ACCENT};
            }}
            QTabBar::tab {{
                background: {Colors.ELEVATED_BG};
                color: {Colors.TEXT_SECONDARY};
                padding: 8px 16px;
                margin-right: 1px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background: {Colors.PRIMARY_BG};
                color: {Colors.TEXT_PRIMARY};
                border-color: {Colors.BORDER_ACCENT};
                border-bottom: 2px solid {Colors.PRIMARY_BG}; /* Overlap pane border */
            }}
            QTabBar::tab:hover {{
                background: {Colors.HOVER_BG};
            }}
            QTabBar::close-button {{
                image: url(none); /* Hide default icon, or use custom */
                subcontrol-position: right;
            }}
            QTabBar::close-button:hover {{
                background: {Colors.HOVER_BG};
            }}
        """)

    def _create_menus(self):
        pass  # For now, keeping it simple. Can add later.

    def _connect_signals(self):
        """Connect component signals"""
        self.file_tree.file_selected.connect(self._open_file)
        self.file_tree.folder_selected.connect(self._focus_folder)

    @Slot(str)
    def load_project(self, project_path: str):
        """Load a project directory - main entry point for workflow integration"""
        self.current_project = project_path
        self.file_tree.load_project(project_path)
        self.interactive_terminal.set_working_directory(project_path)
        self.setWindowTitle(f"AvA - IDE [{Path(project_path).name}]")

    @Slot(str)
    def _open_file(self, file_path: str):
        """Open file in a new tab or switch to existing tab"""
        file_path_obj = Path(file_path)

        # Check if file is already open
        if file_path in self.open_files:
            editor_widget = self.open_files[file_path]
            self.main_tabs.setCurrentWidget(editor_widget)
            return

        # Create new editor tab for the file
        editor = CodeEditor()
        editor.load_file(str(file_path_obj))
        editor.file_modified.connect(self._on_file_modified)

        tab_index = self.main_tabs.addTab(editor, file_path_obj.name)
        self.main_tabs.setCurrentIndex(tab_index)

        self.open_files[file_path] = editor

    def _focus_folder(self, folder_path: str):
        pass

    @Slot(int)
    def _close_tab(self, index: int):
        """Close an editor tab"""
        # Do not close the terminal tab (index 0)
        if index <= 0:
            return

        widget = self.main_tabs.widget(index)
        if isinstance(widget, CodeEditor):
            # Check for unsaved changes before closing
            if widget.is_modified:
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    f"File '{Path(widget.file_path).name}' has unsaved changes. Save before closing?",
                    QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
                )
                if reply == QMessageBox.StandardButton.Save:
                    widget.save_file()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return

            # Remove from open files dict and close tab
            if widget.file_path in self.open_files:
                del self.open_files[widget.file_path]

        self.main_tabs.removeTab(index)

    def _save_current_file(self):
        """Save the currently active editor file"""
        current_widget = self.main_tabs.currentWidget()
        if isinstance(current_widget, CodeEditor):
            if current_widget.save_file():
                current_index = self.main_tabs.currentIndex()
                file_name = Path(current_widget.file_path).name
                self.main_tabs.setTabText(current_index, file_name)

    def _on_file_modified(self, file_path: str):
        """Handle file modification by updating tab title"""
        if file_path in self.open_files:
            editor_widget = self.open_files[file_path]
            for i in range(self.main_tabs.count()):
                if self.main_tabs.widget(i) == editor_widget:
                    file_name = Path(file_path).name
                    self.main_tabs.setTabText(i, f"{file_name} *")
                    break

        content = editor_widget.toPlainText()
        self.file_changed.emit(file_path, content)

    @Slot()
    def refresh_project(self):
        """Refresh the project view"""
        if self.current_project:
            self.file_tree.refresh_tree()

    @Slot(str)
    def auto_open_file(self, file_path: str):
        """Automatically open a file, called by the workflow"""
        if Path(file_path).exists():
            self._open_file(file_path)