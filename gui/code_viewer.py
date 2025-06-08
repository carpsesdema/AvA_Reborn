# windows/code_viewer.py - Professional Code Viewer/IDE Window

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QPushButton, QLabel,
    QFileDialog, QMessageBox, QTabWidget, QFrame, QToolBar, QMenuBar,
    QMenu, QApplication, QHeaderView
)
from PySide6.QtCore import Qt, Signal, Slot, QFileSystemWatcher
from PySide6.QtGui import QFont, QAction, QIcon, QSyntaxHighlighter, QTextCharFormat, QColor
import re


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
                selection-background-color: #0078d4;
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
                if item_path.name.startswith('.'):
                    continue  # Skip hidden files

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
                        item.setIcon(0, self.style().standardIcon(self.style().StandardPixmap.SP_FileIcon))
                    else:
                        item.setIcon(0, self.style().standardIcon(self.style().StandardPixmap.SP_FileIcon))

        except PermissionError:
            pass  # Skip directories we can't access

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

        # Add actions
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
        self.setWindowTitle("AvA - Code Viewer & Editor")
        self.setGeometry(200, 100, 1200, 800)

        # State
        self.open_files = {}  # file_path -> CodeEditor
        self.current_project = None

        self._init_ui()
        self._apply_theme()
        self._create_menus()
        self._connect_signals()

    def _init_ui(self):
        """Initialize the UI components"""
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - File tree
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.Shape.Box)
        left_panel.setStyleSheet("""
            QFrame {
                background: #252526;
                border: 1px solid #3e3e42;
                border-radius: 6px;
            }
        """)
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(400)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(8, 8, 8, 8)

        # File tree header
        tree_header = QLabel("📁 Project Explorer")
        tree_header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        tree_header.setStyleSheet("color: #0078d4; margin-bottom: 8px;")

        # File tree
        self.file_tree = FileTree()

        # Tree controls
        tree_controls = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setToolTip("Refresh file tree")
        self.refresh_btn.setMaximumWidth(40)
        self.refresh_btn.clicked.connect(self.file_tree.refresh_tree)

        self.open_project_btn = QPushButton("📂 Open Project")
        self.open_project_btn.clicked.connect(self._open_project_dialog)

        tree_controls.addWidget(self.refresh_btn)
        tree_controls.addWidget(self.open_project_btn)

        left_layout.addWidget(tree_header)
        left_layout.addWidget(self.file_tree)
        left_layout.addLayout(tree_controls)
        left_panel.setLayout(left_layout)

        # Right panel - Code editor with tabs
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.Shape.Box)
        right_panel.setStyleSheet("""
            QFrame {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 6px;
            }
        """)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(8, 8, 8, 8)

        # Editor header
        editor_header = QHBoxLayout()
        editor_title = QLabel("📝 Code Editor")
        editor_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        editor_title.setStyleSheet("color: #0078d4;")

        # Save button
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self._save_current_file)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #106ebe;
            }
        """)

        editor_header.addWidget(editor_title)
        editor_header.addStretch()
        editor_header.addWidget(self.save_btn)

        # Tab widget for multiple files
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d30;
                color: #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #0078d4;
                color: white;
            }
            QTabBar::tab:hover {
                background: #3e3e42;
            }
        """)

        # Welcome message
        self.welcome_widget = QWidget()
        welcome_layout = QVBoxLayout()
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        welcome_label = QLabel("Welcome to AvA Code Viewer")
        welcome_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        welcome_label.setStyleSheet("color: #0078d4; margin: 20px;")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        instructions = QLabel("""
        • Open a project folder to start viewing files
        • Generated files will appear here automatically
        • Click files in the tree to open them
        • Multiple files can be open in tabs
        """)
        instructions.setStyleSheet("color: #cccccc; margin: 20px; line-height: 1.6;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)

        welcome_layout.addWidget(welcome_label)
        welcome_layout.addWidget(instructions)
        self.welcome_widget.setLayout(welcome_layout)

        self.tab_widget.addTab(self.welcome_widget, "Welcome")

        right_layout.addLayout(editor_header)
        right_layout.addWidget(self.tab_widget)
        right_panel.setLayout(right_layout)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])  # Give more space to editor

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _apply_theme(self):
        """Apply dark theme to the window"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QPushButton {
                background: #2d2d30;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #cccccc;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3e3e42;
                border-color: #0078d4;
            }
        """)

    def _create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Project...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_project_dialog)
        file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_current_file)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        close_action = QAction("Close Tab", self)
        close_action.setShortcut("Ctrl+W")
        close_action.triggered.connect(lambda: self._close_tab(self.tab_widget.currentIndex()))
        file_menu.addAction(close_action)

    def _connect_signals(self):
        """Connect component signals"""
        self.file_tree.file_selected.connect(self._open_file)
        self.file_tree.folder_selected.connect(self._focus_folder)

    def load_project(self, project_path: str):
        """Load a project directory - main entry point for workflow integration"""
        self.current_project = project_path
        self.file_tree.load_project(project_path)

        # Remove welcome tab if it exists
        if self.tab_widget.count() > 0 and self.tab_widget.tabText(0) == "Welcome":
            self.tab_widget.removeTab(0)

        self.setWindowTitle(f"AvA - Code Viewer [{Path(project_path).name}]")

    def _open_project_dialog(self):
        """Open project selection dialog"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Folder",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            self.load_project(folder)

    def _open_file(self, file_path: str):
        """Open file in a new tab or switch to existing tab"""
        file_path = str(Path(file_path))

        # Check if file is already open
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditor) and widget.file_path == file_path:
                self.tab_widget.setCurrentIndex(i)
                return

        # Create new editor tab
        editor = CodeEditor()
        editor.load_file(file_path)
        editor.file_modified.connect(self._on_file_modified)

        # Add tab with file name
        file_name = Path(file_path).name
        tab_index = self.tab_widget.addTab(editor, file_name)
        self.tab_widget.setCurrentIndex(tab_index)

        # Store reference
        self.open_files[file_path] = editor

    def _focus_folder(self, folder_path: str):
        """Handle folder selection"""
        # Could expand folder or show folder contents
        pass

    def _close_tab(self, index: int):
        """Close a tab"""
        if index < 0 or index >= self.tab_widget.count():
            return

        widget = self.tab_widget.widget(index)
        if isinstance(widget, CodeEditor):
            # Check if file is modified
            if widget.is_modified:
                reply = QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    f"File '{Path(widget.file_path).name}' has unsaved changes. Save before closing?",
                    QMessageBox.StandardButton.Save |
                    QMessageBox.StandardButton.Discard |
                    QMessageBox.StandardButton.Cancel
                )

                if reply == QMessageBox.StandardButton.Save:
                    widget.save_file()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return

            # Remove from open files
            if widget.file_path in self.open_files:
                del self.open_files[widget.file_path]

        self.tab_widget.removeTab(index)

    def _save_current_file(self):
        """Save the currently active file"""
        current_widget = self.tab_widget.currentWidget()
        if isinstance(current_widget, CodeEditor):
            if current_widget.save_file():
                # Update tab title to remove modified indicator
                current_index = self.tab_widget.currentIndex()
                file_name = Path(current_widget.file_path).name
                self.tab_widget.setTabText(current_index, file_name)

    def _on_file_modified(self, file_path: str):
        """Handle file modification"""
        # Update tab title to show modified state
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditor) and widget.file_path == file_path:
                file_name = Path(file_path).name
                self.tab_widget.setTabText(i, f"{file_name} *")
                break

        # Emit signal for external listeners
        if file_path in self.open_files:
            content = self.open_files[file_path].toPlainText()
            self.file_changed.emit(file_path, content)

    def refresh_project(self):
        """Refresh the project view - called when new files are generated"""
        if self.current_project:
            self.file_tree.refresh_tree()

    def auto_open_file(self, file_path: str):
        """Automatically open a file (called by workflow when files are generated)"""
        if Path(file_path).exists():
            self._open_file(file_path)
            # Switch to the new file
            for i in range(self.tab_widget.count()):
                widget = self.tab_widget.widget(i)
                if isinstance(widget, CodeEditor) and widget.file_path == file_path:
                    self.tab_widget.setCurrentIndex(i)
                    break

    def get_open_files(self) -> list:
        """Get list of currently open file paths"""
        return list(self.open_files.keys())

    def close_all_files(self):
        """Close all open files"""
        while self.tab_widget.count() > 0:
            self._close_tab(0)