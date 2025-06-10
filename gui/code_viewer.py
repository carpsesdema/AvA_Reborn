# gui/code_viewer.py - AvA Code Viewer with Terminal

import re
from pathlib import Path
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction, QFont, QTextCharFormat, QColor, QSyntaxHighlighter, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QTextEdit, QTreeWidget, QTreeWidgetItem, QLabel, QMessageBox, QFileDialog
)

from gui.components import Colors, Typography
from gui.interactive_terminal import InteractiveTerminal


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Define colors
        keyword_color = QColor("#569CD6")  # Blue
        string_color = QColor("#CE9178")  # Orange
        comment_color = QColor("#6A9955")  # Green
        function_color = QColor("#DCDCAA")  # Yellow
        class_color = QColor("#4EC9B0")  # Cyan

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(keyword_color)
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'exec', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not',
            'or', 'pass', 'print', 'raise', 'return', 'try', 'while', 'yield', 'None', 'True', 'False'
        ]
        self.highlighting_rules = [(rf'\b{keyword}\b', keyword_format) for keyword in keywords]

        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(string_color)
        self.highlighting_rules.append((r'".*?"', string_format))
        self.highlighting_rules.append((r"'.*?'", string_format))

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(comment_color)
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((r'#.*', comment_format))

        # Function names
        function_format = QTextCharFormat()
        function_format.setForeground(function_color)
        self.highlighting_rules.append((r'\bdef\s+([_a-zA-Z][_a-zA-Z0-9]+)', function_format))

        # Class names
        class_format = QTextCharFormat()
        class_format.setForeground(class_color)
        class_format.setFontWeight(QFont.Weight.Bold)
        self.highlighting_rules.append((r'\bclass\s+([_a-zA-Z][_a-zA-Z0-9]+)', class_format))

        # Decorators
        decorator_format = QTextCharFormat()
        decorator_format.setForeground(QColor("#C586C0"))  # Purple
        self.highlighting_rules.append((r'@[a-zA-Z0-9_.]+', decorator_format))

        # Multi-line strings
        self.tri_single = (re.compile(r"'''"), 1)
        self.tri_double = (re.compile(r'"""'), 2)
        self.multiLineStringFormat = QTextCharFormat()
        self.multiLineStringFormat.setForeground(string_color)

    def highlightBlock(self, text):
        for pattern, fmt in self.highlighting_rules:
            for match in re.finditer(pattern, text):
                start, end = match.span(1 if '(' in pattern else 0)
                self.setFormat(start, end - start, fmt)

        self.setCurrentBlockState(0)

        in_multiline = self.previousBlockState()
        if not in_multiline in [self.tri_single[1], self.tri_double[1]]:
            # If not in a multiline string, check for the start of one
            for pattern, state in [self.tri_double, self.tri_single]:
                for match in pattern.finditer(text):
                    start = match.start()
                    end = pattern.search(text, start + 3)
                    if end:
                        length = (end.start() - start) + 3
                        self.setFormat(start, length, self.multiLineStringFormat)
                    else:
                        self.setCurrentBlockState(state)
                        self.setFormat(start, len(text) - start, self.multiLineStringFormat)
                        return  # Block is fully consumed


class CodeEditor(QTextEdit):
    file_modified = Signal(str)

    def __init__(self):
        super().__init__()
        self.file_path = None
        self.is_modified = False
        self.setFont(Typography.code())
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.PRIMARY_BG};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                padding: 8px;
                selection-background-color: {Colors.ACCENT_BLUE};
                selection-color: {Colors.PRIMARY_BG};
            }}
        """)
        self.highlighter = PythonSyntaxHighlighter(self.document())
        self.textChanged.connect(self._on_text_changed)

    def load_file(self, file_path: str):
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            self.setPlainText(content)
            self.file_path = file_path
            self.document().setModified(False)
            self.is_modified = False
            self.moveCursor(QTextCursor.MoveOperation.Start)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file: {e}")

    def save_file(self) -> bool:
        if not self.file_path: return False
        try:
            Path(self.file_path).write_text(self.toPlainText(), encoding='utf-8')
            self.document().setModified(False)
            self.is_modified = False
            return True
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save file: {e}")
            return False

    def _on_text_changed(self):
        if not self.is_modified:
            self.is_modified = True
            self.document().setModified(True)
            self.file_modified.emit(self.file_path)


class FileTree(QTreeWidget):
    file_selected = Signal(str)
    folder_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setStyleSheet(f"""
            QTreeWidget {{
                background: {Colors.PRIMARY_BG};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                outline: none;
                selection-background-color: {Colors.ACCENT_BLUE};
                selection-color: {Colors.PRIMARY_BG};
            }}
            QTreeWidget::item {{
                padding: 4px 8px;
                border: none;
            }}
            QTreeWidget::item:hover {{
                background: {Colors.HOVER_BG};
            }}
            QTreeWidget::item:selected {{
                background: {Colors.ACCENT_BLUE};
            }}
        """)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    def load_project(self, project_path: str):
        self.clear()
        root_path = Path(project_path)
        if not root_path.exists():
            return

        root_item = QTreeWidgetItem([root_path.name])
        root_item.setData(0, Qt.ItemDataRole.UserRole, str(root_path))
        self.addTopLevelItem(root_item)
        self._populate_tree(root_item, root_path)
        root_item.setExpanded(True)

    def _populate_tree(self, parent_item, path: Path):
        if not path.is_dir():
            return

        try:
            items = []
            # Directories first
            for item_path in sorted(path.iterdir()):
                if item_path.name.startswith('.') or item_path.name == '__pycache__':
                    continue

                if item_path.is_dir():
                    dir_item = QTreeWidgetItem([f"📁 {item_path.name}"])
                    dir_item.setData(0, Qt.ItemDataRole.UserRole, str(item_path))
                    parent_item.addChild(dir_item)
                    self._populate_tree(dir_item, item_path)
                    items.append((dir_item, True))
                else:
                    # File icons based on extension
                    icon = self._get_file_icon(item_path.suffix)
                    file_item = QTreeWidgetItem([f"{icon} {item_path.name}"])
                    file_item.setData(0, Qt.ItemDataRole.UserRole, str(item_path))
                    parent_item.addChild(file_item)
                    items.append((file_item, False))
        except PermissionError:
            pass

    def _get_file_icon(self, extension: str) -> str:
        icon_map = {
            '.py': '🐍',
            '.txt': '📄',
            '.md': '📝',
            '.json': '⚙️',
            '.yml': '⚙️',
            '.yaml': '⚙️',
            '.toml': '⚙️',
            '.ini': '⚙️',
            '.cfg': '⚙️',
            '.html': '🌐',
            '.css': '🎨',
            '.js': '📜',
            '.ts': '📜',
            '.sql': '🗃️',
            '.gitignore': '🚫',
            '.env': '🔑',
        }
        return icon_map.get(extension.lower(), '📄')

    @Slot(QTreeWidgetItem, int)
    def _on_item_double_clicked(self, item, column):
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if file_path and Path(file_path).is_file():
            self.file_selected.emit(file_path)


class CodeViewerWindow(QMainWindow):
    run_project_requested = Signal()
    file_changed = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.current_project = None
        self.open_files = {}

        self.setWindowTitle("AvA - Code Viewer")
        self.setGeometry(100, 100, 1400, 900)

        self._init_ui()
        self._create_menus()
        self._connect_signals()
        self._apply_theme()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        # Left panel (File Explorer)
        left_panel = self._create_file_explorer_panel()
        main_splitter.addWidget(left_panel)

        # Right splitter (Editor + Terminal)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)

        # Editor panel
        editor_panel = self._create_editor_panel()
        right_splitter.addWidget(editor_panel)

        # Terminal panel
        self.terminal = InteractiveTerminal()
        right_splitter.addWidget(self.terminal)

        # Set splitter ratios
        right_splitter.setSizes([600, 300])
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([300, 1100])

        layout.addWidget(main_splitter)

    def _create_file_explorer_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(400)
        panel.setStyleSheet(f"background: {Colors.SECONDARY_BG}; border-right: 1px solid {Colors.BORDER_DEFAULT};")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 0, 12, 0)
        header = QLabel("Project Explorer")
        header.setFont(Typography.heading_small())
        header.setStyleSheet("color: #cccccc; border: none; background: transparent;")
        header_layout.addWidget(header)
        header_layout.addStretch()

        self.file_tree = FileTree()

        layout.addLayout(header_layout)
        layout.addWidget(self.file_tree)
        return panel

    def _create_editor_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ border-top: 1px solid {Colors.BORDER_DEFAULT}; }}
            QTabBar::tab {{
                background: {Colors.SECONDARY_BG}; color: {Colors.TEXT_SECONDARY}; 
                padding: 8px 16px; border: 1px solid transparent;
                border-bottom: none;
            }}
            QTabBar::tab:hover {{ background: {Colors.HOVER_BG}; color: {Colors.TEXT_PRIMARY}; }}
            QTabBar::tab:selected {{ 
                background: {Colors.PRIMARY_BG}; color: {Colors.TEXT_PRIMARY}; 
                border-color: {Colors.BORDER_DEFAULT};
            }}
            QTabBar::close-button {{ /* Style if needed */ }}
        """)

        welcome_widget = QLabel("Welcome to the AvA Code Viewer!\n\nLoad a project or let AvA generate one.")
        welcome_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_widget.setFont(QFont("Segoe UI", 14))
        welcome_widget.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self.tab_widget.addTab(welcome_widget, "Welcome")

        layout.addWidget(self.tab_widget)
        return panel

    def _apply_theme(self):
        self.setStyleSheet(
            f"QMainWindow, QWidget {{ background-color: {Colors.SECONDARY_BG}; color: {Colors.TEXT_PRIMARY}; }}")

    def _create_menus(self):
        menubar = self.menuBar()
        menubar.setStyleSheet(f"background-color: {Colors.SECONDARY_BG};")
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Project...", self)
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
        self.file_tree.file_selected.connect(self._open_file_in_new_tab)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.terminal.force_run_requested.connect(self.run_project_requested.emit)

    @Slot(str)
    def load_project(self, project_path: str):
        self.current_project = project_path
        self.file_tree.load_project(project_path)
        self.terminal.set_working_directory(project_path)
        self.terminal.clear_terminal()
        self.terminal.append_system_message(f"Project '{Path(project_path).name}' loaded.")
        # Remove welcome tab if it's still there
        if self.tab_widget.tabText(0) == "Welcome":
            self.tab_widget.removeTab(0)
        self.setWindowTitle(f"AvA - Code Viewer [{Path(project_path).name}]")

    @Slot()
    def _open_project_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self.load_project(folder)

    @Slot(str)
    def _open_file_in_new_tab(self, file_path: str):
        file_path_str = str(Path(file_path))
        # Check if file is already open
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditor) and widget.file_path == file_path_str:
                self.tab_widget.setCurrentIndex(i)
                return

        # Remove welcome tab if it's the only tab
        if self.tab_widget.count() == 1 and not isinstance(self.tab_widget.widget(0), CodeEditor):
            self.tab_widget.removeTab(0)

        # Create new editor
        editor = CodeEditor()
        editor.load_file(file_path_str)
        editor.file_modified.connect(self._on_file_modified)

        tab_index = self.tab_widget.addTab(editor, Path(file_path_str).name)
        self.tab_widget.setTabToolTip(tab_index, file_path_str)
        self.tab_widget.setCurrentIndex(tab_index)
        self.open_files[file_path_str] = editor

    @Slot(int)
    def _close_tab(self, index: int):
        if index < 0 or index >= self.tab_widget.count():
            return

        widget = self.tab_widget.widget(index)
        if isinstance(widget, CodeEditor):
            if widget.document().isModified():
                reply = QMessageBox.question(
                    self, "Unsaved Changes", f"Save changes?",
                    QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
                )
                if reply == QMessageBox.StandardButton.Save:
                    widget.save_file()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return

            if widget.file_path in self.open_files:
                del self.open_files[widget.file_path]

        self.tab_widget.removeTab(index)

    @Slot()
    def _save_current_file(self):
        widget = self.tab_widget.currentWidget()
        if isinstance(widget, CodeEditor) and widget.save_file():
            self.tab_widget.setTabText(self.tab_widget.currentIndex(), Path(widget.file_path).name)

    @Slot(str)
    def _on_file_modified(self, file_path: str):
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditor) and widget.file_path == file_path:
                self.tab_widget.setTabText(i, f"{Path(file_path).name} *")
                break

        if file_path in self.open_files:
            self.file_changed.emit(file_path, self.open_files[file_path].toPlainText())

    @Slot(str)
    def auto_open_file(self, file_path: str):
        if Path(file_path).exists():
            self._open_file_in_new_tab(file_path)