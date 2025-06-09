# gui/code_viewer.py - Professional Code Viewer with Integrated Terminal & Modern UI

import re
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QFileSystemWatcher, QPoint
from PySide6.QtGui import QFont, QAction, QIcon, QSyntaxHighlighter, QTextCharFormat, QColor, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QLabel,
    QFileDialog, QMessageBox, QTabWidget, QFrame, QMenu, QHeaderView
)

from gui.components import Colors, Typography
# Import our UI components
from gui.interactive_terminal import InteractiveTerminal


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Professional Python syntax highlighter with multiline support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Define colors from our design system
        keyword_color = QColor(Colors.ACCENT_BLUE)
        string_color = QColor(Colors.ACCENT_ORANGE)
        comment_color = QColor(Colors.ACCENT_GREEN)
        function_color = QColor("#DCDCAA")  # A nice yellow for functions
        self_color = QColor("#9CDCFE")  # Light blue for 'self'
        class_color = QColor("#4EC9B0")  # Teal for class names

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(keyword_color)
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
            'not', 'or', 'pass', 'raise', 'return', 'try',
            'while', 'with', 'yield', 'None', 'True', 'False', 'async', 'await'
        ]
        self.highlighting_rules.extend([(r'\b' + word + r'\b', keyword_format) for word in keywords])

        # 'self' keyword
        self_format = QTextCharFormat()
        self_format.setForeground(self_color)
        self.highlighting_rules.append((r'\bself\b', self_format))

        # Strings
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
            Path(self.file_path).write_text(self.toPlainText(), encoding='utf-f8')
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
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setStyleSheet(f"""
            QTreeWidget {{
                background: {Colors.PRIMARY_BG};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                outline: none;
            }}
            QTreeView::branch {{
                background: transparent;
            }}
            QTreeWidget::item {{ padding: 6px 4px; border-radius: 4px; }}
            QTreeWidget::item:hover {{ background: {Colors.HOVER_BG}; }}
            QTreeWidget::item:selected {{ background: {Colors.ACCENT_BLUE}; color: white; }}
        """)
        self.itemClicked.connect(self._on_item_clicked)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.watcher = QFileSystemWatcher()
        self.watcher.directoryChanged.connect(self.refresh_tree)
        self.project_root = None

    def load_project(self, project_path: str):
        if self.project_root: self.watcher.removePath(str(self.project_root))
        self.project_root = Path(project_path)
        if not self.project_root.exists(): self.project_root.mkdir(parents=True, exist_ok=True)
        self.watcher.addPath(str(self.project_root))
        self.clear()
        self._build_tree_recursive(self.project_root, self.invisibleRootItem())

    def _build_tree_recursive(self, path: Path, parent_item: QTreeWidgetItem):
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            for item_path in items:
                if item_path.name.startswith(('.', '__pycache__')): continue
                item = QTreeWidgetItem(parent_item, [item_path.name])
                item.setData(0, Qt.ItemDataRole.UserRole, str(item_path))
                if item_path.is_dir():
                    item.setIcon(0, QIcon.fromTheme("folder-open", self.style().standardIcon(
                        self.style().StandardPixmap.SP_DirOpenIcon)))
                    self.watcher.addPath(str(item_path))
                    self._build_tree_recursive(item_path, item)
                else:
                    icon = QIcon.fromTheme("text-x-python" if item_path.suffix == '.py' else "text-x-generic",
                                           self.style().standardIcon(self.style().StandardPixmap.SP_FileIcon))
                    item.setIcon(0, icon)
        except PermissionError:
            pass

    @Slot()
    def refresh_tree(self, path=""):
        if self.project_root: self.load_project(str(self.project_root))

    @Slot(QTreeWidgetItem, int)
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if Path(file_path).is_file():
            self.file_selected.emit(file_path)
        else:
            self.folder_selected.emit(file_path)

    @Slot(QPoint)
    def _show_context_menu(self, position):
        menu = QMenu(self)
        refresh_action = QAction("Refresh Tree", self)
        refresh_action.triggered.connect(self.refresh_tree)
        menu.addAction(refresh_action)
        menu.exec(self.mapToGlobal(position))


class CodeViewerWindow(QMainWindow):
    file_changed = Signal(str, str)
    run_project_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA - Code Viewer & IDE")
        self.setGeometry(200, 100, 1400, 900)
        self.open_files = {}
        self.current_project = None
        self._init_ui()
        self._apply_theme()
        self._create_menus()
        self._connect_signals()

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setStyleSheet(f"""
            QSplitter::handle {{ background-color: {Colors.BORDER_DEFAULT}; }}
            QSplitter::handle:horizontal {{ width: 2px; }}
            QSplitter::handle:vertical {{ height: 2px; }}
            QSplitter::handle:hover {{ background-color: {Colors.ACCENT_BLUE}; }}
        """)

        left_panel = self._create_left_panel()
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setStyleSheet(main_splitter.styleSheet())  # Inherit style

        editor_panel = self._create_editor_panel()
        self.terminal = InteractiveTerminal()

        right_splitter.addWidget(editor_panel)
        right_splitter.addWidget(self.terminal)
        right_splitter.setSizes([600, 250])

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([300, 1100])

        main_layout.addWidget(main_splitter)
        self.setCentralWidget(central_widget)

    def _create_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet(
            f"background-color: {Colors.SECONDARY_BG}; border-right: 1px solid {Colors.BORDER_DEFAULT};")
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
        if self.tab_widget.tabText(0) == "Welcome": self.tab_widget.removeTab(0)
        self.setWindowTitle(f"AvA - Code Viewer [{Path(project_path).name}]")

    @Slot()
    def _open_project_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder: self.load_project(folder)

    @Slot(str)
    def _open_file_in_new_tab(self, file_path: str):
        file_path_str = str(Path(file_path))
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditor) and widget.file_path == file_path_str:
                self.tab_widget.setCurrentIndex(i)
                return
        if self.tab_widget.count() == 1 and not isinstance(self.tab_widget.widget(0),
                                                           CodeEditor): self.tab_widget.removeTab(0)
        editor = CodeEditor()
        editor.load_file(file_path_str)
        editor.file_modified.connect(self._on_file_modified)
        tab_index = self.tab_widget.addTab(editor, Path(file_path_str).name)
        self.tab_widget.setTabToolTip(tab_index, file_path_str)
        self.tab_widget.setCurrentIndex(tab_index)
        self.open_files[file_path_str] = editor

    @Slot(int)
    def _close_tab(self, index: int):
        if index < 0 or index >= self.tab_widget.count(): return
        widget = self.tab_widget.widget(index)
        if isinstance(widget, CodeEditor):
            if widget.document().isModified():
                reply = QMessageBox.question(self, "Unsaved Changes", f"Save changes?",
                                             QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
                if reply == QMessageBox.StandardButton.Save:
                    widget.save_file()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return
            if widget.file_path in self.open_files: del self.open_files[widget.file_path]
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
        if file_path in self.open_files: self.file_changed.emit(file_path, self.open_files[file_path].toPlainText())

    @Slot(str)
    def auto_open_file(self, file_path: str):
        if Path(file_path).exists(): self._open_file_in_new_tab(file_path)