# gui/code_viewer.py - V2 with corrected styling and functionality

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QPushButton, QLabel,
    QFileDialog, QMessageBox, QTabWidget, QFrame, QHeaderView, QMenu
)
from PySide6.QtCore import Qt, Signal, Slot, QFileSystemWatcher
from PySide6.QtGui import QFont, QAction, QIcon, QSyntaxHighlighter, QTextCharFormat, QColor
import re

from gui.interactive_terminal import InteractiveTerminal
from gui.components import Colors, Typography  # <-- NEW: Import our design system


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Professional Python syntax highlighter"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(Colors.ACCENT_BLUE))
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
            'not', 'or', 'pass', 'raise', 'return', 'try',
            'while', 'with', 'yield', 'None', 'True', 'False', 'async', 'await'
        ]
        for word in keywords:
            pattern = r'\b' + word + r'\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))

        self_format = QTextCharFormat()
        self_format.setForeground(QColor("#9CDCFE"))
        self.highlighting_rules.append((re.compile(r'\bself\b'), self_format))

        string_format = QTextCharFormat()
        string_format.setForeground(QColor(Colors.ACCENT_ORANGE))
        self.highlighting_rules.append((re.compile(r'".*?"'), string_format))
        self.highlighting_rules.append((re.compile(r"'.*?'"), string_format))

        self.multiLineStringFormat = QTextCharFormat()
        self.multiLineStringFormat.setForeground(QColor(Colors.ACCENT_ORANGE))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(Colors.TEXT_MUTED))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((re.compile(r'#.*'), comment_format))

        function_format = QTextCharFormat()
        function_format.setForeground(QColor(Colors.ACCENT_PURPLE))
        self.highlighting_rules.append((re.compile(r'\bdef\s+([a-zA-Z_]\w*)'), function_format))

        class_format = QTextCharFormat()
        class_format.setForeground(QColor(Colors.ACCENT_GREEN))
        self.highlighting_rules.append((re.compile(r'\bclass\s+([a-zA-Z_]\w*)'), class_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                # Apply format only to the captured group if it exists (for def, class)
                if match.re.groups > 0:
                    self.setFormat(match.start(1), match.end(1) - match.start(1), format)
                else:
                    self.setFormat(match.start(), match.end() - match.start(), format)


class CodeEditor(QTextEdit):
    file_modified = Signal(str)

    def __init__(self):
        super().__init__()
        self.file_path = None
        self.is_modified = False
        self.setFont(Typography.code())
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.PRIMARY_BG};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER_MUTED};
                border-radius: 6px;
                padding: 8px;
                selection-background-color: {Colors.HOVER_BG};
            }}
        """)
        self.highlighter = PythonSyntaxHighlighter(self.document())
        self.textChanged.connect(self._on_text_changed)

    def load_file(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.setPlainText(content)
            self.file_path = file_path
            self.document().setModified(False)
            self.is_modified = False
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file: {e}")

    def save_file(self) -> bool:
        if not self.file_path: return False
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(self.toPlainText())
            self.document().setModified(False)
            self.is_modified = False
            return True
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save file: {e}")
            return False

    def _on_text_changed(self):
        if not self.document().isModified():
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
                background: {Colors.SECONDARY_BG};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                outline: none;
            }}
            QTreeWidget::item {{ padding: 6px; border-radius: 4px; }}
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
        self.expandAll()

    def _build_tree_recursive(self, path: Path, parent_item: QTreeWidgetItem):
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            for item_path in items:
                if item_path.name.startswith(('.', '__pycache__', 'venv')): continue
                item = QTreeWidgetItem(parent_item, [item_path.name])
                item.setData(0, Qt.ItemDataRole.UserRole, str(item_path))
                if item_path.is_dir():
                    item.setIcon(0, QIcon.fromTheme("folder",
                                                    self.style().standardIcon(self.style().StandardPixmap.SP_DirIcon)))
                    self.watcher.addPath(str(item_path))
                    self._build_tree_recursive(item_path, item)
                else:
                    item.setIcon(0, QIcon.fromTheme("text-x-generic",
                                                    self.style().standardIcon(self.style().StandardPixmap.SP_FileIcon)))
        except PermissionError:
            pass

    def refresh_tree(self, path=""):
        if self.project_root: self.load_project(str(self.project_root))

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if Path(file_path).is_file():
            self.file_selected.emit(file_path)
        else:
            self.folder_selected.emit(file_path)

    def _show_context_menu(self, position):
        item = self.itemAt(position)
        if not item: return
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
        self.setGeometry(200, 100, 1200, 800)
        self.open_files = {}
        self.current_project = None
        self._init_ui()
        self._apply_theme()
        self._connect_signals()

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = self._create_left_panel()
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        editor_panel = self._create_editor_panel()

        self.terminal = InteractiveTerminal()
        self.terminal.setObjectName("integratedTerminal")

        right_splitter.addWidget(editor_panel)
        right_splitter.addWidget(self.terminal)
        right_splitter.setSizes([600, 200])

        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([280, 920])

        main_layout.addWidget(main_splitter)
        self.setCentralWidget(central_widget)

    def _create_left_panel(self) -> QWidget:
        left_panel = QWidget()
        left_panel.setStyleSheet(f"background-color: {Colors.SECONDARY_BG};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        tree_header = QLabel("📁 Project Explorer")
        tree_header.setFont(Typography.heading_small())
        tree_header.setStyleSheet(f"color: {Colors.ACCENT_BLUE}; margin-bottom: 4px; padding-left: 4px;")

        self.file_tree = FileTree()
        left_layout.addWidget(tree_header)
        left_layout.addWidget(self.file_tree)
        return left_panel

    def _create_editor_panel(self) -> QWidget:
        editor_panel = QWidget()
        editor_layout = QVBoxLayout(editor_panel)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {Colors.PRIMARY_BG}; }}
            QTabBar::tab {{
                background: {Colors.SECONDARY_BG}; color: {Colors.TEXT_SECONDARY}; 
                padding: 8px 16px; border: 1px solid {Colors.BORDER_MUTED};
                border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px;
            }}
            QTabBar::tab:hover {{ background: {Colors.HOVER_BG}; }}
            QTabBar::tab:selected {{ 
                background: {Colors.PRIMARY_BG}; color: {Colors.TEXT_PRIMARY}; 
                border-color: {Colors.BORDER_DEFAULT};
            }}
        """)

        welcome_widget = QLabel("Welcome to the AvA Code Viewer!\n\nLoad a project or let AvA generate one.")
        welcome_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_widget.setFont(Typography.body())
        welcome_widget.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self.tab_widget.addTab(welcome_widget, "Welcome")

        editor_layout.addWidget(self.tab_widget)
        return editor_panel

    def _apply_theme(self):
        self.setStyleSheet(
            f"QMainWindow, QWidget {{ background-color: {Colors.PRIMARY_BG}; color: {Colors.TEXT_PRIMARY}; }}")

    def _connect_signals(self):
        self.file_tree.file_selected.connect(self._open_file_in_new_tab)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.terminal.force_run_requested.connect(self.run_project_requested.emit)

    @Slot(str)
    def load_project(self, project_path: str):
        self.current_project = project_path
        self.file_tree.load_project(project_path)
        # --- MODIFIED: This is the critical fix ---
        self.terminal.set_working_directory(project_path)
        self.terminal.set_force_run_enabled(True)
        self.terminal.clear_terminal()
        self.terminal.append_system_message(f"Project '{Path(project_path).name}' loaded.")

        if self.tab_widget.tabText(0) == "Welcome": self.tab_widget.removeTab(0)
        self.setWindowTitle(f"AvA - Code Viewer [{Path(project_path).name}]")

    @Slot(str)
    def _open_file_in_new_tab(self, file_path: str):
        file_path = str(Path(file_path))
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditor) and widget.file_path == file_path:
                self.tab_widget.setCurrentIndex(i)
                return
        if self.tab_widget.count() == 1 and not isinstance(self.tab_widget.widget(0), CodeEditor):
            self.tab_widget.removeTab(0)
        editor = CodeEditor()
        editor.load_file(file_path)
        editor.file_modified.connect(self._on_file_modified)
        file_name = Path(file_path).name
        tab_index = self.tab_widget.addTab(editor, file_name)
        self.tab_widget.setTabToolTip(tab_index, file_path)
        self.tab_widget.setCurrentIndex(tab_index)
        self.open_files[file_path] = editor

    @Slot(int)
    def _close_tab(self, index: int):
        if index < 0 or index >= self.tab_widget.count(): return
        widget = self.tab_widget.widget(index)
        if isinstance(widget, CodeEditor):
            if widget.document().isModified():
                reply = QMessageBox.question(self, "Unsaved Changes", f"Save changes to {Path(widget.file_path).name}?",
                                             QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
                if reply == QMessageBox.StandardButton.Save:
                    widget.save_file()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return
            if widget.file_path in self.open_files: del self.open_files[widget.file_path]
        self.tab_widget.removeTab(index)

    @Slot(str)
    def _on_file_modified(self, file_path: str):
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, CodeEditor) and widget.file_path == file_path:
                file_name = Path(file_path).name
                self.tab_widget.setTabText(i, f"{file_name} *")
                break
        if file_path in self.open_files:
            self.file_changed.emit(file_path, self.open_files[file_path].toPlainText())

    @Slot(str)
    def auto_open_file(self, file_path: str):
        if Path(file_path).exists():
            self._open_file_in_new_tab(file_path)