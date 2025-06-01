# code_viewer.py

from PySide6.QtWidgets import (
    QWidget, QTreeView, QSplitter, QTextEdit, QVBoxLayout, QFileSystemModel, QMainWindow
)
from PySide6.QtCore import Qt

class CodeViewer(QWidget):
    """
    Dark-themed code editor with file tree on left and code editor on right.
    Supports loading files and tracking changes. Future: diff viewer integration.
    """
    def __init__(self, root_dir: str = "."):  # default to project root
        super().__init__()
        self.root_dir = root_dir
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # File tree model
        self.model = QFileSystemModel()
        self.model.setRootPath(self.root_dir)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(self.root_dir))
        self.tree.clicked.connect(self._on_file_selected)

        # Code editor area
        self.editor = QTextEdit()
        self.editor.setReadOnly(False)
        self.editor.setStyleSheet(
            "background-color: #1e1e1e; color: #c6c6c6; font-family: 'JetBrains Mono';"
        )

        splitter.addWidget(self.tree)
        splitter.addWidget(self.editor)
        splitter.setSizes([200, 800])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _on_file_selected(self, index):
        path = self.model.filePath(index)
        if path.endswith('.py'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.editor.setPlainText(content)
            except Exception:
                self.editor.setPlainText("Unable to load file.")

class CodeViewerWindow(QMainWindow):
    """
    Main window wrapper for CodeViewer so it can be displayed separately.
    """
    def __init__(self, root_dir: str = "."):  # default to project root
        super().__init__()
        self.setWindowTitle("Code Viewer")
        self.resize(1000, 800)
        self.viewer = CodeViewer(root_dir=root_dir)
        self.setCentralWidget(self.viewer)
